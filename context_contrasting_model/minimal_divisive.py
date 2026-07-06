# author: Matúš Halák (@matushalak)
import torch
import torch.nn as nn
from typing import Literal

from .neuron_utils import EMA, ThresholdReLU, GainSigmoid, nonnegative, randn_reparam

class CCNeuron(nn.Module):
    """
    Minimal contextual-contrasting model with:
      - one pyramidal neuron y (scalar),
      - one PV neuron p (scalar),
      - feedforward input x (size D),
      - contextual input c (size D).

    Dynamics:
      p = phi(w_pv^T x)
      y = phi(w_ff^T x + w_fb^T c - w_lat^T p)

    Local learning rules:
      dw_ff  ~ -ff_plasticity_scale * (y * x)     (anti-Hebbian)
      dw_fb  ~ (alpha / (y + alpha)) * (c)   (dampened-anti-Hebbian)
            OR y * c (Hebbian)
      dw_lat ~  (y * p)                           (Hebbian)
      dW_pv  ~  p x^T                             (Hebbian)
    """
    # using randn_reparam
    def __init__(
        self,
        n_features: int = 3,
        n_pv: int = 1,
        n_context: int = 3,
        activation: nn.Module | None = None,
        # Core continuous parameters with a lot of influence
        lr_ff: float = 0.01, # 1
        w_ff_init:dict = {'mu': [0.5, 0.5, 0.5], 'sigma': 1e-2}, # 2,3
        lr_fb: float = 0.01, # 4
        w_fb_init:dict = {'mu': [0.1, 0.1, 0.1], 'sigma': 1e-2}, # 5,6
        lr_lat: float = 0.01, # 7
        w_lat_init:dict = {'mu': [0.2], 'sigma': 1e-2}, # 8,9
        w_pv_lat_init:dict | None = None, # 10, 11
        lr_pv: float = 0.01, # 12
        W_pv_init:dict = {'mu': [0.1, 0.1, 0.1], 'sigma': [1e-2, 1e-2, 1e-2]}, # 13,14,15,16
        # Categorical parameters with a lot of influence
        receives_context:tuple[bool, bool ] = (True, True, True), # 17,18
        FFrule:Literal['anti-Hebbian', 'Hebbian'] = 'anti-Hebbian', # 18
        FBrule:Literal["dampened-anti-Hebbian", "Hebbian"] = "dampened-anti-Hebbian", # 20
        # Other hyperparameters (fixed between initial conditions)
        pyc_decay:float = 0.1,
        pv_decay:float = 0.25,
        apical_drive_threshold: float = 0.2,
        # Apical dendritic drive threshold nonlinearity. Default (subtractive=False,
        # hasMax=False) is the hard threshold (x>threshold)*x with no cap. Setting
        # apical_drive_subtractive=True switches to a shifted ReLU max(0, x-threshold)
        # so the feedback drive ramps continuously from 0 once it crosses threshold.
        apical_drive_subtractive: bool = False,
        apical_gain_strength: float = 2.0,
        apical_gain_k: float = 5.0,
        apical_gain_threshold: float = 0.0,
        baseline_drive_mu: float = 0.0,
        baseline_drive_sigma: float = 0.2,
        # Divisive-surround gain k: response drive is normalized by (1 + k*y_lat).
        # k>1 lets the bounded surround (w_lat<=1, single PV -> y_lat<=1) divide hard
        # enough to pull O-responder NO down toward the baseline floor (vertical),
        # while the baseline term stays OUTSIDE the division so it cannot overshoot
        # below baseline (Carandini-Heeger form; shunting clamps toward rest, the
        # subtractive adaptation `a` is the only thing that goes below baseline).
        divisive_gain: float = 1.0,
        pv_noise_sigma: float = 0.06,
        alpha: float = 1.0,
        weight_decay: float = 0.0,
        seed:int = 42,
        use_FF_connection:bool = True,
        FF_plasticity:bool = True,
        ff_plasticity_scale: float | list[float] | tuple[float, ...] = 1.0,
        use_FB_connection:bool = True,
        FB_plasticity:bool = True,
        use_lat_connection:bool = True,
        lat_plasticity:bool = True,
        use_pv_connection:bool = True,
        pv_plasticity:bool = True,
        use_pv_lat_connection:bool = True,
        pv_lat_plasticity:bool = True
    ):
        super().__init__()
        if alpha <= 0:
            raise ValueError("alpha must be > 0.")
        if weight_decay < 0 or weight_decay > 1:
            raise ValueError("weight_decay must be 0 <= wd <= 1.")

        torch.manual_seed(seed) # set random seed for weight initialization
        assert FFrule in ["anti-Hebbian", "Hebbian"], "FFrule must be either 'anti-Hebbian' or 'Hebbian'."
        self.FFrule = FFrule
        assert FBrule in ["dampened-anti-Hebbian", "Hebbian"], "FBrule must be either 'dampened-anti-Hebbian' or 'Hebbian'."
        self.FBrule = FBrule
        assert len(receives_context) == n_context, "receives_context must match n_context."
        self.receives_context = torch.tensor(receives_context, dtype=torch.bool)
        assert n_features == n_context
        if w_pv_lat_init is None:
            w_pv_lat_init = dict(w_lat_init)

        self.n_features = n_features
        self.n_pv = n_pv
        self.n_context = n_context
        # Cellular activation: ReLU-with-cap. ThresholdReLU(threshold=0,
        # subtractive=False, hasMax=True, maxValue=1.0) == max(0, x) clamped to 1.0,
        # i.e. the rate is non-negative and capped at the max firing rate.
        self.activation = activation if activation is not None else ThresholdReLU(
            threshold=0.0, subtractive=False, hasMax=True, maxValue=1.0
        )

        # Learnable weights updated manually via local rules
        self.w_ff = nonnegative(randn_reparam(size=(1,), **w_ff_init))
        self.w_fb = nonnegative(randn_reparam(size=(1,), **w_fb_init)) * self.receives_context
        self.w_lat = nonnegative(randn_reparam(size=(1,), **w_lat_init)).reshape(-1)
        self.w_pv_lat = nonnegative(randn_reparam(size=(1,), **w_pv_lat_init)).reshape(-1)
        self.W_pv = nonnegative(randn_reparam(size=(1,), **W_pv_init)).reshape(1, -1)
        
        # Hyperpatameters
        self.lr_ff = lr_ff
        self.lr_fb = lr_fb
        self.lr_lat = lr_lat
        self.lr_pv = lr_pv
        self.ff_plasticity_scale = torch.as_tensor(ff_plasticity_scale, dtype=self.w_ff.dtype)
        if self.ff_plasticity_scale.ndim == 0:
            self.ff_plasticity_scale = torch.full_like(self.w_ff, float(self.ff_plasticity_scale))
        elif self.ff_plasticity_scale.shape != self.w_ff.shape:
            raise ValueError("ff_plasticity_scale must be scalar or match n_features.")
        if torch.any(self.ff_plasticity_scale < 0):
            raise ValueError("ff_plasticity_scale must be nonnegative.")
        self.alpha = alpha
        self.weight_decay = weight_decay
        self.baseline_drive_mu = baseline_drive_mu
        self.baseline_drive_sigma = baseline_drive_sigma
        self.divisive_gain = divisive_gain
        self.pv_noise_sigma = pv_noise_sigma

        # State variables for PV and pyramidal neurons, implemented as EMAs.
        self.pv = EMA(shape=(n_pv,), alpha=pv_decay)
        self.pyramidal = EMA(shape=(), alpha=pyc_decay)
        self.adapt = EMA(shape=(), alpha=pyc_decay*0.2)

        self.threshold = ThresholdReLU(threshold=apical_drive_threshold, subtractive=apical_drive_subtractive, hasMax=False)
        self.sigmoid = GainSigmoid(gain=apical_gain_strength, k=apical_gain_k, threshold=apical_gain_threshold)

        # EMA of weights to implement decay towards baseline in absence of input (optional)
        # Baselines
        self.w_ff_baseline = self.w_ff.detach().clone()
        self.w_fb_baseline = self.w_fb.detach().clone()
        self.w_lat_baseline = self.w_lat.detach().clone()
        self.w_pv_lat_baseline = self.w_pv_lat.detach().clone()
        self.W_pv_baseline = self.W_pv.detach().clone()

        # Feedback specificity (decoding image identity with 60% accuracy)
        self.fb_specificity = torch.eye(self.n_features)*0.6 + (1 - torch.eye(self.n_features))*0.2
        self.pv_specificity = torch.eye(self.n_features)*0.8 + (1 - torch.eye(self.n_features))*0.1 

        # Ablation parameters
        self.use_FF_connection = use_FF_connection
        self.FF_plasticity = FF_plasticity
        self.use_FB_connection = use_FB_connection
        self.FB_plasticity = FB_plasticity
        self.use_lat_connection = use_lat_connection
        self.lat_plasticity = lat_plasticity
        self.use_pv_lat_connection = use_pv_lat_connection
        self.pv_lat_plasticity = pv_lat_plasticity 
        self.use_pv_connection = use_pv_connection
        self.pv_plasticity = pv_plasticity

    @torch.no_grad()
    def forward(self, x: torch.Tensor, c: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: bottom-up input, shape (n_features,)
            c: contextual input, shape (n_context,)
        Returns:
            y: pyramidal activity, scalar tensor shape ()
            p: PV activity, shape (n_pv,)
        """
        assert x.shape == (self.n_features,) and c.shape == (self.n_context,)

        # feedforward excitation to PV neurons
        pv_ff = (self.W_pv @ x).reshape(-1) * self.use_pv_connection
        y_t = self.pyramidal.ema
        pv_lat = y_t * self.w_pv_lat * self.use_pv_lat_connection
        p = self.pv(self.activation(
            pv_ff + pv_lat 
            + randn_reparam(size=self.pv.ema.shape, mu=0.0, sigma=self.pv_noise_sigma) # small random baseline input
            )) 
        
        a = self.adapt(self.pyramidal.ema) # update adaptation variable 

        y_ff  = torch.dot(self.w_ff, x) * self.use_FF_connection # feedforward excitation
        y_fb = torch.dot(self.w_fb, c * self.receives_context) * self.use_FB_connection # feedback excitation 
        y_lat = torch.dot(self.w_lat, p) * self.use_lat_connection # "lateral" inhibition 
        
        baseline_drive = randn_reparam(
            size=(),
            mu=self.baseline_drive_mu,
            sigma=self.baseline_drive_sigma,
        ) # small random baseline input
        
        basal = y_ff
        apical_drive = self.threshold(y_fb) * self.use_FB_connection
        apical_gain = self.sigmoid(y_fb * self.use_FB_connection)

        # Divisive normalization of the stimulus-evoked drive; baseline_drive stays
        # OUTSIDE the division (additive spontaneous floor) and `a` is subtractive.
        y_next = self.pyramidal(self.activation(
            (apical_gain * basal + apical_drive) / (1 + self.divisive_gain * y_lat)
            + baseline_drive - a
            ))
        
        y_next = torch.clamp(y_next, max=torch.tensor(1.0))
        
        return x, y_t, y_next, p, c

    # NOTE: ADDed SOFT-HEBBIAN boundedness 
    #   (w_max - w_t) for hebbian
    #   (w_t - w_min) for anti-hebbian
    @torch.no_grad()
    def update(self, 
               x_t: torch.Tensor, 
               y_t: torch.Tensor,
               y_next: torch.Tensor, 
               pv_t: torch.Tensor, 
               c_t: torch.Tensor):
        """
        One local update step using current inputs (x_t, c_t).
        Returns y_{t+1}, p_t as computed for this step.
        """
        dw_ff, dw_fb, dw_lat, dw_pv_lat, dw_W_pv = (torch.zeros_like(self.w_ff), 
                                                    torch.zeros_like(self.w_fb), 
                                                    torch.zeros_like(self.w_lat), 
                                                    torch.zeros_like(self.w_pv_lat), 
                                                    torch.zeros_like(self.W_pv))
        
        # 1) Anti-Hebbian update for w_ff
        if self.FF_plasticity and self.use_FF_connection and bool(torch.any(self.ff_plasticity_scale > 0)):
            match self.FFrule:
                case "anti-Hebbian":
                    dw_ff = - self.lr_ff * self.ff_plasticity_scale * (y_next * x_t) * (self.w_ff - 0.0)
                case "Hebbian":
                    dw_ff = self.lr_ff * self.ff_plasticity_scale * (y_next * x_t) * (1.0 - self.w_ff)

        # 2) Dampened-Hebbian update for w_fb
        damp = self.alpha / (y_next + self.alpha)

        if self.FB_plasticity and self.use_FB_connection:
            match self.FBrule:
                # contextual strengthening general (not only the experienced context, also novel)
                case "dampened-anti-Hebbian":
                    dw_fb = self.lr_fb * (damp * self.fb_specificity @ c_t * self.receives_context) * (1.0 - self.w_fb)
                case "Hebbian":
                    dw_fb = self.lr_fb * (y_next * self.fb_specificity @ c_t * self.receives_context) * (1.0 - self.w_fb)

        # 3) Hebbian update for w_lat and w_pv_lat
        if self.lat_plasticity and self.use_lat_connection:
            dw_lat = self.lr_lat * (y_next * pv_t) * (1.0 - self.w_lat)

        if self.pv_lat_plasticity and self.use_pv_lat_connection:
            dw_pv_lat = self.lr_lat * (y_t * pv_t) * (1.0 - self.w_pv_lat)

        # 4) Hebbian update for W_pv
        if self.pv_plasticity and self.use_pv_connection:
            dw_W_pv = self.lr_pv * (pv_t * self.pv_specificity @ x_t) * (1.0 - self.W_pv)

        # Apply updates
        self.w_ff += dw_ff
        self.w_fb += dw_fb
        self.w_lat += dw_lat
        self.w_pv_lat += dw_pv_lat
        self.W_pv += dw_W_pv
        
        # Weight Decay 
        if 0.0 < self.weight_decay < 1.0:
            # towards baseline (initial) weights
            self.w_ff -= (self.w_ff - self.w_ff_baseline) * self.weight_decay * self.ff_plasticity_scale * self.FF_plasticity * self.use_FF_connection
            self.w_fb -= (self.w_fb - self.w_fb_baseline) * self.weight_decay * self.FB_plasticity * self.use_FB_connection
            self.w_lat -= (self.w_lat - self.w_lat_baseline) * self.weight_decay * self.lat_plasticity * self.use_lat_connection
            self.w_pv_lat -= (self.w_pv_lat - self.w_pv_lat_baseline) * self.weight_decay * self.pv_lat_plasticity * self.use_pv_lat_connection
            self.W_pv -= (self.W_pv - self.W_pv_baseline) * self.weight_decay * self.pv_plasticity * self.use_pv_connection
            
            # towards zero
            # self.w_ff -= (self.w_ff) * self.weight_decay * self.FF_plasticity * self.use_FF_connection
            # self.w_fb -= (self.w_fb) * self.weight_decay * self.FB_plasticity * self.use_FB_connection
            # self.w_lat -= (self.w_lat) * self.weight_decay * self.lat_plasticity * self.use_lat_connection
            # self.w_pv_lat -= (self.w_pv_lat) * self.weight_decay * self.pv_lat_plasticity * self.use_pv_lat_connection
            # self.W_pv -= (self.W_pv) * self.weight_decay * self.pv_plasticity * self.use_pv_connection
        
        # Ensure non-negativity of weights
        self.w_ff = nonnegative(self.w_ff)
        self.w_fb = nonnegative(self.w_fb) * self.receives_context
        self.w_lat = nonnegative(self.w_lat)
        self.w_pv_lat = nonnegative(self.w_pv_lat)
        self.W_pv = nonnegative(self.W_pv)


    def _reset_state(self):
        self.pv.reset_state()
        self.pyramidal.reset_state()
        self.adapt.reset_state()

if __name__ == "__main__":
    # Example usage:
    model = CCNeuron()
    n_steps = 50
    X = torch.randn((n_steps, model.n_features)) # random input sequence
    C = torch.randn((n_steps, model.n_context)) # random context sequence

    for step in range(n_steps):
        x, y_t, y_next, p, c = model(X[step], C[step])
        update = model.update(x, y_t, y_next, p, c)
        print(f"Step {step}: y={y_next.item():.4f}, p={p.detach().numpy()}")
