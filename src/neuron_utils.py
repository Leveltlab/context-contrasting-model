# author: Matúš Halák (@matushalak)
from typing import Iterable
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

class EMA(torch.nn.Module):
    '''
    EMA (Exponential Moving Average) = Discretized Leaky Integrator
        Alpha controls history dependence and stability (how many steps it takes to decay to baseline); 
        Low alpha (eg. 1e-4): 
            slower integration, more history dependence, 
            slower decay, takes 10000 steps to decay to baseline
        High alpha (eg. 1e-2): 
            faster integration, more current input dependence, 
            faster decay, takes 100 steps to decay to baseline
    
    If basline is provided, decay towards baseline in absence of input; 
        otherwise, decay towards 0.
    '''
    def __init__(self, shape:tuple, alpha:float = 0.1, baseline:torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.baseline = baseline if baseline is not None else torch.zeros(shape, requires_grad=False)
        self.ema = self.baseline.clone()
    
    @torch.no_grad()
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        self.ema = (1 - self.alpha) * self.ema + self.alpha * (x+self.baseline)
        return self.ema
    
    def reset_state(self):
        self.ema = self.baseline.clone()

class ExponentialMovingAverage(torch.nn.Module):
    """
    Wrapper around torch.optim.swa_utils.AveragedModel for model-weight EMA.
    Weights are frozen and not part of the student computation graph during backprop.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.99, use_buffers: bool = True):
        super().__init__()
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")

        self.decay = decay
        self.ema_model = AveragedModel(
            model,
            multi_avg_fn=get_ema_multi_avg_fn(decay),
            use_buffers=use_buffers,
        )
        self._freeze()

    def _freeze(self):
        self.training = False
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.training = False
        self.ema_model.eval()
        return self

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """Call once after each optimizer step."""
        self.ema_model.update_parameters(model)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.ema_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.ema_model.load_state_dict(state_dict, strict=strict)


class ThresholdReLU(torch.nn.Module):
    '''
    TODO: rewrite docstring to be accurate
    Thresholded ReLU activation function: f(x) = max(0, x - threshold)
    '''
    def __init__(self, threshold:float = 0.0, subtractive:bool = True, hasMax:bool = False, maxValue:float = 1.0):
        super().__init__()
        self.threshold = threshold
        self.subtractive = subtractive
        self.hasMax = hasMax
        self.maxValue = maxValue

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.subtractive:
            x = torch.clamp(x - self.threshold, min=0.0)
        else:
            x = (x > self.threshold).float() * x
        if self.hasMax:
            x = torch.clamp(x, max=self.maxValue)
        return x


class GainSigmoid(torch.nn.Module):
    '''
    Gain modulation by sigmoid function: f(x) = clip_min(1, 1 + gain * sigmoid(k*(x-threshold)) - gain/2).

    The raw form gives f(x) < 1 whenever the sigmoid argument is negative -- i.e.
    when x < threshold (including x=0 whenever threshold>0). That would let the
    apical compartment SUPPRESS somatic firing, which is biophysically wrong:
    apical drive can only AMPLIFY (or be neutral, factor 1). The clip to >=1
    enforces that invariant for any (threshold, k, gain) choice. Above threshold
    the function is unchanged (>=1 anyway), so this is a one-sided floor, not a
    re-shape.
    '''
    def __init__(self, gain:float = 1.0, k:float = 1.0, threshold:float = 0.0):
        super().__init__()
        self.gain = gain
        self.k = k
        self.threshold = threshold

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.clip(1 + self.gain * torch.sigmoid(self.k * (x - self.threshold)) - self.gain / 2, min=1.0)


def nonnegative(x:torch.Tensor)->torch.Tensor:
    '''
    Performs x'= max(0, x) elementwise, ensuring all synaptic weights are non-negative.
    '''
    return torch.clamp(x, min=0.0)

def nonpositive(x:torch.Tensor)->torch.Tensor:
    '''
    Performs x'= min(0, x) elementwise, ensuring all synaptic weights are non-positive.
    '''
    return torch.clamp(x, max=0.0)


def randn_reparam(size:tuple[int, ...], mu:float|Iterable, sigma:float|Iterable) -> torch.Tensor:
    '''
    Reparameterization trick for sampling from a normal distribution with mean mu and std sigma.
    
    The following cases are supported:
        1) Generating a single sample (size=()):
            - mu and sigma can be scalars (shape (1,)) or vectors of shape (n_features,)
            - sample will have shape of mu
        2) Generating a batch of samples (size=(n_samples,)):
            - mu and sigma must be vectors of shape (n_features,) (or scalars)
            - sample will have shape (n_samples, n_features)
        3) Generating a matrix of samples (size=(n_samples, n_features)):
            - mu and sigma must be scalars (shape (1,)) 
    
    In cases 1 and 2, sigma can also be a matrix of shape (n_features, n_features) 
    representing a full covariance matrix.    
    '''
    mu = torch.as_tensor(mu)
    sigma = torch.as_tensor(sigma)
    
    if len(size) == 0:
        size = (1,)

    mu_shape = mu.shape if len(mu.shape) > 0 else (1,)
    sigma_shape = sigma.shape if len(sigma.shape) > 0 else (1,)

    assert isinstance(size, tuple), "size must be a tuple of ints. For generating a single sample, use size=()."
    if len(size) == 1:
        z = torch.randn(*size, *mu_shape)
    else:
        assert (
            (mu_shape == (1,) or mu_shape == (size[1],))
            and (sigma_shape == (1,) or sigma_shape == (size[1],))
        ), "mu and sigma must be scalar if generating a random matrix."
        z = torch.randn(*size)
    if (len(sigma_shape) == 1 and sigma_shape[0] == 1):
        sigma = sigma * torch.eye(z.shape[-1])
    elif len(sigma_shape) == 1:
        sigma = torch.diag(sigma) 
    else:
        assert sigma_shape == (mu_shape[0], mu_shape[0]), "If sigma is a matrix, it must have shape (n_features, n_features)."
    
    # Reparametrization trick: sample z ~ N(0, I) and transform to desired distribution    
    return (mu + z @ sigma).squeeze()
