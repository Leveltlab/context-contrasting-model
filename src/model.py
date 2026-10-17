"""One pyramidal cell and one PV cell, with three image inputs."""

import torch


class CCNeuron:
    def __init__(self, parameters):
        self.parameters = parameters
        torch.manual_seed(parameters["seed"])
        for name in ("w_ff", "w_fb", "w_lat", "w_pv_lat", "W_pv"):
            value = torch.tensor(parameters[name], dtype=torch.float32)
            torch.randn(1, value.numel())
            setattr(self, name, value)
        self.receives_context = torch.tensor(parameters["receives_context"])
        self.w_fb *= self.receives_context
        self.W_pv = self.W_pv.reshape(1, 3)
        self.accumulator = torch.tensor(0.)
        self.reset()

    def reset(self):
        self.y = torch.tensor(0.)
        self.p = torch.zeros(1)
        self.adaptation = torch.tensor(0.)
        # The slow activity accumulator is a learning state, not a firing rate.

    @torch.no_grad()
    def step(self, x, c, learn=False, silence_pv=False):
        q = self.parameters
        y_previous = self.y
        pv_drive = (self.W_pv @ x).reshape(-1) + y_previous * self.w_pv_lat
        pv_noise = (torch.randn(1, 1) * q["pv_noise_sigma"]).squeeze()
        self.p = (1 - q["pv_decay"]) * self.p + q["pv_decay"] * (pv_drive + pv_noise).clamp(0, 1)
        adaptation_rate = q["pyc_decay"] * 0.2
        self.adaptation = (1 - adaptation_rate) * self.adaptation + adaptation_rate * y_previous

        ff = torch.dot(self.w_ff, x)
        fb = torch.dot(self.w_fb, c * self.receives_context)
        lateral = torch.dot(self.w_lat, self.p) * (not silence_pv)
        drive = (fb - q["apical_drive_threshold"]).clamp(min=0)
        gain = (1 + q["apical_gain_strength"] * torch.sigmoid(
            q["apical_gain_k"] * (fb - q["apical_gain_threshold"])
        ) - q["apical_gain_strength"] / 2).clamp(min=1)
        noise = (torch.randn(1, 1) * q["baseline_drive_sigma"]).squeeze()
        voltage = ((gain * ff + drive) / (1 + q["divisive_gain"] * lateral)
                  + noise - self.adaptation)
        self.y = ((1 - q["pyc_decay"]) * self.y + q["pyc_decay"] * voltage.clamp(0, 1))

        if learn:
            rate = q["pyc_decay"] * q["ff_accumulator_alpha_factor"]
            self.accumulator = (1 - rate) * self.accumulator + rate * ff
            scale = q["ff_accumulator_scale"] * self.accumulator ** q["ff_accumulator_power"]
            self.w_ff += -q["lr_ff"] * scale * (self.y * x) * self.w_ff
            self.w_fb += q["lr_fb"] * (c * self.receives_context) * (1 - self.w_fb)
            self.w_lat += q["lr_lat"] * (self.y * self.p) * (1 - self.w_lat)
            self.w_ff.clamp_(min=0)
            self.w_fb.clamp_(min=0)
            self.w_fb *= self.receives_context
            self.w_lat.clamp_(min=0)
        return self.y.item()
