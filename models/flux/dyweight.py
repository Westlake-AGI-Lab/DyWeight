import pdb
import torch
import numpy as np
import torch.utils
import torch.utils.checkpoint
import torch.nn as nn, torch.nn.functional as F

_iPNDM = {
    1: [1.0],                              # AB1  (Euler)
    2: [3/2, -1/2],                        # AB2
    3: [23/12, -16/12, 5/12],              # AB3
    4: [55/24, -59/24, 37/24, -9/24],      # AB4
    5: [1901/720, -1387/360, 109/30, -637/360, 251/720],
    6: [4277/1440, -2641/480, 4991/720, -3649/720, 959/480, -95/288],
}

class DyweightParams(nn.Module):
    def __init__(self, K: int, tscale_low: float = 0.5, tscale_high: float = 1.5,
                 init_mode: str = "ipndm", window_size: int = 0):
        super().__init__()
        self.K = K
        self.tscale_low, self.tscale_high = tscale_low, tscale_high
        self.window_size = int(window_size)
        self.theta = nn.Parameter(torch.zeros(K, K))
        tril = torch.tril(torch.ones(K, K, dtype=torch.bool))
        if self.window_size > 0:
            band = torch.triu(torch.ones(K, K, dtype=torch.bool),
                              diagonal=-(self.window_size - 1))
            win_mask = tril & band
            self.order = min(4, self.window_size)
        else:
            win_mask = tril
            self.order = 2
        self.register_buffer("tril_mask", tril)
        self.register_buffer("win_mask", win_mask)
        self.phi = nn.Parameter(torch.zeros(K))
        self.reset_parameters(init_mode=init_mode, rand_std=0.0)
        self.theta.register_hook(lambda g: g * self.win_mask if g is not None else g)

    def reset_parameters(self, init_mode: str = "ipndm", rand_std: float = 0.0):
        with torch.no_grad():
            self.theta.zero_()
            for i in range(self.K):
                if init_mode == "zeros":
                    pass
                elif init_mode == "euler":
                    self.theta[i, i] = 1.0
                elif init_mode == "uniform":
                    self.theta[i, :i+1] = 1.0 / float(i + 1)
                elif init_mode == "ipndm":
                    p = min(i + 1, self.order)
                    coeffs = torch.tensor(_iPNDM[p], dtype=self.theta.dtype, device=self.theta.device)
                    self.theta[i, i - p + 1: i + 1] = coeffs.flip(0)
                else:
                    raise ValueError(f"unknown init_mode: {init_mode}")

            if rand_std > 0:
                self.theta.add_(torch.randn_like(self.theta) * rand_std)
            self.theta.mul_(self.win_mask)
            self.phi.zero_()

    def weights_row(self, i: int, mode: str = "raw",):
        start = max(0, i - (self.window_size - 1))
        sl = slice(start, i + 1)
        if mode == "softmax":
            w = torch.softmax(self.theta[i, sl], dim=0)
        else:
            w = self.theta[i, sl]
        return (w, start)

    def t_scale(self, i: int) -> torch.Tensor:
        s = torch.sigmoid(self.phi[i])
        return self.tscale_low + (self.tscale_high - self.tscale_low) * s

    def project_inplace(self):
        with torch.no_grad():
            self.theta.mul_(self.win_mask)

class DyweightRunner(nn.Module):
    def __init__(self, transformer, dyweight_params):
        super().__init__()
        self.transformer = transformer
        self.dyweight_params = dyweight_params

    def forward(
        self, latents: torch.Tensor,
        timesteps,
        guidance,
        pooled_prompt_embeds, prompt_embeds,
        text_ids, image_ids,
        mode: str = "teacher",
    ) -> torch.Tensor:
        if mode == "student":
            return self._student_forward(
                latents=latents,
                sigmas=timesteps,
                guidance=guidance,
                pooled_prompt_embeds=pooled_prompt_embeds,
                prompt_embeds=prompt_embeds,
                text_ids=text_ids,
                image_ids=image_ids,
            )
        elif mode == "teacher":
            with torch.no_grad():
                return self._teacher_forward(
                    latents=latents,
                    sigmas=timesteps,
                    guidance=guidance,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    prompt_embeds=prompt_embeds,
                    text_ids=text_ids,
                    image_ids=image_ids,
                )
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
    def _teacher_forward(
        self, latents: torch.Tensor, sigmas, guidance,
        pooled_prompt_embeds, prompt_embeds, text_ids, image_ids,
    ) -> torch.Tensor:
        x = latents
        for k in range(len(sigmas) - 1):
            sigma_cur, sigma_next = sigmas[k], sigmas[k+1]
            t_in = torch.full((latents.shape[0],), sigma_cur, device=latents.device, dtype=latents.dtype)
            v = self.transformer(
                hidden_states=x,
                timestep=t_in,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=image_ids,
                return_dict=False,
            )[0]
            x = x + (sigma_next - sigma_cur) * v
        return x

    def _student_forward(
        self, latents: torch.Tensor, sigmas: torch.Tensor, guidance,
        pooled_prompt_embeds, prompt_embeds, text_ids, image_ids,
    ) -> torch.Tensor:
        
        def _one_step(x_, sigma_in, pooled_, prompt_, txt_, img_,):
            return self.transformer(
                hidden_states=x_, timestep=sigma_in, guidance=guidance,
                pooled_projections=pooled_, encoder_hidden_states=prompt_,
                txt_ids=txt_, img_ids=img_, return_dict=False,
            )[0]
        
        x = latents
        deriavatives = []
        sigma_true = sigmas[0]
        for k in range(len(sigmas) - 1):
            sigma_cur, sigma_next = sigma_true, sigmas[k + 1]
            scale_cur = self.dyweight_params.t_scale(k) # tensor
            t_in = (scale_cur*sigma_cur).expand(latents.shape[0]).to(latents.dtype)
            v = torch.utils.checkpoint.checkpoint(
                _one_step,
                x, t_in, pooled_prompt_embeds, prompt_embeds, text_ids, image_ids,
                use_reentrant=False,
            )
            deriavatives.append(v)
            
            w_k, start = self.dyweight_params.weights_row(k)
            D_seg = torch.stack(deriavatives[start: k+1], dim=0)
            v_hat = torch.tensordot(w_k.to(D_seg.dtype), D_seg, dims=([0], [0]))
            x = x + (sigma_next - sigma_cur) * v_hat
            sigma_true = sigma_cur + w_k.sum() * (sigma_next - sigma_cur) # !!! important
        return x


if __name__ == "__main__":
    dyweights = DyweightParams(K=8, init_mode="ipndm", window_size=4)