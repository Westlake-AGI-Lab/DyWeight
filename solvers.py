import torch
from solver_utils import *
from torch_utils import distributed as dist

#----------------------------------------------------------------------------

def get_denoised(net, x, t, class_labels=None, condition=None, unconditional_condition=None):
    if hasattr(net, 'guidance_type'):
        denoised = net(x, t, condition=condition, unconditional_condition=unconditional_condition)
    else:
        denoised = net(x, t, class_labels=class_labels)
    return denoised

#----------------------------------------------------------------------------

def _get_t_scale_idx(i, afs):
    """Map step index i to t_scale index, accounting for AFS offset."""
    return (i - 1) if afs else i

def _apply_t_scale(t_cur, t_scale, i, afs):
    """Apply t_scale to t_cur at step i. Returns scaled (and abs'd) t_cur."""
    if t_scale is None:
        return t_cur.abs()
    idx = _get_t_scale_idx(i, afs)
    if afs and i == 0:
        return t_cur.abs()
    if t_scale.dim() == 1:
        if 0 <= idx < len(t_scale):
            return (t_scale[idx] * t_cur).abs()
    else:
        if 0 <= idx < t_scale.shape[1]:
            return (t_scale[:, idx].view(-1, 1, 1, 1) * t_cur).abs()
    return t_cur.abs()

#----------------------------------------------------------------------------
def dyweight_sampler(
    net,
    latents,
    class_labels=None,
    condition=None,
    unconditional_condition=None,
    num_steps=None,
    sigma_min=0.002,
    sigma_max=80,
    schedule_type='polynomial',
    schedule_rho=7,
    afs=False, 
    return_inters=False,
    hybrid_weights=None,
    t_scale=None,
    verbose=False,
    **kwargs
):
    """
    DyWeight sampler that combines historical derivatives with learned weights.
    Uses a lower triangular weight matrix to combine derivatives from previous steps.
    """
    assert hybrid_weights is not None, "DyWeight sampler requires hybrid_weights"
    is_batch = (hybrid_weights.dim() == 3)
    
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device,
                          schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)
    
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)] if return_inters else None

    derivatives = []
    nfe = 0
    t_next_true = t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        t_cur = t_next_true

        if afs and i == 0:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            scaled_t_cur = _apply_t_scale(t_cur, t_scale, i, afs)
            denoised = get_denoised(net, x_cur, scaled_t_cur, class_labels=class_labels, 
                                  condition=condition, unconditional_condition=unconditional_condition)
            nfe += 1
            d_cur = (x_cur - denoised) / (t_cur + 1e-8)

        derivatives.append(d_cur)
        
        t_cur = t_cur.reshape(-1, 1, 1, 1)
        t_next = t_next.reshape(-1, 1, 1, 1)

        w = hybrid_weights[:, i, :i+1] if is_batch else hybrid_weights[i, :i+1]
        
        weighted_derivative = torch.zeros_like(d_cur)
        for j in range(i + 1):
            if is_batch:
                weighted_derivative = weighted_derivative + w[:, j:j+1, None, None] * derivatives[j]
            else:
                weighted_derivative = weighted_derivative + w[j] * derivatives[j]
        
        x_next = x_cur + (t_next - t_cur) * weighted_derivative
        
        w_sum = w.sum(dim=-1) if is_batch else w.sum()
        t_next_true = t_cur + w_sum.view(-1, 1, 1, 1) * (t_next - t_cur)

        if verbose:
            w_display = w[0] if is_batch else w
            weights_str = "/".join([f"{v:.4f}" for v in w_display])
            prefix = "weights[sample_0]" if is_batch else "weights"
            dist.print0(f"Step {i}: {prefix}=[{weights_str}], NFE={nfe}")
                
            if i == num_steps - 2 and t_scale is not None:
                ts_display = t_scale[0] if t_scale.dim() > 1 else t_scale
                formatted = ", ".join([f"{s:.4f}" for s in ts_display.detach().cpu().tolist()])
                ts_prefix = "t_scale[sample_0]" if t_scale.dim() > 1 else "t_scale"
                dist.print0(f"{ts_prefix}=[{formatted}]")

        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if verbose: 
        dist.print0(f"DyWeight sampling completed. Total NFE: {nfe}")

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next

#----------------------------------------------------------------------------
def heun_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    **kwargs
):
    """Heun's second order sampler."""
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, 
                          schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)] if return_inters else None

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        if afs and i == 0:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, 
                                  condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

        denoised = get_denoised(net, x_next, t_next, class_labels=class_labels, 
                              condition=condition, unconditional_condition=unconditional_condition)
        d_prime = (x_next - denoised) / t_next
        x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
        
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, 
                            condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next

#----------------------------------------------------------------------------
def euler_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial',
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    **kwargs
):  
    """Euler sampler."""
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, 
                          schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)] if return_inters else None
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        if afs and i == 0:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, 
                                  condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
            
        x_next = x_cur + (t_next - t_cur) * d_cur
            
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, 
                            condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next

#----------------------------------------------------------------------------
def ipndm_sampler(
    net,
    latents,
    class_labels=None,
    condition=None,
    unconditional_condition=None,
    num_steps=None,
    sigma_min=0.002,
    sigma_max=80,
    schedule_type='polynomial',
    schedule_rho=7,
    afs=False,
    denoise_to_zero=False,
    return_inters=False,
    max_order=4,
    **kwargs
):
    """Improved Pseudo-Numerical Diffusion Method (IPNDM) sampler."""
    assert max_order >= 1 and max_order <= 4
    buffer_model = []

    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, 
                          schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)] if return_inters else None

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        if afs and i == 0:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, 
                                  condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        
        t_cur_r = t_cur.reshape(-1, 1, 1, 1)
        t_next_r = t_next.reshape(-1, 1, 1, 1)

        order = min(max_order, len(buffer_model) + 1)
        if order == 1:
            x_next = x_cur + (t_next_r - t_cur_r) * d_cur
        elif order == 2:
            x_next = x_cur + (t_next_r - t_cur_r) * (3 * d_cur - buffer_model[-1]) / 2
        elif order == 3:
            x_next = x_cur + (t_next_r - t_cur_r) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
        elif order == 4:
            x_next = x_cur + (t_next_r - t_cur_r) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        
        if len(buffer_model) < max_order - 1:
            buffer_model.append(d_cur.detach())
        elif max_order > 1:
            buffer_model.pop(0)
            buffer_model.append(d_cur.detach())

        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, 
                            condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    
    return x_next

#----------------------------------------------------------------------------
def dpm_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial',
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    r=0.5,
    **kwargs
):
    """DPM-Solver-2 sampler."""
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, 
                          schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)
    
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)] if return_inters else None

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        
        if afs and i == 0:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, 
                                  condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur

        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        x_next = x_cur + (t_mid - t_cur) * d_cur

        denoised = get_denoised(net, x_next, t_mid, class_labels=class_labels, 
                              condition=condition, unconditional_condition=unconditional_condition)
        d_mid = (x_next - denoised) / t_mid
        x_next = x_cur + (t_next - t_cur) * ((1 / (2 * r)) * d_mid + (1 - 1 / (2 * r)) * d_cur)
    
        if return_inters:
            inters.append(x_next.unsqueeze(0))
        
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, 
                            condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next

#----------------------------------------------------------------------------
def dpmpp_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial',
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    max_order=3,
    predict_x0=True,
    lower_order_final=True,
    **kwargs
):
    """DPM-Solver++ sampler."""
    if max_order > 3:
        max_order = 3
    assert max_order >= 1 and max_order <= 3
    
    from solver_utils import dpm_pp_update, dynamic_thresholding_fn
    
    t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, 
                          schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)] if return_inters else None
    buffer_model = []
    buffer_t = []

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        step_cur = i + 1
        
        if afs and i == 0:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
            denoised = x_cur - t_cur * d_cur
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, 
                                  condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
            
        buffer_model.append(dynamic_thresholding_fn(denoised) if predict_x0 else d_cur)
        buffer_t.append(t_cur)
        
        if lower_order_final:
            order = step_cur if step_cur < max_order else min(max_order, num_steps - step_cur)
        else:
            order = min(max_order, step_cur)
        
        x_next = dpm_pp_update(x_cur, buffer_model, buffer_t, t_next, order, predict_x0=predict_x0)
            
        if len(buffer_model) >= 3:
            buffer_model = [a.detach() for a in buffer_model[-3:]]
            buffer_t = [a.detach() for a in buffer_t[-3:]]
        else:
            buffer_model = [a.detach() for a in buffer_model]
            buffer_t = [a.detach() for a in buffer_t]
        
        if return_inters:
            inters.append(x_next.unsqueeze(0))
            
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, 
                            condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))
            
    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next

def unipc_sampler(
    net,
    latents,
    class_labels=None,
    condition=None,
    unconditional_condition=None,
    num_steps=None,
    sigma_min=0.002,
    sigma_max=80,
    schedule_type='polynomial',
    schedule_rho=7,
    afs=False,
    denoise_to_zero=False,
    return_inters=False,
    max_order=3,
    predict_x0=True,
    lower_order_final=True,
    variant='bh2',
    t_steps=None,
    **kwargs
):
    """ UniPC sampler: https://arxiv.org/abs/2302.04867. """

    if max_order > 3:
        max_order = 3
    assert 0 < max_order < 4

    if t_steps is None:
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device,
                               schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]

    if afs:
        d_next = x_next / ((1 + t_steps[0]**2).sqrt())
        denoised = x_next - t_steps[0] * d_next
    else:
        denoised = get_denoised(net, x_next, t_steps[0], class_labels=class_labels,
                                condition=condition, unconditional_condition=unconditional_condition)
        d_next = (x_next - denoised) / t_steps[0]

    if predict_x0:
        # LDM works in latent space; skip dynamic thresholding
        buffer_model = [denoised if hasattr(net, 'guidance_type') else dynamic_thresholding_fn(denoised)]
    else:
        buffer_model = [d_next]
    buffer_t = [t_steps[0]]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        if i + 1 < max_order:
            order = i + 1
            use_corrector = True
        else:
            order = min(max_order, num_steps - i - 1) if lower_order_final else max_order
            use_corrector = (i < num_steps - 2)

        x_next, model_out = unipc_update(
            x_cur, buffer_model, buffer_t, t_next, order,
            net=net, class_labels=class_labels, condition=condition,
            unconditional_condition=unconditional_condition,
            use_corrector=use_corrector, predict_x0=predict_x0, variant=variant,
        )

        if i + 1 < max_order:
            buffer_model.append(model_out)
            buffer_t.append(t_next)
        else:
            for k in range(max_order - 1):
                buffer_model[k] = buffer_model[k + 1]
                buffer_t[k] = buffer_t[k + 1]
            buffer_t[-1] = t_next
            if i < num_steps - 2:
                buffer_model[-1] = model_out

        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels,
                              condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next