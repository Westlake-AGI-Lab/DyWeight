"""Main training loop for DyWeight."""
import os
import csv
import time
import copy
import json
import pickle
import numpy as np
import torch
import dnnlib
import random
from torch import autocast
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from models.ldm.util import instantiate_from_config
from torch_utils.download_util import check_file_by_key

#----------------------------------------------------------------------------

def load_ldm_model(config, ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        dist.print0(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        dist.print0("missing keys:", m)
    if len(u) > 0 and verbose:
        dist.print0("unexpected keys:", u)
    return model

#----------------------------------------------------------------------------

def create_model(dataset_name=None, guidance_type=None, guidance_rate=None, device=None):
    model_path, classifier_path = check_file_by_key(dataset_name)
    dist.print0(f'Loading the pre-trained diffusion model from "{model_path}"...')
    net = None

    if dataset_name in ['cifar10', 'ffhq', 'afhqv2', 'imagenet64']:
        with dnnlib.util.open_url(model_path, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['ema'].to(device)
        net.sigma_min = 0.002
        net.sigma_max = 80.0
    elif dataset_name in ['lsun_bedroom']:
        from models.cm.cm_model_loader import load_cm_model
        from models.networks_edm import CMPrecond
        net = load_cm_model(model_path)
        net = CMPrecond(net).to(device)
    else:
        if guidance_type == 'cg':
            assert classifier_path is not None
            from models.guided_diffusion.cg_model_loader import load_cg_model
            from models.networks_edm import CGPrecond
            net, classifier = load_cg_model(model_path, classifier_path)
            net = CGPrecond(net, classifier, guidance_rate=guidance_rate).to(device)
        elif guidance_type in ['uncond', 'cfg']:
            from omegaconf import OmegaConf
            from models.networks_edm import CFGPrecond
            if dataset_name in ['lsun_bedroom_ldm']:
                config = OmegaConf.load('./models/ldm/configs/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml')
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(net, img_resolution=64, img_channels=3, guidance_rate=1., guidance_type='uncond', label_dim=0).to(device)
            elif dataset_name in ['ms_coco']:
                assert guidance_type == 'cfg'
                config = OmegaConf.load('./models/ldm/configs/stable-diffusion/v1-inference.yaml')
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(net, img_resolution=64, img_channels=4, guidance_rate=guidance_rate, guidance_type='classifier-free', label_dim=True).to(device)

    if net is None:
        raise ValueError("Got wrong settings: check dataset_name and guidance_type!")
    net.eval()
    return net

#----------------------------------------------------------------------------

def generate_latents_and_conditions(net, batch_gpu, device, guidance_type, dataset_name, 
                                    sample_captions=None, guidance_rate=0., loss_fn=None):
    latents = loss_fn.sigma_max * torch.randn(
        [batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    labels = c = uc = None

    if net.label_dim:
        if guidance_type == 'cg':
            labels = torch.randint(net.label_dim, size=(batch_gpu,), device=device)
        elif guidance_type == 'cfg' and dataset_name in ['ms_coco']:
            prompts = random.sample(sample_captions, batch_gpu) if sample_captions else [""] * batch_gpu
            uc = None
            if guidance_rate != 1.0:
                uc = net.model.get_learned_conditioning(batch_gpu * [""])
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            c = net.model.get_learned_conditioning(prompts)
        else:
            labels = torch.eye(net.label_dim, device=device)[
                torch.randint(net.label_dim, size=[batch_gpu], device=device)]

    return latents, labels, c, uc

def generate_teacher_output(net, latents, labels, c, uc, guidance_type, loss_fn):
    with torch.no_grad():
        if guidance_type in ['uncond', 'cfg']:
            with autocast("cuda"):
                with net.model.ema_scope():
                    teacher_final = loss_fn.get_final_teacher_output(
                        net=net, tensor_in=latents, labels=labels, condition=c, unconditional_condition=uc)
        else:
            teacher_final = loss_fn.get_final_teacher_output(net=net, tensor_in=latents, labels=labels)
    return teacher_final.detach()

def save_trajectory_visualization(student_final, teacher_final, dataset_name, step, save_path, net=None):
    if student_final is None or teacher_final is None:
        return
    import matplotlib.pyplot as plt

    if dataset_name in ['lsun_bedroom_ldm', 'ms_coco']:
        with torch.no_grad():
            with net.model.ema_scope():
                student_decoded = net.model.decode_first_stage(student_final)
                teacher_decoded = net.model.decode_first_stage(teacher_final)
        final_img_stu = student_decoded.detach().cpu()
        final_img_tea = teacher_decoded.detach().cpu()
    else:
        final_img_stu = student_final.cpu()
        final_img_tea = teacher_final.cpu()

    stu_display = np.clip((final_img_stu[0].numpy().transpose(1, 2, 0) + 1) / 2, 0, 1)
    tea_display = np.clip((final_img_tea[0].numpy().transpose(1, 2, 0) + 1) / 2, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].imshow(stu_display)
    axes[0].set_title(f'Student (Step {step})')
    axes[0].axis('off')
    axes[1].imshow(tea_display)
    axes[1].set_title(f'Teacher (Step {step})')
    axes[1].axis('off')
    plt.suptitle(f'DyWeight Training - {dataset_name}', fontsize=16)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
    plt.close()

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',
    pred_kwargs         = {},
    loss_kwargs         = {},
    optimizer_kwargs    = {},
    seed                = 0,
    batch_size          = None,
    batch_gpu           = None,
    total_kimg          = 20,
    kimg_per_tick       = 1,
    snapshot_ticks      = 1,
    state_dump_ticks    = 20,
    cudnn_benchmark     = True,
    dataset_name        = None,
    prompt_path         = None,
    guidance_type       = None,
    guidance_rate       = 0.,
    loss_type           = 'l2',
    huber_delta         = 0.1,
    use_cosine_annealing = True,
    cosine_min_lr_ratio = 1e-2,
    use_warmup          = False,
    warmup_ratio        = 0.1,
    device              = torch.device('cuda'),
    **kwargs,
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Batch size per GPU and gradient accumulation.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load captions for MS-COCO.
    sample_captions = None
    if dataset_name in ['ms_coco']:
        prompt_path, _ = check_file_by_key('prompts')
        sample_captions = []
        with open(prompt_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                sample_captions.append(row['text'])

    # Load pre-trained diffusion model.
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    net = create_model(dataset_name, guidance_type, guidance_rate, device)
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Construct DyWeight predictor.
    pred_kwargs.update(img_resolution=net.img_resolution, guidance_rate=guidance_rate, guidance_type=guidance_type)
    predictor = dnnlib.util.construct_class_by_name(**pred_kwargs)
    predictor.train().requires_grad_(True).to(device)

    # Setup loss function.
    loss_kwargs.update(
        num_steps=pred_kwargs.get('num_steps', 10),
        sampler_stu=pred_kwargs.get('sampler_stu', 'ipndm'),
        sampler_tea=pred_kwargs.get('sampler_tea', 'heun'),
        teacher_steps=pred_kwargs.get('teacher_steps', 100),
        schedule_type=pred_kwargs.get('schedule_type', 'polynomial'),
        schedule_rho=pred_kwargs.get('schedule_rho', 7),
        afs=pred_kwargs.get('afs', True),
        max_order=pred_kwargs.get('max_order', 4),
        sigma_min=net.sigma_min, sigma_max=net.sigma_max,
        predict_x0=pred_kwargs.get('predict_x0', False),
        lower_order_final=pred_kwargs.get('lower_order_final', True),
        loss_type=loss_type, huber_delta=huber_delta, dataset_name=dataset_name,
    )
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)

    # Setup optimizer.
    optimizer = dnnlib.util.construct_class_by_name(params=predictor.parameters(), **optimizer_kwargs)

    # Setup learning rate scheduler.
    scheduler = None
    total_steps = (total_kimg * 1000) // batch_size
    if use_cosine_annealing:
        initial_lr = optimizer.param_groups[0]['lr']
        min_lr = initial_lr * cosine_min_lr_ratio
        if use_warmup:
            warmup_steps = int(total_steps * warmup_ratio)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr),
                ],
                milestones=[warmup_steps],
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)

    # Setup DDP.
    ddp = torch.nn.parallel.DistributedDataParallel(predictor, device_ids=[device], broadcast_buffers=False, find_unused_parameters=False)

    dist.print0(f'Training for {total_kimg} kimg ({total_steps} steps), batch={batch_size}')

    use_afs = pred_kwargs.get('afs', False)
    use_ema_scope = guidance_type in ['uncond', 'cfg']

    # ---- Training loop ----
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = 0
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(0, total_kimg)
    stats_jsonl = None
    tick_loss_sum = 0.0
    tick_loss_count = 0
    loss_history = []
    kimg_history = []
    step_count = 0

    while True:
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        last_student_final = None
        last_teacher_final = None

        for round_idx in range(num_accumulation_rounds):
            # Generate a fresh mini-batch for each accumulation round.
            latents, labels, c, uc = generate_latents_and_conditions(
                net, batch_gpu, device, guidance_type, dataset_name,
                sample_captions=sample_captions, guidance_rate=guidance_rate, loss_fn=loss_fn)

            teacher_final = generate_teacher_output(net, latents, labels, c, uc, guidance_type, loss_fn)

            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                hybrid_weights, t_scale = ddp(torch.zeros(1, device=device))

                if use_ema_scope:
                    with net.model.ema_scope():
                        student_final = loss_fn.get_student_final_output(
                            net=net, tensor_in=latents, labels=labels,
                            hybrid_weights=hybrid_weights, t_scale=t_scale,
                            condition=c, unconditional_condition=uc, use_afs=use_afs)
                else:
                    student_final = loss_fn.get_student_final_output(
                        net=net, tensor_in=latents, labels=labels,
                        hybrid_weights=hybrid_weights, t_scale=t_scale, use_afs=use_afs)

                loss = loss_fn.compute_loss(student_final, teacher_final, net=net)

                if round_idx == num_accumulation_rounds - 1:
                    last_student_final = student_final.detach()
                    last_teacher_final = teacher_final

                total_loss += loss.item()
                loss.mul(1 / num_accumulation_rounds).backward()

        avg_loss = total_loss / num_accumulation_rounds
        training_stats.report('Loss/batch_loss', avg_loss)
        tick_loss_sum += avg_loss
        tick_loss_count += 1

        if (cur_nimg // batch_size) % 3 == 0 and dist.get_rank() == 0 and last_student_final is not None:
            save_trajectory_visualization(
                last_student_final, last_teacher_final.detach(), dataset_name,
                cur_nimg // batch_size, os.path.join(run_dir, 'trajectory_visualization.png'), net=net)

        torch.nn.utils.clip_grad_norm_(ddp.parameters(), max_norm=1.0)
        optimizer.step()
        step_count += 1
        if scheduler is not None:
            scheduler.step()

        # Periodic weight logging.
        if cur_nimg > 0 and cur_nimg % (100 * batch_size) == 0:
            with torch.no_grad():
                hw, ts = ddp(torch.zeros(1, device=device))
                dist.print0(f"Step {cur_nimg // batch_size}: Loss = {avg_loss:.6f}")
                for i in range(hw.shape[0]):
                    ew = ddp.module.get_effective_window_size() if hasattr(ddp.module, 'get_effective_window_size') else 0
                    row = hw[i, max(0, i - ew):i+1]
                    dist.print0(f"  Weights[{i}]: [{'/'.join(f'{w:.4f}' for w in row)}]")
                if ts is not None:
                    dist.print0(f"  t_scale: [{'/'.join(f'{t:.4f}' for t in ts[:5])}{'...' if len(ts) > 5 else ''}]")

        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # ---- Tick boundary ----
        tick_end_time = time.time()
        tick_avg_loss = tick_loss_sum / max(tick_loss_count, 1)
        loss_history.append(tick_avg_loss)
        kimg_history.append(cur_nimg / 1e3)

        fields = [
            f"tick {training_stats.report0('Progress/tick', cur_tick):<4d}",
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}",
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<10s}",
            f"loss {tick_avg_loss:<.6f}",
            f"lr {training_stats.report0('Learning/current_lr', optimizer.param_groups[0]['lr']):<.2e}",
            f"step {step_count}/{total_steps}" if scheduler else None,
            f"gpu {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):.1f}G",
        ]
        training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time)
        training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3)
        training_stats.report0('Timing/maintenance_sec', maintenance_time)
        training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30)
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(f for f in fields if f))

        if (not done) and dist.should_stop():
            done = True
            dist.print0('\nAborting...')

        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0) and cur_tick > 0:
            data = dict(predictor=ddp.module, loss_fn=loss_fn)
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value
            
            # Remove inception_extractor from loss_fn to reduce snapshot size
            # It will be recreated when loading the checkpoint
            if hasattr(data['loss_fn'], 'inception_extractor'):
                original_inception = data['loss_fn'].inception_extractor
                data['loss_fn'].inception_extractor = None
            
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            
            # Restore inception_extractor
            if hasattr(data['loss_fn'], 'inception_extractor') and data['loss_fn'].inception_extractor is None:
                data['loss_fn'].inception_extractor = original_inception
            
            del data

        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        tick_loss_sum = 0.0
        tick_loss_count = 0

        if done:
            break

    # Save loss curve.
    if dist.get_rank() == 0 and len(loss_history) > 0:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(kimg_history, loss_history, 'b-', linewidth=2)
        plt.xlabel('Training Progress (kimg)')
        plt.ylabel('Loss')
        loss_title = f'Huber Loss (delta={huber_delta:.2f})' if loss_type == 'huber' else f'{loss_type.upper()} Loss'
        plt.title(f'DyWeight Training - {dataset_name} ({loss_title})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f'training_loss_{dataset_name}_{loss_type}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    dist.print0('\nTraining complete.')

#----------------------------------------------------------------------------
