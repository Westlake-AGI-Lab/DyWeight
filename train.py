import os
import re
import json
import click
import torch
import dnnlib

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides')
warnings.filterwarnings('ignore', category=UserWarning, message='.*epoch parameter.*scheduler.step.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*No device id is provided.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*using GPU.*device used by this process.*')

os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '0'

from torch_utils import distributed as dist
from training import training_loop

#----------------------------------------------------------------------------

@click.command()

# General options.
@click.option('--dataset_name',     help='Dataset name', metavar='STR',                                type=click.Choice(['cifar10', 'ffhq', 'afhqv2', 'imagenet64', 'lsun_bedroom', 'lsun_cat', 'imagenet256', 'ms_coco', 'lsun_bedroom_ldm']), required=True)
@click.option('--outdir',           help='Where to save the results', metavar='DIR',                   type=str, default='./exps')
@click.option('--total_kimg',       help='Number of images (k) for training', metavar='INT',           type=int, default=10)
@click.option('--prompt_path',      help='Path to MS-COCO_val2014_30k_captions.csv', metavar='DIR',    type=str)

# DyWeight Predictor options
@click.option('--num_steps',        help='Number of sampling steps', metavar='INT',                    type=click.IntRange(min=2), default=10, show_default=True)
@click.option('--max_history_steps', help='Maximum history steps window size (None = use all)',       type=int, default=None, show_default=True)
@click.option('--init_mode',        help='Initialization mode for hybrid weights',                    type=click.Choice(['ipndm', 'uniform', 'perturbed', 'euler']), default='ipndm', show_default=True)
@click.option('--enable_t_scale_learning', help='Enable t_scale parameter learning', metavar='BOOL',  type=bool, default=True, show_default=True)
@click.option('--t_scale_init_mode', help='t_scale initialization mode',                              type=click.Choice(['ones', 'uniform', 'perturbed']), default='ones', show_default=True)

# Sampling options
@click.option('--sampler_stu',      help='Student sampler', metavar='STR',                             type=click.Choice(['dyweight', 'ipndm', 'heun', 'dpm', 'dpmpp', 'euler']), default='dyweight', show_default=True)
@click.option('--sampler_tea',      help='Teacher sampler', metavar='STR',                             type=click.Choice(['heun', 'dpm', 'dpmpp', 'euler', 'ipndm']), default='heun', show_default=True)
@click.option('--teacher_steps',    help='Number of teacher sampling steps', metavar='INT',           type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--guidance_type',    help='Guidance type',                                              type=click.Choice(['cg', 'cfg', 'uncond', None]), default=None, show_default=True)
@click.option('--guidance_rate',    help='Guidance rate', metavar='FLOAT',                             type=float, default=0.)
@click.option('--schedule_type',    help='Time discretization schedule', metavar='STR',                type=click.Choice(['polynomial', 'logsnr', 'time_uniform', 'discrete']), default='polynomial', show_default=True)
@click.option('--schedule_rho',     help='Time step exponent', metavar='FLOAT',                        type=click.FloatRange(min=0), default=7, show_default=True)
@click.option('--afs',              help='Whether to use AFS', metavar='BOOL',                         type=bool, default=False, show_default=True)

# Additional options for multi-step solvers
@click.option('--max_order',        help='Max order for solvers', metavar='INT',                       type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--predict_x0',       help='Whether to use data prediction mode', metavar='BOOL',        type=bool, default=True, show_default=True)
@click.option('--lower_order_final', help='Lower the order at final stages', metavar='BOOL',           type=bool, default=True, show_default=True)

# Loss type configuration
@click.option('--loss_type',        help='Loss type',                                                  type=click.Choice(['l2', 'l1', 'huber', 'inception']), default='l2', show_default=True)
@click.option('--huber_delta',      help='Huber loss delta parameter', metavar='FLOAT',               type=click.FloatRange(min=0), default=0.1, show_default=True)

# Learning rate scheduling
@click.option('--use_cosine_annealing', help='Use cosine annealing LR schedule', metavar='BOOL',       type=bool, default=True, show_default=True)
@click.option('--cosine_min_lr_ratio', help='Minimum LR ratio for cosine annealing', metavar='FLOAT',  type=click.FloatRange(min=0, max=1), default=1e-2, show_default=True)
@click.option('--use_warmup', help='Use warm-up before cosine annealing', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--warmup_ratio', help='Ratio of warm-up steps to total steps', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.1, show_default=True)

# Hyperparameters.
@click.option('--batch',            help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--lr',               help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=5e-3, show_default=True)

# Performance-related.
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)

# I/O-related.
@click.option('--desc',             help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',         help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',             help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--snap',             help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--dump',             help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',             help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('-n', '--dry-run',    help='Print training options and exit',                            is_flag=True)

def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    try:
        c = dnnlib.EasyDict()
        c.loss_kwargs = dnnlib.EasyDict()
        c.pred_kwargs = dnnlib.EasyDict()
        c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)
        
        # DyWeight predictor architecture.
        c.pred_kwargs.class_name = 'training.networks.DyWeight_predictor'
        c.pred_kwargs.update(
            num_steps=opts.num_steps,
            max_history_steps=opts.max_history_steps,
            init_mode=opts.init_mode,
            enable_t_scale_learning=opts.enable_t_scale_learning,
            t_scale_init_mode=opts.t_scale_init_mode,
            afs=opts.afs,
            sampler_stu=opts.sampler_stu,
            sampler_tea=opts.sampler_tea,
            teacher_steps=opts.teacher_steps,
            schedule_type=opts.schedule_type,
            schedule_rho=opts.schedule_rho,
            max_order=opts.max_order,
            predict_x0=opts.predict_x0,
            lower_order_final=opts.lower_order_final,
        )

        # DyWeight loss function
        c.loss_kwargs.class_name = 'training.loss.DyWeight_loss'

        # Training options.
        c.total_kimg = opts.total_kimg
        c.kimg_per_tick = 1
        c.snapshot_ticks = opts.snap
        c.state_dump_ticks = opts.dump
        c.update(dataset_name=opts.dataset_name, batch_size=opts.batch, batch_gpu=opts.batch_gpu, 
                 gpus=dist.get_world_size(), cudnn_benchmark=opts.bench)
        c.update(guidance_type=opts.guidance_type, guidance_rate=opts.guidance_rate, prompt_path=opts.prompt_path)
        
        # Loss configuration
        c.update(
            loss_type=opts.loss_type,
            huber_delta=opts.huber_delta,
        )
        
        # Learning rate scheduling
        c.update(
            use_cosine_annealing=opts.use_cosine_annealing,
            cosine_min_lr_ratio=opts.cosine_min_lr_ratio,
            use_warmup=opts.use_warmup,
            warmup_ratio=opts.warmup_ratio,
        )
        
        # Random seed.
        if opts.seed is not None:
            c.seed = opts.seed
        else:
            seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
            torch.distributed.broadcast(seed, src=0)
            c.seed = int(seed)

        # Description string.
        if opts.schedule_type == 'polynomial':
            schedule_str = 'poly' + str(opts.schedule_rho)
        elif opts.schedule_type == 'logsnr':
            schedule_str = 'logsnr'
        elif opts.schedule_type == 'time_uniform':
            schedule_str = 'uni' + str(opts.schedule_rho)
        elif opts.schedule_type == 'discrete':
            schedule_str = 'discrete'
        else:
            raise ValueError("Got wrong schedule type: {}".format(opts.schedule_type))
        
        nfe = (opts.num_steps - 1) - 1 if opts.afs else (opts.num_steps - 1)
        nfe = 2 * nfe if opts.dataset_name == 'ms_coco' else nfe
        
        desc_parts = [
            opts.dataset_name,
            f"dw-{opts.num_steps}",
            f"nfe{nfe}",
            f"{opts.sampler_stu}-{opts.sampler_tea}",
            f"tea{opts.teacher_steps}",
            schedule_str,
            opts.init_mode,
            opts.loss_type
        ]
        
        if opts.afs:
            desc_parts.append('afs')
        if opts.max_history_steps is not None:
            desc_parts.append(f'hist{opts.max_history_steps}')
        if not opts.enable_t_scale_learning:
            desc_parts.append('notscale')
        if opts.use_cosine_annealing:
            desc_parts.append('cosine')
            
        desc = '-'.join(desc_parts)
        
        if opts.desc is not None:
            desc += f'-{opts.desc}'

        # Pick output directory.
        if dist.get_rank() != 0:
            c.run_dir = None
        elif opts.nosubdir:
            c.run_dir = opts.outdir
        else:
            prev_run_dirs = []
            if os.path.isdir(opts.outdir):
                prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
            prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
            prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
            cur_run_id = max(prev_run_ids, default=-1) + 1
            c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
            assert not os.path.exists(c.run_dir)

        # Print options.
        dist.print0()
        dist.print0('=' * 60)
        dist.print0('  DYWEIGHT TRAINING CONFIGURATION')
        dist.print0('=' * 60)

        dist.print0(f'  Output dir:     {c.run_dir}')
        dist.print0(f'  GPUs:           {dist.get_world_size()}')
        dist.print0(f'  Batch size:     {c.batch_size}')
        dist.print0(f'  Dataset:        {opts.dataset_name}')
        dist.print0()

        dist.print0(f'  Student solver: {opts.sampler_stu} (steps={opts.num_steps}, NFE={nfe})')
        dist.print0(f'  Teacher solver: {opts.sampler_tea} (steps={opts.teacher_steps})')
        dist.print0(f'  Schedule:       {opts.schedule_type} (rho={opts.schedule_rho})')
        dist.print0(f'  Init mode:      {opts.init_mode}')
        if opts.max_history_steps is not None:
            dist.print0(f'  History window: {opts.max_history_steps}')
        if opts.afs:
            dist.print0(f'  AFS:            True')
        if opts.enable_t_scale_learning:
            dist.print0(f'  t_scale:        enabled (init={opts.t_scale_init_mode})')
        if opts.guidance_type:
            dist.print0(f'  Guidance:       {opts.guidance_type} (rate={opts.guidance_rate})')
        dist.print0()

        loss_desc = opts.loss_type.upper()
        if opts.loss_type == 'huber':
            loss_desc += f' (delta={opts.huber_delta})'
        dist.print0(f'  Loss:           {loss_desc}')
        lr_desc = f'lr={opts.lr}'
        if opts.use_cosine_annealing:
            lr_desc += f', cosine (min_ratio={opts.cosine_min_lr_ratio})'
        if opts.use_warmup:
            lr_desc += f', warmup={opts.warmup_ratio*100:.0f}%'
        dist.print0(f'  Optimizer:      Adam, {lr_desc}')
        dist.print0(f'  Training:       {opts.total_kimg} kimg')
        dist.print0('-' * 60)
        dist.print0()

        # Dry run?
        if opts.dry_run:
            dist.print0('Dry run; exiting.')
            return

        # Create output directory.
        dist.print0('Creating output directory...')
        if dist.get_rank() == 0:
            os.makedirs(c.run_dir, exist_ok=True)
            with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
                json.dump(c, f, indent=2)
            dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

        # Train.
        training_loop.training_loop(**c)
    
    finally:
        dist.cleanup()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
