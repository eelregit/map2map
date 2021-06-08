import os
import socket
import time
import sys
from pprint import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data import FieldDataset, DistFieldSampler
from . import models
from .models import narrow_cast, resample
from .utils import import_attr, load_model_state_dict, plt_slices, plt_power


ckpt_link = 'checkpoint.pt'


def node_worker(args):
    if 'SLURM_STEP_NUM_NODES' in os.environ:
        args.nodes = int(os.environ['SLURM_STEP_NUM_NODES'])
    elif 'SLURM_JOB_NUM_NODES' in os.environ:
        args.nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    else:
        raise KeyError('missing node counts in slurm env')
    args.gpus_per_node = torch.cuda.device_count()
    args.world_size = args.nodes * args.gpus_per_node

    node = int(os.environ['SLURM_NODEID'])

    if args.gpus_per_node < 1:
        raise RuntimeError('GPU not found on node {}'.format(node))

    spawn(gpu_worker, args=(node, args), nprocs=args.gpus_per_node)


def gpu_worker(local_rank, node, args):
    #device = torch.device('cuda', local_rank)
    #torch.cuda.device(device)  # env var recommended over this

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    device = torch.device('cuda', 0)

    rank = args.gpus_per_node * node + local_rank

    # Need randomness across processes, for sampler, augmentation, noise etc.
    # Note DDP broadcasts initial model states from rank 0
    torch.manual_seed(args.seed + rank)
    # good practice to disable cudnn.benchmark if enabling cudnn.deterministic
    #torch.backends.cudnn.deterministic = True

    dist_init(rank, args)

    train_dataset = FieldDataset(
        style_pattern=args.train_style_pattern,
        in_patterns=args.train_in_patterns,
        tgt_patterns=args.train_tgt_patterns,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        callback_at=args.callback_at,
        augment=args.augment,
        aug_shift=args.aug_shift,
        aug_add=args.aug_add,
        aug_mul=args.aug_mul,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        in_pad=args.in_pad,
        tgt_pad=args.tgt_pad,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )
    train_sampler = DistFieldSampler(train_dataset, shuffle=True,
                                     div_data=args.div_data,
                                     div_shuffle_dist=args.div_shuffle_dist)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    if args.val:
        val_dataset = FieldDataset(
            style_pattern=args.val_style_pattern,
            in_patterns=args.val_in_patterns,
            tgt_patterns=args.val_tgt_patterns,
            in_norms=args.in_norms,
            tgt_norms=args.tgt_norms,
            callback_at=args.callback_at,
            augment=False,
            aug_shift=None,
            aug_add=None,
            aug_mul=None,
            crop=args.crop,
            crop_start=args.crop_start,
            crop_stop=args.crop_stop,
            crop_step=args.crop_step,
            in_pad=args.in_pad,
            tgt_pad=args.tgt_pad,
            scale_factor=args.scale_factor,
            **args.misc_kwargs,
        )
        val_sampler = DistFieldSampler(val_dataset, shuffle=False,
                                       div_data=args.div_data,
                                       div_shuffle_dist=args.div_shuffle_dist)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.loader_workers,
            pin_memory=True,
        )

    args.style_size = train_dataset.style_size
    args.in_chan = train_dataset.in_chan
    args.out_chan = train_dataset.tgt_chan

    model = import_attr(args.model, models, callback_at=args.callback_at)
    model = model(args.style_size, sum(args.in_chan), sum(args.out_chan),
                  scale_factor=args.scale_factor, **args.misc_kwargs)
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[device],
            process_group=dist.new_group())

    criterion = import_attr(args.criterion, nn, models,
                            callback_at=args.callback_at)
    criterion = criterion()
    criterion.to(device)

    optimizer = import_attr(args.optimizer, optim, callback_at=args.callback_at)
    optimizer = optimizer(
        model.parameters(),
        lr=args.lr,
        **args.optimizer_args,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **args.scheduler_args)

    if (args.load_state == ckpt_link and not os.path.isfile(ckpt_link)
            or not args.load_state):
        if args.init_weight_std is not None:
            model.apply(init_weights)

        start_epoch = 0

        if rank == 0:
            min_loss = None
    else:
        state = torch.load(args.load_state, map_location=device)

        start_epoch = state['epoch']

        load_model_state_dict(model.module, state['model'],
                              strict=args.load_state_strict)

        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state:
            scheduler.load_state_dict(state['scheduler'])

        torch.set_rng_state(state['rng'].cpu())  # move rng state back

        if rank == 0:
            min_loss = state['min_loss']

            print('state at epoch {} loaded from {}'.format(
                state['epoch'], args.load_state), flush=True)

        del state

    torch.backends.cudnn.benchmark = True

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    logger = None
    if rank == 0:
        logger = SummaryWriter()

    if rank == 0:
        print('pytorch {}'.format(torch.__version__))
        pprint(vars(args))
        sys.stdout.flush()

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        train_loss = train(epoch, train_loader, model, criterion,
                           optimizer, scheduler, logger, device, args)
        epoch_loss = train_loss

        if args.val:
            val_loss = validate(epoch, val_loader, model, criterion,
                                logger, device, args)
            #epoch_loss = val_loss

        if args.reduce_lr_on_plateau:
            scheduler.step(epoch_loss[2])

        if rank == 0:
            logger.flush()

            if min_loss is None or epoch_loss[2] < min_loss:
                min_loss = epoch_loss[2]

            state = {
                'epoch': epoch + 1,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'rng': torch.get_rng_state(),
                'min_loss': min_loss,
            }

            state_file = 'state_{}.pt'.format(epoch + 1)
            torch.save(state, state_file)
            del state

            tmp_link = '{}.pt'.format(time.time())
            os.symlink(state_file, tmp_link)  # workaround to overwrite
            os.rename(tmp_link, ckpt_link)

    dist.destroy_process_group()


def train(epoch, loader, model, criterion,
          optimizer, scheduler, logger, device, args):
    model.train()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    epoch_loss = torch.zeros(3, dtype=torch.float64, device=device)

    for i, data in enumerate(loader):
        batch = epoch * len(loader) + i + 1

        style, input, target = data['style'], data['input'], data['target']

        style = style.to(device, non_blocking=True)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input, style)
        if batch <= 5 and rank == 0:
            print('##### batch :', batch)
            print('style shape :', style.shape)
            print('input shape :', input.shape)
            print('output shape :', output.shape)
            print('target shape :', target.shape)

        if (hasattr(model.module, 'scale_factor')
                and model.module.scale_factor != 1):
            input = resample(input, model.module.scale_factor, narrow=False)
        input, output, target = narrow_cast(input, output, target)
        if batch <= 5 and rank == 0:
            print('narrowed shape :', output.shape)

        loss = criterion(output, target)
        epoch_loss[0] += loss.detach()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        grads = get_grads(model)

        if batch % args.log_interval == 0:
            dist.all_reduce(loss)
            loss /= world_size
            if rank == 0:
                logger.add_scalar('loss/batch/train', loss.item(),
                                  global_step=batch)

                logger.add_scalar('grad/first', grads[0], global_step=batch)
                logger.add_scalar('grad/last', grads[-1], global_step=batch)

    dist.all_reduce(epoch_loss)
    epoch_loss /= len(loader) * world_size
    if rank == 0:
        logger.add_scalar('loss/epoch/train', epoch_loss[0],
                          global_step=epoch+1)

        fig = plt_slices(
            input[-1], output[-1, skip_chan:], target[-1, skip_chan:],
            output[-1, skip_chan:] - target[-1, skip_chan:],
            title=['in', 'out', 'tgt', 'out - tgt'],
            **args.misc_kwargs,
        )
        logger.add_figure('fig/train', fig, global_step=epoch+1)
        fig.clf()

        fig = plt_power(
            input, output[:, skip_chan:], target[:, skip_chan:],
            label=['in', 'out', 'tgt'],
            **args.misc_kwargs,
        )
        logger.add_figure('fig/train/power/lag', fig, global_step=epoch+1)
        fig.clf()

        #fig = plt_power(1.0,
        #    dis=[input, output[:, skip_chan:], target[:, skip_chan:]],
        #    label=['in', 'out', 'tgt'],
        #    **args.misc_kwargs,
        #)
        #logger.add_figure('fig/train/power/eul', fig, global_step=epoch+1)
        #fig.clf()

    return epoch_loss


def validate(epoch, loader, model, criterion, logger, device, args):
    model.eval()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    epoch_loss = torch.zeros(3, dtype=torch.float64, device=device)

    with torch.no_grad():
        for data in loader:
            style, input, target = data['style'], data['input'], data['target']

            style = style.to(device, non_blocking=True)
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input, style)

            if (hasattr(model.module, 'scale_factor')
                    and model.module.scale_factor != 1):
                input = resample(input, model.module.scale_factor, narrow=False)
            input, output, target = narrow_cast(input, output, target)

            loss = criterion(output, target)
            epoch_loss[0] += loss.detach()

    dist.all_reduce(epoch_loss)
    epoch_loss /= len(loader) * world_size
    if rank == 0:
        logger.add_scalar('loss/epoch/val', epoch_loss[0],
                          global_step=epoch+1)

        fig = plt_slices(
            input[-1], output[-1, skip_chan:], target[-1, skip_chan:],
            output[-1, skip_chan:] - target[-1, skip_chan:],
            title=['in', 'out', 'tgt', 'out - tgt'],
            **args.misc_kwargs,
        )
        logger.add_figure('fig/val', fig, global_step=epoch+1)
        fig.clf()

        fig = plt_power(
            input, output[:, skip_chan:], target[:, skip_chan:],
            label=['in', 'out', 'tgt'],
            **args.misc_kwargs,
        )
        logger.add_figure('fig/val/power/lag', fig, global_step=epoch+1)
        fig.clf()

        #fig = plt_power(1.0,
        #    dis=[input, output[:, skip_chan:], target[:, skip_chan:]],
        #    label=['in', 'out', 'tgt'],
        #    **args.misc_kwargs,
        #)
        #logger.add_figure('fig/val/power/eul', fig, global_step=epoch+1)
        #fig.clf()

    return epoch_loss


def dist_init(rank, args):
    dist_file = 'dist_addr'

    if rank == 0:
        addr = socket.gethostname()

        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((addr, 0))
            _, port = s.getsockname()

        args.dist_addr = 'tcp://{}:{}'.format(addr, port)

        with open(dist_file, mode='w') as f:
            f.write(args.dist_addr)
    else:
        while not os.path.exists(dist_file):
            time.sleep(1)

        with open(dist_file, mode='r') as f:
            args.dist_addr = f.read()

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_addr,
        world_size=args.world_size,
        rank=rank,
    )
    dist.barrier()

    if rank == 0:
        os.remove(dist_file)


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        m.weight.data.normal_(0.0, args.init_weight_std)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
        if m.affine:
            # NOTE: dispersion from DCGAN, why?
            m.weight.data.normal_(1.0, args.init_weight_std)
            m.bias.data.fill_(0)


def set_requires_grad(module, requires_grad=False):
    for param in module.parameters():
        param.requires_grad = requires_grad


def get_grads(model):
    """gradients of the weights of the first and the last layer
    """
    grads = list(p.grad for n, p in model.named_parameters()
                 if '.weight' in n)
    grads = [grads[0], grads[-1]]
    grads = [g.detach().norm() for g in grads]
    return grads
