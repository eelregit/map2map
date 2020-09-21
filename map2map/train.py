import os
import socket
import time
import sys
from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data import FieldDataset, DistFieldSampler
from . import models
from .models import (
    narrow_cast, resample,
    WDistLoss, wasserstein_distance_loss, wgan_grad_penalty,
    grad_penalty_reg,
    add_spectral_norm, rm_spectral_norm,
    InstanceNoise,
)
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

    args.in_chan, args.out_chan = train_dataset.in_chan, train_dataset.tgt_chan

    model = import_attr(args.model, models, callback_at=args.callback_at)
    model = model(sum(args.in_chan), sum(args.out_chan),
                  scale_factor=args.scale_factor)
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

    adv_model = adv_criterion = adv_optimizer = adv_scheduler = None
    if args.adv:
        adv_model = import_attr(args.adv_model, models,
                                callback_at=args.callback_at)
        adv_model = adv_model(
            sum(args.in_chan + args.out_chan) if args.cgan
                else sum(args.out_chan),
            1,
            scale_factor=args.scale_factor
        )
        if args.adv_model_spectral_norm:
            add_spectral_norm(adv_model)
        adv_model.to(device)
        adv_model = DistributedDataParallel(adv_model, device_ids=[device],
                                            process_group=dist.new_group())

        adv_criterion = import_attr(args.adv_criterion, nn, models,
                                    callback_at=args.callback_at)
        adv_criterion = adv_criterion()
        adv_criterion.to(device)

        adv_optimizer = import_attr(args.optimizer, optim,
                                    callback_at=args.callback_at)
        adv_optimizer = adv_optimizer(
            adv_model.parameters(),
            lr=args.adv_lr,
            **args.adv_optimizer_args,
        )
        adv_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            adv_optimizer, **args.scheduler_args)

    if (args.load_state == ckpt_link and not os.path.isfile(ckpt_link)
            or not args.load_state):
        if args.init_weight_std is not None:
            model.apply(init_weights)

            if args.adv:
                adv_model.apply(init_weights)

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

        if args.adv:
            if 'adv_model' in state:
                load_model_state_dict(adv_model.module, state['adv_model'],
                                      strict=args.load_state_strict)

            if 'adv_optimizer' in state:
                adv_optimizer.load_state_dict(state['adv_optimizer'])
            if 'adv_scheduler' in state:
                adv_scheduler.load_state_dict(state['adv_scheduler'])

        torch.set_rng_state(state['rng'].cpu())  # move rng state back

        if rank == 0:
            min_loss = state['min_loss']
            if args.adv and 'adv_model' not in state:
                min_loss = None  # restarting with adversary wipes the record

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

    if args.adv:
        args.instance_noise = InstanceNoise(args.instance_noise,
                                            args.instance_noise_batches)

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        train_loss = train(epoch, train_loader,
            model, criterion, optimizer, scheduler,
            adv_model, adv_criterion, adv_optimizer, adv_scheduler,
            logger, device, args)
        epoch_loss = train_loss

        if args.val:
            val_loss = validate(epoch, val_loader,
                model, criterion, adv_model, adv_criterion,
                logger, device, args)
            #epoch_loss = val_loss

        if args.reduce_lr_on_plateau and epoch >= args.adv_start:
            scheduler.step(epoch_loss[0])
            if args.adv:
                adv_scheduler.step(epoch_loss[0])

        if rank == 0:
            logger.flush()

            if ((min_loss is None or epoch_loss[0] < min_loss[0])
                    and epoch >= args.adv_start):
                min_loss = epoch_loss

            state = {
                'epoch': epoch + 1,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'rng': torch.get_rng_state(),
                'min_loss': min_loss,
            }
            if args.adv:
                state.update({
                    'adv_model': adv_model.module.state_dict(),
                    'adv_optimizer': adv_optimizer.state_dict(),
                    'adv_scheduler': adv_scheduler.state_dict(),
                })

            state_file = 'state_{}.pt'.format(epoch + 1)
            torch.save(state, state_file)
            del state

            tmp_link = '{}.pt'.format(time.time())
            os.symlink(state_file, tmp_link)  # workaround to overwrite
            os.rename(tmp_link, ckpt_link)

    dist.destroy_process_group()


def train(epoch, loader, model, criterion, optimizer, scheduler,
        adv_model, adv_criterion, adv_optimizer, adv_scheduler,
        logger, device, args):
    model.train()
    if args.adv:
        adv_model.train()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # loss, loss_adv, adv_loss, adv_loss_fake, adv_loss_real
    # loss: generator (model) supervised loss
    # loss_adv: generator (model) adversarial loss
    # adv_loss: discriminator (adv_model) loss
    epoch_loss = torch.zeros(5, dtype=torch.float64, device=device)
    fake = torch.zeros([1], dtype=torch.float32, device=device)
    real = torch.ones([1], dtype=torch.float32, device=device)
    adv_real = torch.full([1], args.adv_label_smoothing, dtype=torch.float32,
            device=device)

    for i, (input, target) in enumerate(loader):
        batch = epoch * len(loader) + i + 1

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input)
        if batch == 1 and rank == 0:
            print('input shape :', input.shape)
            print('output shape :', output.shape)
            print('target shape :', target.shape)

        if (hasattr(model.module, 'scale_factor')
                and model.module.scale_factor != 1):
            input = resample(input, model.module.scale_factor, narrow=False)
        input, output, target = narrow_cast(input, output, target)
        if batch == 1 and rank == 0:
            print('narrowed shape :', output.shape, flush=True)

        loss = criterion(output, target)
        epoch_loss[0] += loss.item()

        if args.adv and epoch >= args.adv_start:
            noise_std = args.instance_noise.std()
            if noise_std > 0:
                noise = noise_std * torch.randn_like(output)
                output = output + noise
                noise = noise_std * torch.randn_like(target)
                target = target + noise
                del noise

            if args.cgan:
                output = torch.cat([input, output], dim=1)
                target = torch.cat([input, target], dim=1)

            # discriminator
            set_requires_grad(adv_model, True)

            score_out = adv_model(output.detach())
            adv_loss_fake = adv_criterion(score_out, fake.expand_as(score_out))
            epoch_loss[3] += adv_loss_fake.item()

            adv_optimizer.zero_grad()
            adv_loss_fake.backward()

            score_tgt = adv_model(target)
            adv_loss_real = adv_criterion(score_tgt, adv_real.expand_as(score_tgt))
            epoch_loss[4] += adv_loss_real.item()

            adv_loss_real.backward()

            adv_loss = adv_loss_fake + adv_loss_real
            epoch_loss[2] += adv_loss.item()

            if (args.adv_wgan_gp_interval > 0
                and  batch % args.adv_wgan_gp_interval == 0):
                adv_loss_reg = wgan_grad_penalty(adv_model, output, target)
                adv_loss_reg_ = adv_loss_reg * args.adv_wgan_gp_interval

                adv_loss_reg_.backward()

                if rank == 0:
                    logger.add_scalar(
                        'loss/batch/train/adv/reg',
                        adv_loss_reg.item(),
                        global_step=batch,
                    )

            adv_optimizer.step()
            adv_grads = get_grads(adv_model)

            # generator adversarial loss
            set_requires_grad(adv_model, False)

            score_out = adv_model(output)
            loss_adv = adv_criterion(score_out, real.expand_as(score_out))
            epoch_loss[1] += loss_adv.item()

            optimizer.zero_grad()
            loss_adv.backward()
            optimizer.step()
            grads = get_grads(model)
        else:
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
                if args.adv and epoch >= args.adv_start:
                    logger.add_scalar('loss/batch/train/adv/G', loss_adv.item(),
                                      global_step=batch)
                    logger.add_scalars(
                        'loss/batch/train/adv/D',
                        {
                            'total': adv_loss.item(),
                            'fake': adv_loss_fake.item(),
                            'real': adv_loss_real.item(),
                        },
                        global_step=batch,
                    )

                logger.add_scalar('grad/first', grads[0], global_step=batch)
                logger.add_scalar('grad/last', grads[-1], global_step=batch)
                if args.adv and epoch >= args.adv_start:
                    logger.add_scalar('grad/adv/first', adv_grads[0],
                                      global_step=batch)
                    logger.add_scalar('grad/adv/last', adv_grads[-1],
                                      global_step=batch)

                    if noise_std > 0:
                        logger.add_scalar('instance_noise', noise_std,
                                          global_step=batch)

    dist.all_reduce(epoch_loss)
    epoch_loss /= len(loader) * world_size
    if rank == 0:
        logger.add_scalar('loss/epoch/train', epoch_loss[0],
                          global_step=epoch+1)
        if args.adv and epoch >= args.adv_start:
            logger.add_scalar('loss/epoch/train/adv/G', epoch_loss[1],
                              global_step=epoch+1)
            logger.add_scalars(
                'loss/epoch/train/adv/D',
                {
                    'total': epoch_loss[2],
                    'fake': epoch_loss[3],
                    'real': epoch_loss[4],
                },
                global_step=epoch+1,
            )

        skip_chan = 0
        if args.adv and epoch >= args.adv_start and args.cgan:
            skip_chan = sum(args.in_chan)

        fig = plt_slices(
            input[-1], output[-1, skip_chan:], target[-1, skip_chan:],
            output[-1, skip_chan:] - target[-1, skip_chan:],
            title=['in', 'out', 'tgt', 'out - tgt'],
        )
        logger.add_figure('fig/train', fig, global_step=epoch+1)
        fig.clf()

        #fig = plt_power(
        #    input, output[:, skip_chan:], target[:, skip_chan:],
        #    label=['in', 'out', 'tgt'],
        #)
        #logger.add_figure('fig/train/power/lag', fig, global_step=epoch+1)
        #fig.clf()

        #fig = plt_power(
        #    input, output[:, skip_chan:], target[:, skip_chan:],
        #    l2e=True, label=['in', 'out', 'tgt'],
        #)
        #logger.add_figure('fig/train/power/eul', fig, global_step=epoch+1)
        #fig.clf()

    return epoch_loss


def validate(epoch, loader, model, criterion, adv_model, adv_criterion,
        logger, device, args):
    model.eval()
    if args.adv:
        adv_model.eval()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    epoch_loss = torch.zeros(5, dtype=torch.float64, device=device)
    fake = torch.zeros([1], dtype=torch.float32, device=device)
    real = torch.ones([1], dtype=torch.float32, device=device)

    with torch.no_grad():
        for input, target in loader:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input)

            if (hasattr(model.module, 'scale_factor')
                    and model.module.scale_factor != 1):
                input = resample(input, model.module.scale_factor, narrow=False)
            input, output, target = narrow_cast(input, output, target)

            loss = criterion(output, target)
            epoch_loss[0] += loss.item()

            if args.adv and epoch >= args.adv_start:
                if args.cgan:
                    output = torch.cat([input, output], dim=1)
                    target = torch.cat([input, target], dim=1)

                # discriminator
                score_out = adv_model(output)
                adv_loss_fake = adv_criterion(score_out, fake.expand_as(score_out))
                epoch_loss[3] += adv_loss_fake.item()

                score_tgt = adv_model(target)
                adv_loss_real = adv_criterion(score_tgt, real.expand_as(score_tgt))
                epoch_loss[4] += adv_loss_real.item()

                adv_loss = adv_loss_fake + adv_loss_real
                epoch_loss[2] += adv_loss.item()

                # generator adversarial loss
                loss_adv = adv_criterion(score_out, real.expand_as(score_out))
                epoch_loss[1] += loss_adv.item()

    dist.all_reduce(epoch_loss)
    epoch_loss /= len(loader) * world_size
    if rank == 0:
        logger.add_scalar('loss/epoch/val', epoch_loss[0],
                          global_step=epoch+1)
        if args.adv and epoch >= args.adv_start:
            logger.add_scalar('loss/epoch/val/adv/G', epoch_loss[1],
                              global_step=epoch+1)
            logger.add_scalars(
                'loss/epoch/val/adv/D',
                {
                    'total': epoch_loss[2],
                    'fake': epoch_loss[3],
                    'real': epoch_loss[4],
                },
                global_step=epoch+1,
            )

        skip_chan = 0
        if args.adv and epoch >= args.adv_start and args.cgan:
            skip_chan = sum(args.in_chan)

        fig = plt_slices(
            input[-1], output[-1, skip_chan:], target[-1, skip_chan:],
            output[-1, skip_chan:] - target[-1, skip_chan:],
            title=['in', 'out', 'tgt', 'out - tgt'],
        )
        logger.add_figure('fig/val', fig, global_step=epoch+1)
        fig.clf()

        #fig = plt_power(
        #    input, output[:, skip_chan:], target[:, skip_chan:],
        #    label=['in', 'out', 'tgt'],
        #)
        #logger.add_figure('fig/val/power/lag', fig, global_step=epoch+1)
        #fig.clf()

        #fig = plt_power(
        #    input, output[:, skip_chan:], target[:, skip_chan:],
        #    l2e=True, label=['in', 'out', 'tgt'],
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

    if rank != 0:
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
    grads = [g.detach().norm().item() for g in grads]
    return grads
