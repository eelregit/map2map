import os
import socket
import time
import sys
from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data import FieldDataset
from .data.figures import fig3d
from . import models
from .models import (narrow_like,
        adv_model_wrapper, adv_criterion_wrapper,
        add_spectral_norm, rm_spectral_norm,
        InstanceNoise)
from .state import load_model_state_dict


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

    spawn(gpu_worker, args=(node, args), nprocs=args.gpus_per_node)


def gpu_worker(local_rank, node, args):
    set_runtime_default_args(args)

    device = torch.device('cuda', local_rank)
    torch.cuda.device(device)

    torch.manual_seed(args.seed)
    #torch.backends.cudnn.deterministic = True  # NOTE: test perf

    rank = args.gpus_per_node * node + local_rank

    dist_init(rank, args)

    train_dataset = FieldDataset(
        in_patterns=args.train_in_patterns,
        tgt_patterns=args.train_tgt_patterns,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        augment=args.augment,
        crop=args.crop,
        pad=args.pad,
        scale_factor=args.scale_factor,
        cache=args.cache,
        div_data=args.div_data,
        rank=rank,
        world_size=args.world_size,
    )
    if not args.div_data:
        try:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        except TypeError:
            train_sampler = DistributedSampler(train_dataset)  # old pytorch
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batches,
        shuffle=args.div_data,
        sampler=None if args.div_data else train_sampler,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    if args.val:
        val_dataset = FieldDataset(
            in_patterns=args.val_in_patterns,
            tgt_patterns=args.val_tgt_patterns,
            in_norms=args.in_norms,
            tgt_norms=args.tgt_norms,
            augment=False,
            crop=args.crop,
            pad=args.pad,
            scale_factor=args.scale_factor,
            cache=args.cache,
            div_data=args.div_data,
            rank=rank,
            world_size=args.world_size,
        )
        if not args.div_data:
            try:
                val_sampler = DistributedSampler(val_dataset, shuffle=False)
            except TypeError:
                val_sampler = DistributedSampler(val_dataset)  # old pytorch
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batches,
            shuffle=False,
            sampler=None if args.div_data else val_sampler,
            num_workers=args.loader_workers,
            pin_memory=True,
        )

    args.in_chan, args.out_chan = train_dataset.in_chan, train_dataset.tgt_chan

    model = getattr(models, args.model)
    model = model(sum(args.in_chan), sum(args.out_chan))
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[device],
            process_group=dist.new_group())

    criterion = getattr(nn, args.criterion)
    criterion = criterion()
    criterion.to(device)

    optimizer = getattr(torch.optim, args.optimizer)
    optimizer = optimizer(
        model.parameters(),
        lr=args.lr,
        #momentum=args.momentum,
        betas=(0, 0.99),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            factor=0.1, patience=10, verbose=True)

    adv_model = adv_criterion = adv_optimizer = adv_scheduler = None
    if args.adv:
        adv_model = getattr(models, args.adv_model)
        adv_model = adv_model_wrapper(adv_model)
        adv_model = adv_model(sum(args.in_chan + args.out_chan)
                if args.cgan else sum(args.out_chan), 1)
        if args.adv_model_spectral_norm:
            add_spectral_norm(adv_model)
        adv_model.to(device)
        adv_model = DistributedDataParallel(adv_model, device_ids=[device],
                process_group=dist.new_group())

        adv_criterion = getattr(nn, args.adv_criterion)
        adv_criterion = adv_criterion_wrapper(adv_criterion)
        adv_criterion = adv_criterion(reduction='min' if args.min_reduction else 'mean')
        adv_criterion.to(device)

        adv_optimizer = getattr(torch.optim, args.optimizer)
        adv_optimizer = adv_optimizer(
            adv_model.parameters(),
            lr=args.adv_lr,
            betas=(0.5, 0.999),
            weight_decay=args.adv_weight_decay,
        )
        adv_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adv_optimizer,
            factor=0.1, patience=10, verbose=True)

    if args.load_state:
        state = torch.load(args.load_state, map_location=device)

        start_epoch = state['epoch']

        load_model_state_dict(model.module, state['model'],
                strict=args.load_state_strict)

        if args.adv and 'adv_model' in state:
            load_model_state_dict(adv_model.module, state['adv_model'],
                    strict=args.load_state_strict)

        torch.set_rng_state(state['rng'].cpu())  # move rng state back

        if rank == 0:
            min_loss = state['min_loss']
            if args.adv and 'adv_model' not in state:
                min_loss = None  # restarting with adversary wipes the record
            print('state at epoch {} loaded from {}'.format(
                state['epoch'], args.load_state), flush=True)

        del state
    else:
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
        if args.init_weight_std is not None:
            model.apply(init_weights)
            if args.adv:
                adv_model.apply(init_weights)

        start_epoch = 0

        if rank == 0:
            min_loss = None

    torch.backends.cudnn.benchmark = True  # NOTE: test perf

    logger = None
    if rank == 0:
        logger = SummaryWriter()

    if rank == 0:
        pprint(vars(args))
        sys.stdout.flush()

    if args.adv:
        args.instance_noise = InstanceNoise(args.instance_noise,
                                            args.instance_noise_batches)

    for epoch in range(start_epoch, args.epochs):
        if not args.div_data:
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
            epoch_loss = val_loss

        if args.reduce_lr_on_plateau and epoch >= args.adv_start:
            scheduler.step(epoch_loss[0])
            if args.adv:
                adv_scheduler.step(epoch_loss[0])

        if rank == 0:
            try:
                logger.flush()
            except AttributeError:
                logger.close()  # old pytorch

            good = min_loss is None or epoch_loss[0] < min_loss[0]
            if good and epoch >= args.adv_start:
                min_loss = epoch_loss

            state = {
                'epoch': epoch + 1,
                'model': model.module.state_dict(),
                'rng': torch.get_rng_state(),
                'min_loss': min_loss,
            }
            if args.adv:
                state['adv_model'] = adv_model.module.state_dict()

            state_file = 'state_{}.pth'.format(epoch + 1)
            torch.save(state, state_file)
            del state

            ckpt_link = 'checkpoint.pth'
            tmp_link = '{}.pth'.format(time.time())
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
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input)
        if epoch == 0 and i == 0 and rank == 0:
            print('input.shape =', input.shape)
            print('output.shape =', output.shape)
            print('target.shape =', target.shape, flush=True)

        target = narrow_like(target, output)  # FIXME pad
        if hasattr(model.module, 'scale_factor') and model.module.scale_factor != 1:
            input = F.interpolate(input, scale_factor=model.module.scale_factor,
                    mode='trilinear', align_corners=False)
        input = narrow_like(input, output)

        loss = criterion(output, target)
        epoch_loss[0] += loss.item()

        if args.adv and epoch >= args.adv_start:
            noise_std = args.instance_noise.std()
            if noise_std > 0:
                noise = noise_std * torch.randn_like(output)
                output = output + noise.detach()
                noise = noise_std * torch.randn_like(target)
                target = target + noise.detach()
                del noise

            if args.cgan:
                output = torch.cat([input, output], dim=1)
                target = torch.cat([input, target], dim=1)

            # discriminator
            set_requires_grad(adv_model, True)

            eval = adv_model([output.detach(), target])
            adv_loss_fake, adv_loss_real = adv_criterion(eval, [fake, adv_real])
            epoch_loss[3] += adv_loss_fake.item()
            epoch_loss[4] += adv_loss_real.item()
            adv_loss = 0.5 * (adv_loss_fake + adv_loss_real)
            epoch_loss[2] += adv_loss.item()

            adv_optimizer.zero_grad()
            adv_loss.backward()
            adv_optimizer.step()

            # generator adversarial loss
            set_requires_grad(adv_model, False)

            eval_out = adv_model(output)
            loss_adv, = adv_criterion(eval_out, real)
            epoch_loss[1] += loss_adv.item()

            r = loss.item() / (loss_adv.item() + 1e-8)
            f = args.loss_fraction
            e = epoch - args.adv_start
            d = 0.5 ** (e / args.loss_halflife)
            loss = (f + (1 - f) * d) * loss + (1 - f) * (1 - d) * r * loss_adv

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch = epoch * len(loader) + i + 1
        if batch % args.log_interval == 0:
            dist.all_reduce(loss)
            loss /= world_size
            if rank == 0:
                logger.add_scalar('loss/batch/train', loss.item(),
                        global_step=batch)
                if args.adv and epoch >= args.adv_start:
                    logger.add_scalar('loss/batch/train/adv/G',
                            loss_adv.item(), global_step=batch)
                    logger.add_scalars('loss/batch/train/adv/D', {
                            'total': adv_loss.item(),
                            'fake': adv_loss_fake.item(),
                            'real': adv_loss_real.item(),
                        }, global_step=batch)

                # gradients of the weights of the first and the last layer
                grads = list(p.grad for n, p in model.named_parameters()
                        if '.weight' in n)
                grads = [grads[0], grads[-1]]
                grads = [g.detach().norm().item() for g in grads]
                logger.add_scalars('grad', {
                        'first': grads[0],
                        'last': grads[-1],
                    }, global_step=batch)
                if args.adv and epoch >= args.adv_start:
                    grads = list(p.grad for n, p in adv_model.named_parameters()
                            if '.weight' in n)
                    grads = [grads[0], grads[-1]]
                    grads = [g.detach().norm().item() for g in grads]
                    logger.add_scalars('grad/adv', {
                            'first': grads[0],
                            'last': grads[-1],
                        }, global_step=batch)

                if args.adv and epoch >= args.adv_start and noise_std > 0:
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
            logger.add_scalars('loss/epoch/train/adv/D', {
                    'total': epoch_loss[2],
                    'fake': epoch_loss[3],
                    'real': epoch_loss[4],
                }, global_step=epoch+1)

        skip_chan = 0
        if args.adv and epoch >= args.adv_start and args.cgan:
            skip_chan = sum(args.in_chan)
        logger.add_figure('fig/epoch/train', fig3d(
                input[-1],
                output[-1, skip_chan:],
                target[-1, skip_chan:],
                output[-1, skip_chan:] - target[-1, skip_chan:],
            ), global_step=epoch+1)

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

            target = narrow_like(target, output)  # FIXME pad
            if hasattr(model.module, 'scale_factor') and model.module.scale_factor != 1:
                input = F.interpolate(input, scale_factor=model.module.scale_factor,
                        mode='trilinear', align_corners=False)
            input = narrow_like(input, output)

            loss = criterion(output, target)
            epoch_loss[0] += loss.item()

            if args.adv and epoch >= args.adv_start:
                if args.cgan:
                    output = torch.cat([input, output], dim=1)
                    target = torch.cat([input, target], dim=1)

                # discriminator
                eval = adv_model([output, target])
                adv_loss_fake, adv_loss_real = adv_criterion(eval, [fake, real])
                epoch_loss[3] += adv_loss_fake.item()
                epoch_loss[4] += adv_loss_real.item()
                adv_loss = 0.5 * (adv_loss_fake + adv_loss_real)
                epoch_loss[2] += adv_loss.item()

                # generator adversarial loss
                eval_out, _ = adv_criterion.split_input(eval, [fake, real])
                loss_adv, = adv_criterion(eval_out, real)
                epoch_loss[1] += loss_adv.item()

    dist.all_reduce(epoch_loss)
    epoch_loss /= len(loader) * world_size
    if rank == 0:
        logger.add_scalar('loss/epoch/val', epoch_loss[0],
                global_step=epoch+1)
        if args.adv and epoch >= args.adv_start:
            logger.add_scalar('loss/epoch/val/adv/G', epoch_loss[1],
                    global_step=epoch+1)
            logger.add_scalars('loss/epoch/val/adv/D', {
                    'total': epoch_loss[2],
                    'fake': epoch_loss[3],
                    'real': epoch_loss[4],
                }, global_step=epoch+1)

        skip_chan = 0
        if args.adv and epoch >= args.adv_start and args.cgan:
            skip_chan = sum(args.in_chan)
        logger.add_figure('fig/epoch/val', fig3d(
                input[-1],
                output[-1, skip_chan:],
                target[-1, skip_chan:],
                output[-1, skip_chan:] - target[-1, skip_chan:],
            ), global_step=epoch+1)

    return epoch_loss


def set_runtime_default_args(args):
    args.val = args.val_in_patterns is not None and \
            args.val_tgt_patterns is not None

    args.adv = args.adv_model is not None

    if args.adv:
        if args.adv_lr is None:
            args.adv_lr = args.lr
        if args.adv_weight_decay is None:
            args.adv_weight_decay = args.weight_decay


def dist_init(rank, args):
    dist_file = 'dist_addr'

    if rank == 0:
        addr = socket.gethostname()

        with socket.socket() as s:
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

    if rank == 0:
        os.remove(dist_file)


def set_requires_grad(module, requires_grad=False):
    for param in module.parameters():
        param.requires_grad = requires_grad
