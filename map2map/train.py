import os
import shutil
from pprint import pprint
import torch
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
from .models import narrow_like
from .models.adversary import adv_model_wrapper, adv_criterion_wrapper
from .state import load_model_state_dict


def node_worker(args):
    torch.manual_seed(args.seed)  # NOTE: why here not in gpu_worker?
    #torch.backends.cudnn.deterministic = True  # NOTE: test perf

    args.gpus_per_node = torch.cuda.device_count()
    args.nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    args.world_size = args.gpus_per_node * args.nodes

    node = int(os.environ['SLURM_NODEID'])
    if node == 0:
        pprint(vars(args))
    args.node = node

    spawn(gpu_worker, args=(args,), nprocs=args.gpus_per_node)


def gpu_worker(local_rank, args):
    args.device = torch.device('cuda', local_rank)
    torch.cuda.device(args.device)

    args.rank = args.gpus_per_node * args.node + local_rank

    dist.init_process_group(
        backend=args.dist_backend,
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank
    )

    train_dataset = FieldDataset(
        in_patterns=args.train_in_patterns,
        tgt_patterns=args.train_tgt_patterns,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        augment=args.augment,
        crop=args.crop,
        pad=args.pad,
        scale_factor=args.scale_factor,
        noise_chan=args.noise_chan,
        cache=args.cache,
        div_data=args.div_data,
        rank=args.rank,
        world_size=args.world_size,
    )
    if not args.div_data:
        #train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batches,
        shuffle=args.div_data,
        sampler=None if args.div_data else train_sampler,
        num_workers=args.loader_workers,
        pin_memory=True
    )

    args.val = args.val_in_patterns is not None and \
            args.val_tgt_patterns is not None
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
            noise_chan=args.noise_chan,
            cache=args.cache,
            div_data=args.div_data,
            rank=args.rank,
            world_size=args.world_size,
        )
        if not args.div_data:
            #val_sampler = DistributedSampler(val_dataset, shuffle=False)
            val_sampler = DistributedSampler(val_dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batches,
            shuffle=False,
            sampler=None if args.div_data else val_sampler,
            num_workers=args.loader_workers,
            pin_memory=True
        )

    args.in_chan, args.out_chan = train_dataset.in_chan, train_dataset.tgt_chan

    model = getattr(models, args.model)
    model = model(sum(args.in_chan) + args.noise_chan, sum(args.out_chan))
    model.to(args.device)
    model = DistributedDataParallel(model, device_ids=[args.device],
            process_group=dist.new_group())

    criterion = getattr(torch.nn, args.criterion)
    criterion = criterion()
    criterion.to(args.device)

    optimizer = getattr(torch.optim, args.optimizer)
    optimizer = optimizer(
        model.parameters(),
        lr=args.lr,
        #momentum=args.momentum,
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            factor=0.1, patience=10, verbose=True)

    adv_model = adv_criterion = adv_optimizer = adv_scheduler = None
    args.adv = args.adv_model is not None
    if args.adv:
        adv_model = getattr(models, args.adv_model)
        adv_model = adv_model_wrapper(adv_model)
        adv_model = adv_model(sum(args.in_chan + args.out_chan)
                if args.cgan else sum(args.out_chan), 1)
        adv_model.to(args.device)
        adv_model = DistributedDataParallel(adv_model, device_ids=[args.device],
                process_group=dist.new_group())

        adv_criterion = getattr(torch.nn, args.adv_criterion)
        adv_criterion = adv_criterion_wrapper(adv_criterion)
        adv_criterion = adv_criterion(reduction='min' if args.min_reduction else 'mean')
        adv_criterion.to(args.device)

        if args.adv_lr is None:
            args.adv_lr = args.lr
        if args.adv_weight_decay is None:
            args.adv_weight_decay = args.weight_decay

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
        state = torch.load(args.load_state, map_location=args.device)
        args.start_epoch = state['epoch']
        args.adv_delay += args.start_epoch
        load_model_state_dict(model.module, state['model'],
                strict=args.load_state_strict)
        if 'adv_model' in state and args.adv:
            load_model_state_dict(adv_model.module, state['adv_model'],
                    strict=args.load_state_strict)
        torch.set_rng_state(state['rng'].cpu())  # move rng state back
        if args.rank == 0:
            min_loss = state['min_loss']
            if 'adv_model' not in state and args.adv:
                min_loss = None  # restarting with adversary wipes the record
            print('checkpoint at epoch {} loaded from {}'.format(
                state['epoch'], args.load_state))
        del state
    else:
#        def init_weights(m):
#            classname = m.__class__.__name__
#            if isinstance(m, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
#                m.weight.data.normal_(0.0, 0.02)
#            elif isinstance(m, torch.nn.BatchNorm3d):
#                m.weight.data.normal_(1.0, 0.02)
#                m.bias.data.fill_(0)
#        model.apply(init_weights)
#
        args.start_epoch = 0
        if args.rank == 0:
            min_loss = None

    torch.backends.cudnn.benchmark = True  # NOTE: test perf

    if args.rank == 0:
        args.logger = SummaryWriter()

    for epoch in range(args.start_epoch, args.epochs):
        if not args.div_data:
            train_sampler.set_epoch(epoch)

        train_loss = train(epoch, train_loader,
            model, criterion, optimizer, scheduler,
            adv_model, adv_criterion, adv_optimizer, adv_scheduler,
            args)
        epoch_loss = train_loss

        if args.val:
            val_loss = validate(epoch, val_loader,
                model, criterion, adv_model, adv_criterion,
                args)
            epoch_loss = val_loss

        if epoch >= args.adv_delay:
            scheduler.step(epoch_loss[0])
            if args.adv:
                adv_scheduler.step(epoch_loss[0])
        else:
            scheduler.last_epoch = epoch
            if args.adv:
                adv_scheduler.last_epoch = epoch

        if args.rank == 0:
            print(end='', flush=True)
            args.logger.close()

            good = min_loss is None or epoch_loss[0] < min_loss[0]
            if good and epoch >= args.adv_delay:
                min_loss = epoch_loss

            state = {
                'epoch': epoch + 1,
                'model': model.module.state_dict(),
                'rng': torch.get_rng_state(),
                'min_loss': min_loss,
            }
            if args.adv:
                state.update({
                    'adv_model': adv_model.module.state_dict(),
                })
            ckpt_file = 'checkpoint.pth'
            state_file = 'state_{}.pth'
            torch.save(state, ckpt_file)
            del state

            if good:
                shutil.copyfile(ckpt_file, state_file.format(epoch + 1))
                #if os.path.isfile(state_file.format(epoch)):
                #    os.remove(state_file.format(epoch))

    dist.destroy_process_group()


def train(epoch, loader, model, criterion, optimizer, scheduler,
        adv_model, adv_criterion, adv_optimizer, adv_scheduler, args):
    model.train()
    if args.adv:
        adv_model.train()

    # loss, loss_adv, adv_loss, adv_loss_fake, adv_loss_real
    # loss: generator (model) supervised loss
    # loss_adv: generator (model) adversarial loss
    # adv_loss: discriminator (adv_model) loss
    epoch_loss = torch.zeros(5, dtype=torch.float64, device=args.device)
    real = torch.ones(1, dtype=torch.float32, device=args.device)
    fake = torch.zeros(1, dtype=torch.float32, device=args.device)

    for i, (input, target) in enumerate(loader):
        input = input.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        output = model(input)
        if args.noise_chan > 0:
            input = input[:, :-args.noise_chan]  # remove noise channels

        target = narrow_like(target, output)  # FIXME pad
        loss = criterion(output, target)
        epoch_loss[0] += loss.item()

        # generator adversarial loss
        if args.adv:
            if args.cgan:
                if hasattr(model, 'scale_factor') and model.scale_factor != 1:
                    input = F.interpolate(input,
                            scale_factor=model.scale_factor, mode='trilinear')
                input = narrow_like(input, output)
                output = torch.cat([input, output], dim=1)
                target = torch.cat([input, target], dim=1)

            eval_out = adv_model(output)
            loss_adv, = adv_criterion(eval_out, real)
            epoch_loss[1] += loss_adv.item()

            if epoch >= args.adv_delay:
                loss_fac = loss.item() / (loss_adv.item() + 1e-8)
                loss += loss_fac * (loss_adv - loss_adv.item())  # FIXME does this work?

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # discriminator
        if args.adv:
            eval = adv_model([output.detach(), target])
            adv_loss_fake, adv_loss_real = adv_criterion(eval, [fake, real])
            epoch_loss[3] += adv_loss_fake.item()
            epoch_loss[4] += adv_loss_real.item()
            adv_loss = 0.5 * (adv_loss_fake + adv_loss_real)
            epoch_loss[2] += adv_loss.item()

            adv_optimizer.zero_grad()
            adv_loss.backward()
            adv_optimizer.step()

        batch = epoch * len(loader) + i + 1
        if batch % args.log_interval == 0:
            dist.all_reduce(loss)
            loss /= args.world_size
            if args.rank == 0:
                args.logger.add_scalar('loss/batch/train', loss.item(),
                        global_step=batch)
                if args.adv:
                    args.logger.add_scalar('loss/batch/train/adv/G',
                            loss_adv.item(), global_step=batch)
                    args.logger.add_scalars('loss/batch/train/adv/D', {
                            'total': adv_loss.item(),
                            'fake': adv_loss_fake.item(),
                            'real': adv_loss_real.item(),
                        }, global_step=batch)

    dist.all_reduce(epoch_loss)
    epoch_loss /= len(loader) * args.world_size
    if args.rank == 0:
        args.logger.add_scalar('loss/epoch/train', epoch_loss[0],
                global_step=epoch+1)
        if args.adv:
            args.logger.add_scalar('loss/epoch/train/adv/G', epoch_loss[1],
                    global_step=epoch+1)
            args.logger.add_scalars('loss/epoch/train/adv/D', {
                    'total': epoch_loss[2],
                    'fake': epoch_loss[3],
                    'real': epoch_loss[4],
                }, global_step=epoch+1)

        skip_chan = sum(args.in_chan) if args.adv and args.cgan else 0
        args.logger.add_figure('fig/epoch/train/in',
                fig3d(narrow_like(input, output)[-1]), global_step =epoch+1)
        args.logger.add_figure('fig/epoch/train/out',
                fig3d(output[-1, skip_chan:], target[-1, skip_chan:],
                    output[-1, skip_chan:] - target[-1, skip_chan:]),
                global_step =epoch+1)

    return epoch_loss


def validate(epoch, loader, model, criterion, adv_model, adv_criterion, args):
    model.eval()
    if args.adv:
        adv_model.eval()

    epoch_loss = torch.zeros(5, dtype=torch.float64, device=args.device)
    fake = torch.zeros(1, dtype=torch.float32, device=args.device)
    real = torch.ones(1, dtype=torch.float32, device=args.device)

    with torch.no_grad():
        for input, target in loader:
            input = input.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            output = model(input)
            if args.noise_chan > 0:
                input = input[:, :-args.noise_chan]  # remove noise channels

            target = narrow_like(target, output)  # FIXME pad
            loss = criterion(output, target)
            epoch_loss[0] += loss.item()

            if args.adv:
                if args.cgan:
                    if hasattr(model, 'scale_factor') and model.scale_factor != 1:
                        input = F.interpolate(input,
                                scale_factor=model.scale_factor, mode='trilinear')
                    input = narrow_like(input, output)
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
    epoch_loss /= len(loader) * args.world_size
    if args.rank == 0:
        args.logger.add_scalar('loss/epoch/val', epoch_loss[0],
                global_step=epoch+1)
        if args.adv:
            args.logger.add_scalar('loss/epoch/val/adv/G', epoch_loss[1],
                    global_step=epoch+1)
            args.logger.add_scalars('loss/epoch/val/adv/D', {
                    'total': epoch_loss[2],
                    'fake': epoch_loss[3],
                    'real': epoch_loss[4],
                }, global_step=epoch+1)

        skip_chan = sum(args.in_chan) if args.adv and args.cgan else 0
        args.logger.add_figure('fig/epoch/val/in',
                fig3d(narrow_like(input, output)[-1]), global_step =epoch+1)
        args.logger.add_figure('fig/epoch/val',
                fig3d(output[-1, skip_chan:], target[-1, skip_chan:],
                    output[-1, skip_chan:] - target[-1, skip_chan:]),
                global_step =epoch+1)

    return epoch_loss
