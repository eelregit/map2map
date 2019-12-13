import os
import shutil
import random
import torch
from torch.multiprocessing import spawn
from torch.distributed import init_process_group, destroy_process_group, all_reduce
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data import FieldDataset
from . import models
from .models import narrow_like


def node_worker(args):
    if args.seed is None:
        args.seed = random.randint(0, 65535)
    torch.manual_seed(args.seed)  # NOTE: why here not in gpu_worker?
    #torch.backends.cudnn.deterministic = True  # NOTE: test perf

    args.gpus_per_node = torch.cuda.device_count()
    args.nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    args.world_size = args.gpus_per_node * args.nodes

    node = int(os.environ['SLURM_NODEID'])
    if node == 0:
        print(args)
    args.node = node

    spawn(gpu_worker, args=(args,), nprocs=args.gpus_per_node)


def gpu_worker(local_rank, args):
    args.device = torch.device('cuda', local_rank)
    torch.cuda.device(args.device)

    args.rank = args.gpus_per_node * args.node + local_rank

    init_process_group(
        backend=args.dist_backend,
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank
    )

    train_dataset = FieldDataset(
        in_patterns=args.train_in_patterns,
        tgt_patterns=args.train_tgt_patterns,
        augment=args.augment,
        norms=args.norms,
        pad_or_crop=args.pad_or_crop,
    )
    #train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batches,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.loader_workers,
        pin_memory=True
    )

    val_dataset = FieldDataset(
        in_patterns=args.val_in_patterns,
        tgt_patterns=args.val_tgt_patterns,
        augment=False,
        norms=args.norms,
        pad_or_crop=args.pad_or_crop,
    )
    #val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batches,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.loader_workers,
        pin_memory=True
    )

    in_channels, out_channels = train_dataset.channels

    model = models.__dict__[args.model](in_channels, out_channels)
    model.to(args.device)
    model = DistributedDataParallel(model, device_ids=[args.device])

    criterion = torch.nn.__dict__[args.criterion]()
    criterion.to(args.device)

    optimizer = torch.optim.__dict__[args.optimizer](
        model.parameters(),
        lr=args.lr,
        #momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            factor=0.1, verbose=True)

    if args.load_state:
        state = torch.load(args.load_state, map_location=args.device)
        args.start_epoch = state['epoch']
        model.module.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng'].cpu())  # move rng state back
        if args.rank == 0:
            min_loss = state['min_loss']
            print('checkpoint at epoch {} loaded from {}'.format(
                state['epoch'], args.load_state))
        del state
    else:
        args.start_epoch = 0
        if args.rank == 0:
            min_loss = None

    torch.backends.cudnn.benchmark = True  # NOTE: test perf

    if args.rank == 0:
        args.logger = SummaryWriter()
        #hparam = {k: v if isinstance(v, (int, float, str, bool, torch.Tensor))
        #        else str(v) for k, v in vars(args).items()}
        #args.logger.add_hparams(hparam_dict=hparam, metric_dict={})

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        train(epoch, train_loader, model, criterion, optimizer, scheduler, args)

        val_loss = validate(epoch, val_loader, model, criterion, args)

        scheduler.step(val_loss)

        if args.rank == 0:
            print(end='', flush=True)
            args.logger.close()

            state = {
                'epoch': epoch + 1,
                'model': model.module.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'rng' : torch.get_rng_state(),
                'min_loss': min_loss,
            }
            ckpt_file = 'checkpoint.pth'
            best_file = 'best_model_{}.pth'
            torch.save(state, ckpt_file)
            del state

            if min_loss is None or val_loss < min_loss:
                min_loss = val_loss
                shutil.copyfile(ckpt_file, best_file.format(epoch + 1))
                if os.path.isfile(best_file.format(epoch)):
                    os.remove(best_file.format(epoch))

    destroy_process_group()


def train(epoch, loader, model, criterion, optimizer, scheduler, args):
    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        output = model(input)
        target = narrow_like(target, output)  # FIXME pad

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if scheduler is not None:  # for batch scheduler
            #scheduler.step()

        batch = epoch * len(loader) + i + 1
        if batch % args.log_interval == 0:
            all_reduce(loss)
            loss /= args.world_size
            if args.rank == 0:
                args.logger.add_scalar('loss/train', loss.item(), global_step=batch)


def validate(epoch, loader, model, criterion, args):
    model.eval()

    loss = 0

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            output = model(input)
            target = narrow_like(target, output)  # FIXME pad

            loss += criterion(output, target)

    all_reduce(loss)
    loss /= len(loader) * args.world_size
    if args.rank == 0:
        args.logger.add_scalar('loss/val', loss.item(), global_step=epoch+1)

    return loss.item()
