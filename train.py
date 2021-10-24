import sys
import time
import math
import random
import logging
import argparse
from pathlib import Path
from copy import deepcopy

import yaml
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.cuda import amp
import torch.nn.functional as F
from torch.optim import Adam, SGD, lr_scheduler

import val
from models.yolo import Model
from utils.loss import ComputeLoss
from utils.plots import plot_labels, plot_lr_scheduler
from utils.metrics import fitness
from utils.loggers import Loggers
from utils.callbacks import Callbacks
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, intersect_dicts, select_device
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, check_dataset, check_img_size, check_requirements, check_file,\
    check_yaml, check_suffix, one_cycle, colorstr, methods, set_logging


FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())
LOGGER = logging.getLogger(__name__)


def train(hyp,
          args,
          device,
          callbacks
          ):
    [save_dir, epochs, batch_size, pretrained_path,
     evolve, data_cfg, model_cfg, resume, no_val, no_save, workers] = Path(args.save_dir), args.epochs, \
                                                                      args.batch_size, args.weights, \
                                                                      args.evolve, args.data_cfg, args.model_cfg, \
                                                                      args.resume, args.noval, args.nosave, args.workers

    # Directories
    weight_path = save_dir / 'weights'  # weights dir
    weight_path.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = weight_path / 'last.pt', weight_path / 'best.pt'

    # Hyper parameters
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp: dict = yaml.safe_load(f)  # load hyper parameter dict
    LOGGER.info(colorstr('Hyper parameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(args), f, sort_keys=False)

    # Loggers
    loggers = Loggers(save_dir, pretrained_path, args, hyp, LOGGER)

    # Register actions
    for k in methods(loggers):
        callbacks.register_action(k, callback=getattr(loggers, k))

    """
    ===============================
        Config
    ===============================
    """
    plots: bool = not evolve
    cuda: bool = device.type != 'cpu'
    init_seeds(0)

    data_dict = check_dataset(data_cfg)
    train_path, val_path = data_dict['train'], data_dict['val']
    num_class = int(data_dict['num_class'])  # number of classes
    class_name = data_dict['names']

    """
    ===============================
        Model
    ===============================
    """
    check_suffix(pretrained_path, '.pt')
    use_pretrained = pretrained_path.endswith('.pt')
    check_point = None
    if use_pretrained:
        check_point = torch.load(pretrained_path, map_location=device)  # load checkpoint

        # create model
        model = Model(model_cfg or check_point['model'].yaml, ch=3, nc=num_class, anchors=hyp.get('anchors')).to(device)
        exclude = ['anchor'] if (model_cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = check_point['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {pretrained_path}')  # report
    else:
        # create model
        model = Model(model_cfg, ch=3, nc=num_class, anchors=hyp.get('anchors')).to(device)

    """
    ===============================
        Optimizer
    ===============================
    """
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if args.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    LOGGER.info(f"{colorstr('Optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2

    # Scheduler
    if args.linear_lr:
        lr_lambda = lambda y: (1 - y / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lr_lambda = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model)

    start_epoch, best_fitness = 0, 0.0
    if use_pretrained:
        # Optimizer
        if check_point['optimizer'] is not None:
            optimizer.load_state_dict(check_point['optimizer'])
            best_fitness = check_point['best_fitness']

        # EMA
        if ema and check_point.get('ema'):
            ema.ema.load_state_dict(check_point['ema'].float().state_dict())
            ema.updates = check_point['updates']

        # Epochs
        start_epoch = check_point['epoch'] + 1
        if resume:
            assert start_epoch > 0, f'{pretrained_path} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            LOGGER.info("{} has been trained for {} epochs. Fine-tuning for {} more epochs.".format(
                pretrained_path,
                check_point['epoch'],
                epochs
            ))

        del check_point, csd

    # Image sizes
    grid_size = max(int(model.stride.max()), 32)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    img_size = check_img_size(args.img_size, grid_size, floor=grid_size * 2)  # verify img_size is gs-multiple

    # Train Loader
    train_loader, dataset = create_dataloader(train_path, img_size, batch_size, grid_size,
                                              hyp=hyp, augment=False, cache=args.cache, rect=args.rect,
                                              workers=workers, image_weights=args.image_weights, quad=args.quad,
                                              prefix=colorstr('Train: '))

    max_label_class = int(np.concatenate(dataset.labels, 0)[:, 0].max())
    num_batches = len(train_loader)
    assert max_label_class < num_class, \
        'Label class {} exceeds num_class={} in {}. Possible class labels are 0-{}'.format(
            max_label_class,
            num_class,
            data_cfg,
            num_class - 1
        )

    # Val Loader
    val_loader = create_dataloader(val_path, img_size, batch_size * 2, grid_size,
                                   hyp=hyp, cache=None if no_val else args.cache, rect=True,
                                   workers=workers, pad=0.5,
                                   prefix=colorstr('Val: '))[0]

    if not resume:
        labels = np.concatenate(dataset.labels, 0)

        if plots:
            plot_labels(labels, class_name, save_dir)

        # Anchors
        if not args.noautoanchor:
            check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=img_size)
        model.half().float()  # pre-reduce anchor precision

    callbacks.run('on_pretrain_routine_end')

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= num_class / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (img_size / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = args.label_smoothing
    model.nc = num_class  # attach number of classes to model
    model.hyp = hyp  # attach hyper parameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, num_class).to(
        device) * num_class  # attach class weights
    model.names = class_name

    # Start training
    t0 = time.time()
    num_warmup_inters = min(round(hyp['warmup_epochs'] * num_batches), 1000)
    last_opt_step = -1
    maps = np.zeros(num_class)
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=args.patience)
    compute_loss = ComputeLoss(model)  # init loss class
    LOGGER.info(f'Image sizes {img_size} train, {img_size} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')

    final_epoch = 0
    for epoch in range(start_epoch, epochs):
        final_epoch = max(final_epoch, epoch)
        model.train()
        if args.image_weights:
            class_weight = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / num_class
            image_weight = labels_to_image_weights(dataset.labels, nc=num_class, class_weights=class_weight)
            dataset.indices = random.choices(range(dataset.n), weights=image_weight, k=dataset.n)  # rand weighted idx

        mean_losses = torch.zeros(3, device=device)

        plot_bar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))

        plot_bar = tqdm(plot_bar, total=num_batches)
        optimizer.zero_grad()

        for i, (img_batch, targets, paths, _) in plot_bar:
            num_inters = i + num_batches * epoch

            # Preprocess
            img_batch = img_batch.to(device, non_blocking=True).float() / 255.0

            # Warmup
            if num_inters <= num_warmup_inters:
                xi = [0, num_warmup_inters]  # x interp

                accumulate = max(1, np.interp(num_inters, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(num_inters, xi,
                                        [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lr_lambda(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(num_inters, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if args.multi_scale:
                size = random.randrange(img_size * 0.5, img_size * 1.5 + grid_size) // grid_size * grid_size
                scale_factor = size / max(img_batch.shape[2:])
                if scale_factor != 1:
                    new_shape = [math.ceil(x * scale_factor / grid_size) * grid_size for x in img_batch.shape[2:]]
                    img_batch = F.interpolate(img_batch, size=new_shape, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(img_batch)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))

                if args.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if num_inters - last_opt_step >= accumulate:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = num_inters

            # Log
            # Update mean losses
            mean_losses = (mean_losses * i + loss_items) / (i + 1)
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            plot_bar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                f'{epoch}/{epochs - 1}', mem, *mean_losses, targets.shape[0], img_batch.shape[-1]))
            callbacks.run('on_train_batch_end', num_inters, model, img_batch, targets, paths, plots, args.sync_bn)

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        # mAP
        callbacks.run('on_train_epoch_end', epoch=epoch)
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
        final_epoch = (epoch + 1 == epochs) or stopper.possible_stop

        if not no_val or final_epoch:  # Calculate mAP
            results, maps, _ = val.run(data_dict,
                                       batch_size=batch_size * 2,
                                       img_size=img_size,
                                       model=ema.ema,
                                       dataloader=val_loader,
                                       save_dir=save_dir,
                                       verbose=num_class < 50 and final_epoch,
                                       plots=plots and final_epoch,
                                       callbacks=callbacks,
                                       compute_loss=compute_loss)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, wAP@.5, mAP@.5, mAP@.5-.95]
        if fi > best_fitness:
            best_fitness = fi
        log_val = list(mean_losses) + list(results) + lr
        callbacks.run('on_fit_epoch_end', log_val, epoch, best_fitness, fi)

        # Save model
        if (not no_save) or (final_epoch and not evolve):  # if save
            check_point = {'epoch': epoch,
                           'best_fitness': best_fitness,
                           'model': deepcopy(de_parallel(model)).half(),
                           'ema': deepcopy(ema.ema).half(),
                           'updates': ema.updates,
                           'optimizer': optimizer.state_dict()}

            # Save last, best and delete
            torch.save(check_point, last)
            if best_fitness == fi:
                torch.save(check_point, best)
            del check_point
            callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # Stop Single-GPU
        if stopper(epoch=epoch, fitness=fi):
            break

    # End training
    LOGGER.info('{} epochs completed in {:.3f} hours.'.format(
        final_epoch - start_epoch + 1,
        (time.time() - t0) / 3600
    ))

    if not evolve:
        # Strip optimizers
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
    callbacks.run('on_train_end', last, best, plots, final_epoch)
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    # Release gpu memory
    torch.cuda.empty_cache()

    return results


def parser(known=False):
    args = argparse.ArgumentParser()
    args.add_argument('--data_cfg', type=str, default='config/data_cfg.yaml', help='dataset config file path')
    args.add_argument('--batch-size', type=int, default=64, help='batch size')
    args.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    args.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    args.add_argument('--name', type=str, help='define your version experience', required=True)
    args = args.parse_known_args()[0] if known else args.parse_args()

    with open(Path('config') / 'train_cfg.yaml') as f:
        temp_args: dict = yaml.safe_load(f)

    keys = list(temp_args.keys())
    already_keys = list(args.__dict__.keys())

    for key in keys:
        if key not in already_keys:
            args.__setattr__(key, temp_args[key])

    return args


def main(args, callbacks=Callbacks()):

    set_logging()
    print(colorstr('Train: ') + ', '.join(f'{k}={v}' for k, v in vars(args).items()))

    # Check requirements
    check_requirements(requirements=FILE.parent / 'requirements.txt', exclude=['thop'])

    args.data_cfg = check_file(args.data_cfg)
    args.model_cfg = check_yaml(args.model_cfg)
    args.hyp = check_yaml(args.hyp)
    assert len(args.model_cfg) or len(args.weights), 'either --cfg or --weights must be specified'

    args.save_dir = str(increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok))

    # DDP mode
    device = select_device(args.device, batch_size=args.batch_size)
    print(device)

    train(args.hyp, args, device, callbacks)


if __name__ == "__main__":
    main(args=parser())
