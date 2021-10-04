import sys
import argparse
from pathlib import Path
from threading import Thread

import torch
import numpy as np
from tqdm import tqdm

from utils.callbacks import Callbacks
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.plots import plot_images, output_to_target
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.torch_utils import select_device, time_sync
from utils.general import check_dataset, check_img_size, check_suffix, check_yaml, box_iou,\
    non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr


FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())


def save_one_txt(normed_pred, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain wh x wh
    for *xyxy, conf, cls in normed_pred.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized x y w h
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def process_batch(detections, labels, iou_thresholds):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
        iou_thresholds: list iou thresholds from 0.5 -> 0.95
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iou_thresholds.shape[0], dtype=torch.bool, device=iou_thresholds.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    x = torch.where((iou >= iou_thresholds[0]) & (labels[:, 0:1] == detections[:, 5]))
    if x[0].shape[0]:
        # [label, detection, iou]
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iou_thresholds.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iou_thresholds

    return correct


def cal_weighted_ap(ap50):
    return 0.2 * ap50[1] + 0.3 * ap50[0] + 0.5 * ap50[2]


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        img_size=640,  # inference size (pixels)
        conf_threshold=0.001,  # confidence threshold
        iou_threshold=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        project='results/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        ):

    # Initialize/load model and set device
    is_loaded_model = model is not None
    grid_size = None

    if is_loaded_model:
        device = next(model.parameters()).device
    else:
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

        # Load model
        check_suffix(weights, '.pt')
        model = attempt_load(weights, map_location=device)
        grid_size = max(int(model.stride.max()), 32)
        img_size = check_img_size(img_size, s=grid_size)

        # Data
        data = check_dataset(data)

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    num_class = int(data['num_class'])
    iou_thresholds = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    num_thresholds = iou_thresholds.numel()

    # Dataloader
    if not is_loaded_model:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))
        task = task if task in ('train', 'val', 'test') else 'val'
        dataloader = create_dataloader(data[task], img_size, batch_size, grid_size, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    num_per_class = [0] * num_class

    confusion_matrix = ConfusionMatrix(nc=num_class)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    s = ('%20s' + '%11s' * 8) % ('Class', 'Images', 'Labels', 'Boxes', 'P', 'R', 'wAP@.5', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map, wap50 = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    loss = torch.zeros(3, device=device)
    stats, ap, ap_class = [], [], []

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t1 = time_sync()

        # Preprocess
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0

        for i in range(num_class):
            num_per_class[i] += len(np.where(targets[:, 1] == i)[0])
        targets = targets.to(device)

        batch_size, _, height, width = img.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Run model
        out, train_out = model(img, augment=augment)  # inference and training outputs
        dt[1] += time_sync() - t2

        # Compute loss
        if compute_loss:
            # box, obj, cls
            loss += compute_loss([x.float() for x in train_out], targets)[1]

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(batch_size)] if save_hybrid else []
        t3 = time_sync()

        # Note depth 8 -> 6
        out = non_max_suppression(out, conf_threshold, iou_threshold, labels=lb, multi_label=True)
        dt[2] += time_sync() - t3

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            target_class = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, num_thresholds, dtype=torch.bool),
                                  torch.Tensor(), torch.Tensor(), target_class))
                continue

            normed_pred = pred.clone()
            scale_coords(img[si].shape[1:], normed_pred[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                target_boxes = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[si].shape[1:], target_boxes, shape, shapes[si][1])  # native-space labels
                labels_per_img = torch.cat((labels[:, 0:1], target_boxes), 1)  # native-space labels
                correct = process_batch(normed_pred, labels_per_img, iou_thresholds)
                if plots:
                    confusion_matrix.process_batch(normed_pred, labels_per_img)
            else:
                correct = torch.zeros(pred.shape[0], num_thresholds, dtype=torch.bool)

            # correct, confidence, pred_label, target_label
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target_class))

            # Save/log
            if save_txt:
                save_one_txt(normed_pred, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            callbacks.run('on_val_image_end', pred, normed_pred, path, names, img[si])

        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]

    # Count detected boxes per class
    boxes_per_class = np.bincount(stats[2].astype(np.int64), minlength=num_class)
    ap50 = None

    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, wap50, map50, map = p.mean(), r.mean(), cal_weighted_ap(ap50), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=num_class)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    print_format = '%20s' + '%11i' * 3 + '%11.3g' * 5  # print format
    print(print_format % ('all', seen, nt.sum(), sum(boxes_per_class), mp, mr, wap50, map50, map))

    # Print results per class
    if (verbose or (num_class < 50 and not is_loaded_model)) and num_class > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(print_format % (names[c], num_per_class[i], nt[c],
                                  boxes_per_class[i], p[i], r[i], ap50[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)
    if not is_loaded_model:
        shape = (batch_size, 3, img_size, img_size)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Return results
    model.float()
    if not is_loaded_model:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(num_class) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, wap50, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parser():
    args = argparse.ArgumentParser(prog='val.py')
    args.add_argument('--data', type=str, default='config/data_cfg.yaml', help='dataset.yaml path')
    args.add_argument('--weights', type=str, help='specify your weight path', required=True)
    args.add_argument('--task', help='train, val, test', required=True)
    args.add_argument('--name', help='save to project/name', required=True)
    args.add_argument('--batch-size', type=int, default=64, help='batch size')
    args.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = args.parse_args()

    args.img_size = 640
    args.conf_threshold = 0.001
    args.iou_threshold = 0.6
    args.augment = False
    args.exist_ok = False
    args.half = False
    args.project = 'results/evaluate/' + args.task
    args.save_conf = False
    args.save_hybrid = False
    args.save_txt = False
    args.verbose = False
    args.plots = True

    args.save_txt |= args.save_hybrid
    args.data = check_yaml(args.data)

    return args


def main(args):
    set_logging()
    print(colorstr('val: ') + ', '.join(f'{k}={v}' for k, v in vars(args).items()))

    if args.task in ('train', 'val', 'test'):  # run normally
        run(**vars(args))


if __name__ == "__main__":
    main(parser())
