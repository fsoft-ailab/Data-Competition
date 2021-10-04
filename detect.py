import argparse
import sys
from pathlib import Path

import cv2
import torch

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, colorstr, is_ascii, \
    non_max_suppression, scale_coords, xyxy2xywh, set_logging, increment_path, \
    save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync


FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())


@torch.no_grad()
def run(weights,  # model.pt path(s)
        source,  # file/dir
        img_size,  # inference size (pixels)
        conf_threshold,  # confidence threshold
        iou_threshold,  # NMS IOU threshold
        max_det,  # maximum detections per image
        device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img,  # show results
        save_txt,  # save results to *.txt
        save_conf,  # save confidences in --save-txt labels
        save_crop,  # save cropped prediction boxes
        nosave,  # do not save images
        classes,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms,  # class-agnostic NMS
        augment,  # augmented inference
        visualize,  # visualize features
        dir,  # save results to results/detect/
        exist_ok,  # existing results/detect/ ok, do not increment
        line_thickness,  # bounding box thickness (pixels)
        hide_labels,  # hide labels
        hide_conf,  # hide confidences
        half,  # use FP16 half-precision inference
        ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path(dir), exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    suffix = Path(w).suffix.lower()
    assert suffix == ".pt"

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    img_size = check_img_size(img_size, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    # Dataloader
    dataset = LoadImages(source, img_size=img_size, stride=stride, auto=True)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, *img_size).to(device).type_as(next(model.parameters())))  # run once
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, img, im0s, _ in dataset:
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference

        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(img, augment=augment, visualize=visualize)[0]
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_threshold, iou_threshold, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            print(f'{s}Done. ({t3 - t2:.3f}s)')

            im0 = annotator.result()
            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *img_size)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}")


def parser():
    args = argparse.ArgumentParser()
    args.add_argument('--weights', type=str, help='specify your weight path', required=True)
    args.add_argument('--source', type=str, help='folder contain image', required=True)
    args.add_argument('--dir',type=str, help='save results to dir', required=True)
    args.add_argument('--conf-threshold', type=float, default=0.25, help='confidence threshold')
    args.add_argument('--iou-threshold', type=float, default=0.6, help='NMS IoU threshold')
    args.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    args.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    args.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    args.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    args.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    args.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    args = args.parse_args()

    args.agnostic_nms = False
    args.augment = False
    args.classes = None
    args.exist_ok = False
    args.img_size = [640, 640]
    args.nosave = False
    args.view_img = False
    args.visualize = False
    args.max_det = 1000
    args.line_thickness = 2

    return args


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    main(parser())
