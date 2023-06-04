from ultralytics import YOLO
import argparse
import torch


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--model_name', default="yolov8n", type=str)
    parser.add_argument('--data_path', default="", type=str)
    parser.add_argument('--project', default="", type=str)
    parser.add_argument('--name', default="", type=str)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr0', default=0.001, type=float)
    parser.add_argument('--batch', default=8, type=int)
    parser.add_argument('--imgsz', default=1280, type=int)
    parser.add_argument('--workers', default=4, type=int)
    
    parser.add_argument('--model', type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dpi', default=72, type=int)


    parser.add_argument('--optimizer', default="AdamW", type=str)

    # augmentations
    parser.add_argument('--hsv_h', default=0.015, type=float)
    parser.add_argument('--hsv_s', default=0.7, type=float)
    parser.add_argument('--hsv_v', default=0.4, type=float)
    parser.add_argument('--translate', default=0.1, type=float)
    parser.add_argument('--scale', default=0.5, type=float)
    parser.add_argument('--fliplr', default=0.5, type=float)
    parser.add_argument('--mosaic', default=1.0, type=float)

    parser.add_argument('--ch', default=6, type=int)
    parser.add_argument('--char_grid_encoder', default="three_digit_0", type=str, help="Options: one_digit, three_digit_0, three_digit_1")

    parser.add_argument('--save_period', default=-1, type=int)

    return parser


if __name__=="__main__":
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    print(args)

    model_args = {
        "data": args.data_path,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
        "lr0": args.lr0,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "char_grid_encoder": args.char_grid_encoder,
        "save_period": args.save_period,
        "seed": args.seed,
        "dpi": args.dpi,
    }

    aug_args = {
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "translate": args.translate,
        "scale": args.scale,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
    }
    model_args.update(aug_args)

    if args.resume:
        model_args["resume"] = args.resume
        model_args["model"] = args.model
        model = YOLO(args.model)
    elif not args.resume and args.model:
        if args.ch:
            # TODO: find better solution
            # if pretrained model have different number of channels
                # 1. load pretrained model
                # 2. crete new model from scratch with the required number of channels
                # 3. get input conv layer from new model
                # 4. copy pretrained weights to conv layer
                # 5. set conv layer as input layer of pretrained model
            model = YOLO(args.model)
            conv = model.model.model[0].conv

            model_tmp = YOLO(f"{args.model_name}.yaml", ch=args.ch)
            conv_tmp = model_tmp.model.model[0].conv

            with torch.no_grad():
                conv_tmp.weight[:,:3,:,:].copy_(conv.weight)
                if args.ch == 6:
                    conv_tmp.weight[:,3:,:,:].copy_(conv.weight)
            model.model.model[0].conv = conv_tmp
            model.model.yaml["ch"] = args.ch
        else:
            model = YOLO(args.model)
    else:
        if args.ch:
            model = YOLO(f"{args.model_name}.yaml", ch=args.ch)
        else:
            model = YOLO(f"{args.model_name}.yaml")

    model.train(
        **model_args
    )