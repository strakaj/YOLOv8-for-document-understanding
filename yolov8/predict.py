import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from docile.dataset import Field, store_predictions
from docile.dataset import KILE_FIELDTYPES, LIR_FIELDTYPES, BBox, Dataset
from docile.evaluation.evaluate import evaluate_dataset
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.yolo.data.chargrid import create_encodings_one_digit, create_encodings_three_digit_1, \
    create_encodings_three_digit_0, get_char_grid, characters

LIR_FIELDTYPES = LIR_FIELDTYPES + ["line_item"]


def vertical_intersection(box_1, box_2):
    _, t0, _, b0 = box_1
    _, t1, _, b1 = box_2
    intersection = min(float(b0), float(b1)) - max(float(t0), float(t1))

    return intersection


def horizontal_intersection(box_1, box_2):
    l0, _, r0, _ = box_1
    l1, _, r1, _ = box_2
    intersection = min(float(r0), float(r1)) - max(float(l0), float(l1))
    return intersection


def vertical_nms(predictions, box_height_overlap=0.5):
    # create matrix with overlapping values
    overlap_mat = np.zeros([len(predictions), len(predictions)])
    score_mat = np.zeros(overlap_mat.shape)
    for i, p0 in enumerate(predictions):
        for j, p1 in enumerate(predictions):
            vi = vertical_intersection(p0[0], p1[0])
            if vi > 0 and j <= i:
                overlap_mat[i, j] = vi
    # get only true overlaps -> at least overlap > half of the height of the box
    for i in range(len(predictions)):
        col = overlap_mat[:, i]
        box_h = col[i]
        for j, c in enumerate(col):
            if c < box_h * box_height_overlap:
                col[j] = 0
        overlap_mat[:, i] = col
    # get scores of overlapping boxes
    for i in range(len(predictions)):
        col = overlap_mat[:, i]
        for j, c in enumerate(col):
            if c > 0:
                score = predictions[j][2]
                col[j] = score
            score_mat[:, i] = col
    # if multiple overlapping boxes, keep only one with max score
    to_remove = []
    for i in range(len(predictions)):
        col = overlap_mat[:, i]
        candidates = np.where(col > 0)[0]
        if len(candidates) > 1:
            amax = np.argmax(col)
            to_remove.extend(list(set(candidates) - set([amax])))
    to_remove = list(set(to_remove))

    for ri in np.sort(to_remove)[::-1]:
        predictions.pop(ri)

    return predictions


def group_fields_vertical(fields, line_item_fields, box_height_overlap=0.5):
    if len(line_item_fields) == 0 or len(fields) == 0:
        return {}

    grouped_predictions = {}
    remaining_fields_indexes = list(np.arange(len(fields)))
    vertical_intersection_mat = np.zeros([len(line_item_fields), len(fields)])
    for i, li in enumerate(line_item_fields):
        li_box = li[0]
        li_box_h = li_box[3] - li_box[1]
        grouped_predictions[i] = []

        for j, f in enumerate(fields):
            f_box = f[0]
            inter = vertical_intersection(li_box, f_box)
            if inter > li_box_h * box_height_overlap:
                vertical_intersection_mat[i, j] = inter

    for j, f in enumerate(fields):
        column = vertical_intersection_mat[:, j]
        max_value = np.max(column)
        argmx = np.argmax(column)
        if max_value > 0:
            grouped_predictions[argmx].append(f)
            remaining_fields_indexes.remove(j)
    remaining_fields = [fields[idx] for idx in remaining_fields_indexes]
    return grouped_predictions


def group_fields_full(fields, line_item_fields, box_height_overlap=0.1):
    if len(line_item_fields) == 0 or len(fields) == 0:
        return {}
    grouped_predictions = {}
    remaining_fields_indexes = list(np.arange(len(fields)))
    intersection_mat = np.zeros([len(line_item_fields), len(fields)])
    for i, li in enumerate(line_item_fields):
        li_box = li[0]
        li_box_h = li_box[3] - li_box[1]
        grouped_predictions[i] = []

        li_box = BBox(*li_box)
        for j, f in enumerate(fields):
            f_box = f[0]
            f_box = BBox(*f_box)
            inter = li_box.intersection(f_box).area / f_box.area  # vertical_intersection(li_box, f_box)
            if inter > box_height_overlap:
                intersection_mat[i, j] = inter

    for j, f in enumerate(fields):
        column = intersection_mat[:, j]
        max_value = np.max(column)
        argmx = np.argmax(column)
        if max_value > 0:
            grouped_predictions[argmx].append(f)
            remaining_fields_indexes.remove(j)
    remaining_fields = [fields[idx] for idx in remaining_fields_indexes]
    return grouped_predictions


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--run_path', default="/home/strakajk/MLProjects/DocILE-2023/official_code2/YOLOv8-for-document-understanding/yolov8/docile_yolov8/dbg12", type=str)
    parser.add_argument('--dataset_path', default="/media/strakajk/Data/Datasets/DocILE/docile", type=str)
    parser.add_argument('--split', default="val", type=str)

    parser.add_argument('--grouping_method', default="vertical", type=str, help="Grouping method of line items. "
                                                                                "Vertical (based only on y axes) or "
                                                                                "full (based on overlap with line "
                                                                                "item field).")
    parser.add_argument('--grouping_overlap', default=0.2, type=float, help="Min overlap between filed and line item "
                                                                            "field, when vertical grouping method "
                                                                            "chosen.")
    parser.add_argument('--grouping_area', default=0.1, type=float, help="Min overlap area between filed and line item "
                                                                         "field, when full grouping method "
                                                                         "chosen.")
    parser.add_argument('--ch', default=6, type=int, help="Number of input channels of the model.")
    parser.add_argument('--line_item_class_id', default=55, type=int, help="Id of line item class in "
                                                                           "/ultralytics/datasets/docile.yaml")
    parser.add_argument('--box_height_overlap', default=0.5, type=float, help="If two line item objects overlap, one "
                                                                              "with lower score is discarded. This "
                                                                              "value controls how much of overlap is "
                                                                              "necessery to consider objects "
                                                                              "overlapping.")

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    # prepare all parameters and folders for output
    weights_path = os.path.join(args.run_path, "weights", "last.pt")
    args_path = os.path.join(args.run_path, "args.yaml")
    with open(args_path, "r") as f:
        model_args = yaml.safe_load(f)

    dpi = model_args["dpi"] if "dpi" in model_args else 72
    imgsz = model_args["imgsz"] if "imgsz" in model_args else 640
    char_grid_encoder = model_args["char_grid_encoder"] if "imgsz" in model_args else ""
    only_grid = model_args["only_grid"] if "only_grid" in model_args else False

    predictin_path = os.path.dirname(os.path.dirname(weights_path))
    predictin_path = os.path.join(predictin_path, f"{args.split}")
    os.makedirs(predictin_path, exist_ok=True)

    # prepare dataset and model
    dataset = Dataset(args.split, args.dataset_path)
    model = YOLO(weights_path, ch=args.ch)

    kile_fields = {}
    lir_fields = {}
    for doc in tqdm(dataset, desc="Run prediction"):
        kile_fields[doc.docid] = []
        lir_fields[doc.docid] = []
        pages = doc.annotation.page_count
        for p in range(pages):
            image_size = doc.page_image_size(page=p, dpi=dpi)
            img = doc.page_image(page=p, image_size=image_size)
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # prepare chargrid
            if char_grid_encoder:
                if char_grid_encoder == "one_digit":
                    encodings = create_encodings_one_digit(characters)
                elif char_grid_encoder == "three_digit_0":
                    encodings = create_encodings_three_digit_0(characters)
                elif char_grid_encoder == "three_digit_1":
                    encodings = create_encodings_three_digit_1(characters)

                char_grid = get_char_grid(doc.ocr.get_all_words(page=p, snapped=True), img.shape[:2], encodings)
                char_grid = (char_grid * 255).astype(np.uint8)
                if only_grid:
                    img = char_grid
                else:
                    img = np.concatenate((img, char_grid), axis=2)

            # predict
            out = model.predict(img, verbose=False, imgsz=imgsz)

            # gather predictions
            boxes = out[0].boxes.xyxyn.cpu().numpy()
            iw, ih = image_size
            boxesa = [np.array(b) * np.array([iw, ih, iw, ih]) for b in boxes]
            classes = out[0].boxes.cls.cpu().numpy().astype(int)
            confidences = out[0].boxes.conf.cpu().numpy()
            class_names = out[0].names

            predictions = list(zip(boxes, classes, confidences))

            # divided predicted boxes into kile and lir classes
            kile_predictions = []
            lir_predictions = []
            for bb, cls, conf in predictions:
                bb = bb.astype(float)
                conf = conf.astype(float)
                if class_names[cls] in KILE_FIELDTYPES:
                    kile_predictions.append([bb, cls, conf])
                elif class_names[cls] in LIR_FIELDTYPES:
                    lir_predictions.append([bb, cls, conf])
                else:
                    print(f"Class {class_names[cls]} not in kile or lir field types.")

            # group lir fields based on predicted line_items
            line_items_predictions = []
            lir_fields_predictions = []
            for lip in lir_predictions:
                if lip[1] == args.line_item_class_id:
                    line_items_predictions.append(lip)
                else:
                    lir_fields_predictions.append(lip)
            line_items_predictions = vertical_nms(line_items_predictions, box_height_overlap=args.box_height_overlap)

            if args.grouping_method == "vertical":
                grouped_predictions = group_fields_vertical(lir_fields_predictions, line_items_predictions,
                                                       box_height_overlap=args.grouping_overlap)
            elif args.grouping_method == "full":
                grouped_predictions = group_fields_full(lir_fields_predictions, line_items_predictions,
                                                            args.grouping_area)

            # crete fields for evaluation
            for bb, cls, conf in kile_predictions:
                new_field = Field(
                    fieldtype=class_names[cls],
                    bbox=BBox(*bb),
                    page=p,
                    score=conf,
                )
                kile_fields[doc.docid].append(new_field)

            for gid in grouped_predictions:
                for bb, cls, conf in grouped_predictions[gid]:
                    new_field = Field(
                        fieldtype=class_names[cls],
                        bbox=BBox(*bb),
                        page=p,
                        score=conf,
                        line_item_id=gid
                    )
                    lir_fields[doc.docid].append(new_field)

    # store predictions
    store_predictions(Path(os.path.join(predictin_path, "kile_predictions.json")), kile_fields)
    store_predictions(Path(os.path.join(predictin_path, "lir_predictions.json")), lir_fields)

    if args.split != "test":

        try:
            # KILE
            evaluation_result_KILE = evaluate_dataset(dataset, kile_fields, {})
            kile_metrics = evaluation_result_KILE.get_metrics("kile")
            print(evaluation_result_KILE.print_report())

            # LIR
            evaluation_result_LIR = evaluate_dataset(dataset, {}, lir_fields)
            lir_metrics = evaluation_result_LIR.get_metrics("lir")
            print(evaluation_result_LIR.print_report())

            summary = f"{os.path.basename(os.path.dirname(os.path.dirname(weights_path)))}\n   " \
                       f"kile: {evaluation_result_KILE.get_primary_metric('kile')} AP\n   " \
                       f"lir: {evaluation_result_LIR.get_primary_metric('lir')} F1\n"
            print(summary)

        except:
            print("Evaluation failed:", os.path.basename(os.path.dirname(os.path.dirname(weights_path))))



