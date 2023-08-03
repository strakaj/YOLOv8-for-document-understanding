import argparse
import json
import os
from glob import glob

import cv2
import easyocr
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.yolo.data.chargrid import create_encodings_one_digit, create_encodings_three_digit_1, \
    create_encodings_three_digit_0, characters, get_char_grid_easy

LIR_FIELDTYPES = ['line_item_amount_gross', 'line_item_amount_net', 'line_item_code', 'line_item_currency',
                  'line_item_date', 'line_item_description', 'line_item_discount_amount', 'line_item_discount_rate',
                  'line_item_hts_number', 'line_item_order_id', 'line_item_person_name', 'line_item_position',
                  'line_item_quantity', 'line_item_tax', 'line_item_tax_rate', 'line_item_unit_price_gross',
                  'line_item_unit_price_net', 'line_item_units_of_measure', 'line_item_weight', 'line_item']
KILE_FIELDTYPES = ['account_num', 'amount_due', 'amount_paid', 'amount_total_gross', 'amount_total_net',
                   'amount_total_tax', 'bank_num', 'bic', 'currency_code_amount_due', 'customer_billing_address',
                   'customer_billing_name', 'customer_delivery_address', 'customer_delivery_name', 'customer_id',
                   'customer_order_id', 'customer_other_address', 'customer_other_name', 'customer_registration_id',
                   'customer_tax_id', 'date_due', 'date_issue', 'document_id', 'iban', 'order_id', 'payment_reference',
                   'payment_terms', 'tax_detail_gross', 'tax_detail_net', 'tax_detail_rate', 'tax_detail_tax',
                   'vendor_address', 'vendor_email', 'vendor_name', 'vendor_order_id', 'vendor_registration_id',
                   'vendor_tax_id']


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


def reshape_input_image(input_image, new_height = 1280):
    org_height = input_image.shape[0]
    new_width = int(np.round((input_image.shape[1] * new_height) / org_height))
    resized_image = cv2.resize(input_image, (new_width, new_height))
    return resized_image


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--checkpoint_path', default="", type=str)
    parser.add_argument('--output_path', default="", type=str)
    parser.add_argument('--data_path', default="", type=str)

    parser.add_argument('--grouping_overlap', default=0.2, type=float, help="Min overlap between filed and line item "
                                                                            "field, when vertical grouping method "
                                                                            "chosen.")

    parser.add_argument('--line_item_class_id', default=55, type=int, help="Id of line item class in "
                                                                           "/ultralytics/datasets/docile.yaml")
    parser.add_argument('--box_height_overlap', default=0.5, type=float, help="If two line item objects overlap, one "
                                                                              "with lower score is discarded. This "
                                                                              "value controls how much of overlap is "
                                                                              "necessery to consider objects "
                                                                              "overlapping.")

    parser.add_argument('--imgsz', default=1280, type=int, help="")
    parser.add_argument('--ch', default=6, type=int, help="Number of input channels of the model.")
    parser.add_argument('--char_grid_encoder', default="three_digit_0", type=str)
    parser.add_argument('--only_grid', action='store_true')
    parser.add_argument('--ocr_threshold', default=0.8, type=float, help="")

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    # prepare dataset and model
    image_paths = glob(os.path.join(args.data_path, "*"))
    model = YOLO(args.checkpoint_path, ch=args.ch)
    reader = easyocr.Reader(['cs', 'en'])

    ann_data = []
    image_data = []
    ann_idx = 0
    for img_idx, path in enumerate(tqdm(image_paths, desc="Run prediction")):

        img = cv2.imread(path)
        ih, iw = img.shape[:2]
        image_data.append({
            "id": img_idx,
            "width": iw,
            "height": ih,
            "file_name": os.path.basename(path)
        })

        # prepare chargrid
        if args.char_grid_encoder:
            if args.char_grid_encoder == "one_digit":
                encodings = create_encodings_one_digit(characters)
            elif args.char_grid_encoder == "three_digit_0":
                encodings = create_encodings_three_digit_0(characters)
            elif args.char_grid_encoder == "three_digit_1":
                encodings = create_encodings_three_digit_1(characters)

            ocr_img = img.copy()
            #ocr_img = reshape_input_image(ocr_img)
            ocr_result = reader.readtext(ocr_img)
            ocr_result = [[ np.round([*o[0][0], *o[0][2]]).astype(int), o[1], o[2]] for o in ocr_result if o[2] > args.ocr_threshold]

            char_grid = get_char_grid_easy(ocr_result, img.shape[:2], encodings)
            char_grid = (char_grid * 255).astype(np.uint8)
            if args.only_grid:
                img = char_grid
            else:
                img = np.concatenate((img, char_grid), axis=2)

        # predict
        out = model.predict(img, verbose=False, imgsz=args.imgsz)

        # gather predictions
        boxes = np.round(out[0].boxes.xyxy.cpu().numpy()).astype(int)
        # iw, ih = img.shape[:2]
        # boxesa = [np.array(b) * np.array([iw, ih, iw, ih]) for b in boxes]
        classes = out[0].boxes.cls.cpu().numpy().astype(int)
        confidences = out[0].boxes.conf.cpu().numpy()
        class_names = out[0].names

        predictions = list(zip(boxes, classes, confidences))

        # divided predicted boxes into kile and lir classes
        kile_predictions = []
        lir_predictions = []
        for bb, cls, conf in predictions:
            bb = np.round(bb).astype(float)
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


        grouped_predictions = group_fields_vertical(lir_fields_predictions, line_items_predictions,
                                                    box_height_overlap=args.grouping_overlap)


        # save in coco format
        for bb, cls, conf in kile_predictions:
            x0, y0, x1, y1 = np.round(bb)
            ann_data.append({
                "id": ann_idx,
                "image_id": img_idx,
                "category_id": int(cls),
                "bbox": [int(x0), int(y0), int(x1-x0), int(y1-y0)],
                "score": float(conf)
            })
            ann_idx += 1

        for gid in grouped_predictions:
            for bb, cls, conf in grouped_predictions[gid]:
                x0, y0, x1, y1 = np.round(bb).astype(np.int32)
                ann_data.append({
                    "id": ann_idx,
                    "image_id": img_idx,
                    "category_id": int(cls),
                    "bbox": [int(x0), int(y0), int(x1-x0), int(y1-y0)],
                    "score": float(conf),
                    "line_item_id": gid
                })
                ann_idx += 1

    class_names = {n:c for c, n in class_names.items()}
    categories = [
        *[{"id": class_names[name], "name": name, "supercategory": "kile"} for name in KILE_FIELDTYPES],
        *[{"id": class_names[name], "name": name, "supercategory": "lir"} for name in LIR_FIELDTYPES]
    ]
    annotations = {
        "images": image_data,
        "annotations": ann_data,
        "categories": categories
    }


    with open(os.path.join(args.output_path, "results.json"), "w") as f:
        json.dump(annotations, f, indent=4)
