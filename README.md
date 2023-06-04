# YOLOv8 for document understanding


## Insallation


Environment should have with torch>=1.7 ([yolov8](yolov8/README.md)) for example:
```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

Install YOLOv8 requirements:
```bash
pip install -r yolov8/requirements.txt
```

Install DocILE library:
```bash
pip install docile-benchmark
```

## Training and prediction
Before running the code it is necessary to edit the dataset config file: `yolov8/ultralytics/datasets/docile.yaml`.
The file contains information about the dataset:
 - path: `<path/to/docile/dataset>`
 - cache_location: `<folder/for/yolo/to/save/cache/files>`

YOLOv8 config file with parameters is located at: `yolov8/ultralytics/yolo/cfg/default.yaml`

### Train
```bash
python yolov8/train.py \
    --model_name yolov8x \
    --data_path ylov8/ultralytics/datasets/docile.yaml \
    --epochs 30 \
    --lr0 0.001 \
    --batch 8 \
    --imgsz 1280 \
    --workers 8 \
    --optimizer AdamW \
    --model yolov8x.pt \
    --char_grid_encoder three_digit_0 \
    --ch 6 \
    --seed 0 \
    --hsv_h 0.0 \
    --hsv_s 0.0 \
    --hsv_v 0.0 \
    --scale 0.0 \
    --fliplr 0.0 \
    --mosaic 0.0 
```
The above code will reproduce KILE results, to reproduce LIR results, `epochs` should be changed to `50` and `seed` to `1`

### Predict
```bash
python yolov8/predict.py \
    --run_path <path/to/yolov8/output/folder> \
    --dataset_path <path/to/docile/dataset>
```
