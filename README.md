# Oriented Object Detection for Aerial Imagery

![Python](https://img.shields.io/badge/python-3.8-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

Anchor-free, single-stage **oriented object detector** for aerial and drone imagery, using box boundary-aware vectors (BBAVectors) and center-keypoint detection. Adapted from the DOTA-dataset architecture to a custom drone dataset, so detections include object **orientation** — not just axis-aligned boxes.

## Key Results

| Setup | Result |
| --- | --- |
| Detection accuracy (orientation classification + external OBB size params) | 72.32% |
| Detection accuracy (larger training batch) | **75.36%** |
| mAP on custom drone dataset (after class rebalancing) | **84.71%** |
| Inference speed | 11 FPS |

Per-class breakdown (7 classes): two-wheeler 97.18%, cars 94.69%, heavy vehicles 96.07%, pedestrians 35.29% (known weak point — small objects in low resolution).

## Method

1. **Dataset adaptation:** converted the custom AU drone dataset into DOTA format; filtered and re-split from 10 classes down to 7, ~6,000 frames with 15–20 objects each (~4,000 for training).
2. **Class imbalance handling:** applied class weighting to counter severe skew (e.g. far more two-wheelers than cars), which drove the mAP gain to 84.71%.
3. **Model:** ResNet101 backbone with BBAVectors + center-keypoint heads for oriented bounding boxes, trained on GPU with CUDA.
4. **Tracking:** experiments logged with Weights & Biases.

## Repository Structure

```
.
├── Main Code/                 # BBAVectors-based oriented detector implementation
│   ├── main.py                # training / testing / evaluation entry point
│   ├── train.py               # training script
│   ├── eval.py                # evaluation script (mAP, per-class breakdown)
│   ├── test.py                # inference script
│   ├── models/                # model definitions (ResNet101 backbone, BBAVectors heads)
│   ├── decoder.py             # oriented bounding box decoder
│   ├── loss.py                # loss functions
│   ├── nms.py                 # non-maximum suppression
│   ├── func_utils.py          # utilities
│   ├── draw_loss.py           # training loss visualization
│   ├── detasets/              # dataset loaders (DOTA format)
│   ├── examplesplit/          # image splitting utilities
│   ├── merge_dota/            # DOTA result merging for evaluation
│   ├── result_dota/           # evaluation outputs
│   ├── weights_dota/          # trained weights
│   └── performance_metrics.txt
├── Final Report.pdf           # full project report
└── README.md
```

## Setup

```
git clone https://github.com/Dp3xD/oriented-object-detection-aerial-images.git
cd oriented-object-detection-aerial-images
pip install -r requirements.txt
```

Requires Python 3.8+, PyTorch with CUDA, OpenCV, and torchvision.

## Usage

```
python train.py --config configs/dota_drone.yaml
python eval.py --checkpoint checkpoints/best.pth
```

## Datasets

Custom drone dataset (~6,000 annotated frames) derived from the AU Drone Dataset, converted to DOTA format. Dataset files are not committed; place them under `data/` following the structure above.

## Limitations

- Pedestrian detection remains weak (35.29%) — small objects at altitude are the hardest case
- ~11 FPS inference; not yet real-time on edge hardware
- Trained and evaluated on a single custom dataset; generalization to other aerial datasets untested

## Team

Hrishikesh Rana, Divya Patel, Aditya Chaudhari, Preksha Morbia — Ahmedabad University

## License

MIT
