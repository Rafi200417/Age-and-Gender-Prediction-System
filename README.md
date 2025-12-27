# Age and Gender Identification Project

## Overview
This project implements age and gender prediction using computer vision and deep learning. It includes pre-trained model inference (static and real-time), custom model training, and prediction, controlled via a main script.

## Folder Structure
```
AgeGenderIdentification/
├── data/
│   ├── UTKFace/                # Dataset for custom training
│   └── test_images/            # Sample images for testing
├── models/
│   ├── caffe/                  # Pre-trained Caffe models
│   │   ├── deploy.prototxt
│   │   ├── res10_300x300_ssd_iter_140000.caffemodel
│   │   ├── age_deploy.prototxt
│   │   ├── age_net.caffemodel
│   │   ├── gender_deploy.prototxt
│   │   └── gender_net.caffemodel
│   └── custom/                 # Custom trained model
│       └── age_gender_model.h5
├── src/
│   ├── pretrained_static.py    # Static image prediction
│   ├── pretrained_realtime.py  # Real-time video prediction
│   ├── custom_train.py         # Custom model training
│   ├── custom_predict.py       # Custom model prediction
│   └── main.py                 # Main entry point
├── README.md
└── requirements.txt
```

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the required models and place them in the `models/caffe/` directory:
   - Face Detection: `res10_300x300_ssd_iter_140000.caffemodel`
   - Age Prediction: `age_net.caffemodel`
   - Gender Prediction: `gender_net.caffemodel`
   - And their corresponding `.prototxt` files

3. For custom training:
   - Download the UTKFace dataset
   - Extract it to `data/UTKFace/`

4. Add test images to `data/test_images/`

## Usage
Run the main script with one of these modes:

1. Static Image Prediction:
```bash
python src/main.py static [image_path]
```

2. Real-Time Video:
```bash
python src/main.py realtime
```

3. Train Custom Model:
```bash
python src/main.py train
```

4. Custom Model Prediction:
```bash
python src/main.py custom [image_path]
```

If no image path is provided, it will use the default image at `data/test_images/person1.jpg`.

## Model Sources
- Pre-trained Caffe models: [OpenCV Face Detection](https://github.com/opencv/opencv/tree/master/samples/dnn)
- UTKFace Dataset: [UTKFace](https://susanqq.github.io/UTKFace/)
