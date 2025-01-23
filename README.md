# Breast Cancer Metastasis Detection and Ratio Prediction using Deep Learning

## Overview

This repository contains the implementation for Assignment 3 of the AIN2001 course, which consists of two parts, each modeled after a Kaggle competition:

1. **Part 1: Classification Task**
2. **Part 2: Regression Task**

Both parts utilize the same dataset and aim to solve different machine learning problems.

---

## Kaggle Competitions

- **Part 1: Classification**
  - **Competition URL:** [Part 1: AIN2001 Fall22 A3P1](https://www.kaggle.com/competitions/bau-ain2001-fall22-a3p1)
  - **Metric:** Area Under Receiver Operating Characteristic Curve (AUROC)
  - **Result:** AUROC Score = 0.87625 (Rank: 5/17)

- **Part 2: Regression**
  - **Competition URL:** [Part 2: AIN2001 Fall22 A3P2](https://www.kaggle.com/competitions/bau-ain2001-fall22-a3p2)
  - **Metric:** Mean Absolute Error (MAE)
  - **Result:** MAE Score = 0.21155 (Rank: 6/15)

---

## Dataset

The dataset is organized into the following directories:

- `data/`
  - `train.csv`: Training data with labels for classification and regression tasks.
  - `test.csv`: Testing data without labels.
  - `sample_submission.csv`: Example submission file.
  - `images/`: Directory containing all image files used in the tasks.

---

## Assignment Details

### Part 1: Classification Task

**Aim:**  
Design deep learning models to detect breast cancer metastases in underarm lymph nodes.

**Objective:**  
Develop a convolutional neural network (CNN) to classify images into two categories: "normal" (0) and "cancer" (1). The performance is evaluated using the AUROC metric.

**Dataset:**  
Consisting of images cropped from histopathology slides of lymph node sections. The images have labels of "normal" (0) and "cancer" (1).

**Task:**  
Given an image, predict whether it is "normal" or "cancer".

**Methods Used:**

- **Data Preprocessing:**
  - Image resizing and normalization using `torchvision.transforms`.
  - Splitting data into training and testing sets.

- **Model Architecture:**
  - A custom CNN with multiple convolutional layers, ReLU activations, pooling layers, dropout for regularization, and a sigmoid activation for binary classification.

- **Training:**
  - Binary Cross-Entropy Loss (`nn.BCELoss`).
  - Optimizer: Adam (`torch.optim.Adam`).
  - Learning rate scheduler: StepLR.

- **Evaluation:**
  - Calculating AUROC using `sklearn.metrics`.
  - Saving the best model based on the lowest test loss and highest AUROC.

**Results:**  
Achieved an AUROC score of **0.87625**, ranking **5th out of 17** participants.

### Part 2: Regression Task

**Aim:**  
Design deep learning models to predict the breast cancer metastases ratio in digital histopathology images of lymph node sections.

**Objective:**  
Build a regression model to predict the metastasis ratio in images. The performance is measured using Mean Absolute Error (MAE).

**Dataset:**  
Consisting of images cropped from histopathology slides of lymph node sections. The percentages of metastases regions in images are provided.

**Task:**  
Given an image, predict the metastases ratio in the image.

**Methods Used:**

- **Data Preprocessing:**
  - Similar image preprocessing as in the classification task.
  - Handling of regression labels (`metastasis_ratio`).

- **Model Architecture:**
  - A modified CNN tailored for regression tasks, utilizing Mean Squared Error Loss (`nn.MSELoss`).

- **Training:**
  - Optimizer: Adam (`torch.optim.Adam`).
  - Learning rate scheduler: StepLR.

- **Evaluation:**
  - Calculating MAE and an additional L1 Loss for robustness.
  - Saving the best model based on the lowest test loss.

**Results:**  
Achieved an MAE score of **0.21155**, ranking **6th out of 15** participants.

---

## Installation

1. **Download the Dataset:**

   - Visit the competition page: [AIN2001 Fall22 A3P2](https://www.kaggle.com/competitions/bau-ain2001-fall22-a3p2)
   - Accept the competition rules
   - Download the dataset using one of these methods:
     ```bash
     # Option 1: Using Kaggle API
     pip install kaggle
     kaggle competitions download -c bau-ain2001-fall22-a3p2
     unzip bau-ain2001-fall22-a3p2.zip -d data/
     
     # Option 2: Manual download
     # Click the "Data" tab on the competition page and download the files manually
     ```

2. **Set Up the Conda Environment:**

   Ensure you have [Conda](https://docs.conda.io/en/latest/) installed.

   ```bash
   conda env create -f environment.yml
   conda activate ain2001_env
   ```

3. **Data Preparation:**

   - Place all image files in the `data/images/` directory
   - Ensure `train.csv`, `train2.csv` and `test.csv` are in the `data/` directory

---

## Usage

### Classification Task

1. **Run the Classification Script:**

   ```bash
   python src/classification.py 
   ```

2. **Outputs:**

   - Predictions are saved in `predictions/part1/`.
   - Models are saved in `saved_models/part1/`.
   - Training statistics are recorded in `stats/part1/`.

### Regression Task


1. **Run the Regression Script:**

   ```bash
   python src/regression.py 
   ```

3. **Outputs:**

   - Predictions are saved in `predictions/part2/`.
   - Models are saved in `saved_models/part2/`.
   - Training statistics are recorded in `stats/part2/`.

---

## Project Structure

```
ain2001-assignment3/
├── src/
│   ├── classification.py
│   └── regression.py
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   └── images/
├── predictions/
│   ├── part1/
│   └── part2/
├── saved_models/
│   ├── part1/
│   └── part2/
├── stats/
│   ├── part1/
│   └── part2/
├── environment.yml
├── README.md
└── LICENSE
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


