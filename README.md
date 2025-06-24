# Histopathologic Cancer Detection - CNN

This project uses a Convolutional Neural Network (CNN) to detect cancer in histopathologic images, based on the [Kaggle Histopathologic Cancer Detection competition](https://www.kaggle.com/c/histopathologic-cancer-detection).

## Project Overview

- **Dataset:** 220,025 RGB images (96x96 pixels), each labeled as "normal" or "cancer".
- **Goal:** Binary classification to identify cancerous tissue in images.
- **Frameworks:** TensorFlow, Keras, Keras Tuner, scikit-learn, pandas, matplotlib, seaborn.

## Main Steps

1. **Data Preparation**
   - Loads labels from `data/train_labels.csv`.
   - Splits data into training and validation sets (80/20 split, stratified).
   - Organizes images into directory structure compatible with `keras.utils.image_dataset_from_directory`.
   - Optionally converts `.tif` images to `.jpg`.

2. **Exploratory Data Analysis (EDA)**
   - Visualizes label distribution and sample images.

3. **Dataset Creation**
   - Uses `image_dataset_from_directory` to create TensorFlow datasets for training, validation, and testing.
   - Prefetches data for performance.

4. **Model Architecture & Hyperparameter Tuning**
   - Defines a CNN with 3 Conv2D + MaxPooling2D blocks, followed by Flatten, Dropout, and Dense layers.
   - Uses Keras Tuner (Bayesian Optimization) to search for optimal dropout, dense units, and learning rate.

5. **Training**
   - Trains the best model on the full training set with early stopping and model checkpointing.

6. **Evaluation & Submission**
   - Loads the best model.
   - Predicts on the test set.
   - Applies a threshold of 0.5 to generate binary predictions.
   - Saves results to `submission.csv` for Kaggle submission.

## Usage

1. **Install dependencies:**
   - Python 3.8+
   - TensorFlow, Keras, keras-tuner, scikit-learn, pandas, matplotlib, seaborn, tqdm, Pillow

2. **Prepare data:**
   - Download and extract the dataset into the `data/` directory as described in the notebook.

3. **Run the notebook:**
   - Open [`cancer_detection.ipynb`](cancer_detection.ipynb) in Jupyter or VS Code.
   - Execute cells sequentially.

4. **Submit predictions:**
   - The notebook generates `submission.csv` for Kaggle submission.

## Directory Structure

```
.
├── cancer_detection.ipynb
├── best_model.keras
├── submission.csv
├── data/
│   ├── histopathologic-cancer-detection.zip
│   ├── sample_submission.csv
│   ├── train_labels.csv
│   ├── train/
│   ├── val/
│   └── test/
├── kt_dir/
├── tmp_model/
└── README.md
```

## Notes

- The notebook includes code for data augmentation and conversion, but these steps are optional and controlled by the `move_labels` and `convert_to_jpg` flags.
- Hyperparameter tuning is performed on a small subset for speed.
- The final model is trained on the full dataset.

---

For more details, see the code and comments in [`cancer_detection.ipynb`](cancer_detection.ipynb).
