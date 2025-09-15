# alzheimers-multiclass-mobilenetv2-lime

**Multiclass Alzheimer MRI classification (NonDemented / VeryMildDemented / MildDemented / ModerateDemented)**

This repository contains a reproducible Jupyter/Colab notebook that trains a MobileNetV2-based convolutional neural network to classify Alzheimer MRI images into four classes and uses LIME (Local Interpretable Model-agnostic Explanations) to explain model predictions.

---

## Key features

* Data loading + automatic split into Train / Val / Test (70 / 15 / 15)
* Data augmentation with `ImageDataGenerator`
* Transfer learning using `MobileNetV2` (fine-tune last \~30 non-BatchNorm layers)
* Class imbalance handled with `class_weight='balanced'`
* Model training with callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
* Detailed evaluation: classification report + confusion matrix
* Per-image explainability using `LIME` (LimeImageExplainer)

---

## Dataset (as used in the notebook)

The notebook assumes you have the original dataset arranged as class folders (example path in notebook):

```
/content/drive/MyDrive/alzheimer_dataset/OriginalDataset
  ├─ NonDemented/
  ├─ VeryMildDemented/
  ├─ MildDemented/
  └─ ModerateDemented/
```

**Detected class distribution in the notebook run (example):**

* `NonDemented`    : 3210 images (≈50.00%)
* `VeryMildDemented`: 2250 images (≈35.05%)
* `MildDemented`   : 896 images (≈13.96%)
* `ModerateDemented`: 64 images (≈1.00%)

*Total images: 6420*

> Note: your numbers may vary depending on your dataset source. The notebook will reproduce the dataset split and recompute class weights automatically.

---

## Repo structure (recommended)

```
alzheimers-multiclass-mobilenetv2-lime/
├─ README.md
├─ requirements.txt
├─ Alzheimer(Multi_Class+LIME).ipynb   # main notebook (this project)
├─ scripts/                             # optional: helper scripts for preprocessing, inference
├─ models/                              # saved model(s) (e.g. alzheimer_multiclass_best_model.h5)
└─ sample_explanations/                 # generated LIME visualizations
```

---

## Requirements

Create a virtual environment (or use Colab) and install dependencies. Example `requirements.txt` contents (not exhaustive):

```
tensorflow>=2.8
tensorflow.keras
numpy
pandas
scikit-learn
matplotlib
seaborn
opencv-python
lime
jupyter
nbformat
```

Install quickly with:

```bash
pip install -r requirements.txt
# or for Colab inside a cell
!pip install lime
```

---

## Quick start (Colab / Jupyter)

1. Open `Alzheimer(Multi_Class+LIME).ipynb` in Colab or Jupyter.
2. Mount Google Drive (if using Colab) so the notebook can read and write model checkpoints and data.

```python
from google.colab import drive
drive.mount('/content/drive')

# set dataset path (edit if your path is different)
dataset_path = '/content/drive/MyDrive/alzheimer_dataset/OriginalDataset'
```

3. Run cells in order. The notebook will:

   * Verify dataset folders
   * Create a working split at `/content/alz_multiclass_split` (train/val/test)
   * Build generators (`IMG_SIZE=224`, `BATCH_SIZE=32`)
   * Build and compile the MobileNetV2-based model
   * Train with `epochs=30` (with callbacks)
   * Save best model: `/content/drive/MyDrive/alzheimer_multiclass_best_model.h5`

---

## Training notes

* The model uses `MobileNetV2` as the base and adds a `GlobalAveragePooling2D` layer + Dense(128) + Dropout before the softmax output for `NUM_CLASSES=4`.
* The notebook freezes all layers initially then unfreezes the last \~30 non-BatchNorm layers for fine-tuning.
* Optimizer: `Adam(learning_rate=1e-4)` and loss `categorical_crossentropy`.
* Class imbalance is addressed with `sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', ...)` and passed to `model.fit(..., class_weight=class_weights)`.

Recommended callback settings are present (EarlyStopping with `restore_best_weights=True`, ReduceLROnPlateau, ModelCheckpoint monitored on `val_accuracy`).

---

## Evaluation & Inference

After training, the notebook:

* Runs `model.predict()` on `test_generator`
* Prints a classification report (`precision`, `recall`, `f1-score`) and plots a confusion matrix via `seaborn.heatmap`

To run inference on a single image (example from the notebook):

```python
from tensorflow.keras.preprocessing import image
import numpy as np
img_path = 'path/to/image.jpg'
pil_img = image.load_img(img_path, target_size=(224,224))
arr = image.img_to_array(pil_img)
arr = np.expand_dims(arr, axis=0)
arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
pred = model.predict(arr)
pred_class = pred.argmax(axis=1)[0]
print('Predicted:', class_labels[pred_class])
```

Saved model example path used in the notebook:

```
/content/drive/MyDrive/alzheimer_multiclass_best_model.h5
```

---

## LIME explainability

The notebook includes a `predict_fn(imgs)` wrapper and a `run_lime(pil_img, pred_class)` helper that:

* Uses `lime.lime_image.LimeImageExplainer()`
* Calls `explain_instance(np.array(pil_img), predict_fn, top_labels=4, num_samples=1000)`
* Generates a mask with `explanation.get_image_and_mask(...)` and visualizes boundaries with `skimage.segmentation.mark_boundaries`

Usage (from the notebook):

```python
from skimage.segmentation import mark_boundaries
lime_img = run_lime(pil_img, pred_class)
plt.imshow(lime_img)
plt.title(f"LIME\n{label}")
plt.show()
```

**Tip:** LIME can be slow because it queries the model many times (`num_samples=1000` by default). Reduce `num_samples` for faster but noisier explanations.

---

## Reproducibility tips

* Fix seeds (`numpy`, `tensorflow`) if deterministic runs are desired (note: GPU non-determinism may remain).
* Use the same `IMG_SIZE`, `BATCH_SIZE`, and preprocessing function (`tf.keras.applications.mobilenet_v2.preprocess_input`).
* If you have limited GPU memory, reduce `BATCH_SIZE` or freeze more base layers.

---

## Files to add to the repo

* `Alzheimer(Multi_Class+LIME).ipynb` (the notebook)
* `requirements.txt` (pinned versions if possible)
* `.gitignore` (ignore large model files & dataset, e.g. `models/*`, `__pycache__/`, `*.h5`)
* `LICENSE` (MIT recommended for code samples)

---

## License

This repository is released under the **MIT License**. Change as appropriate.

---

## Acknowledgements

* The notebook uses `MobileNetV2` and the `lime` library. This project was built as a proof-of-concept classification + explainability pipeline for an Alzheimer image dataset.

---
Author: Imran Mansha
