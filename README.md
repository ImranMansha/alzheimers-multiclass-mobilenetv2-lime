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

## Dataset

The dataset used in this project is available on **Google Drive**:
👉 [Download MRI Dataset](https://drive.google.com/drive/folders/1bMjyLQRYaugvy2LyqDcSuez2n_hkVwDu?usp=sharing)

The dataset should be arranged in class folders as shown:

```
OriginalDataset/
  ├─ NonDemented/
  ├─ VeryMildDemented/
  ├─ MildDemented/
  └─ ModerateDemented/
```

**Example class distribution in the notebook run:**

* `NonDemented`    : 3210 images (≈50.00%)
* `VeryMildDemented`: 2250 images (≈35.05%)
* `MildDemented`   : 896 images (≈13.96%)
* `ModerateDemented`: 64 images (≈1.00%)

*Total images: 6420*

### How to download automatically

You can fetch the dataset programmatically using `gdown` in Colab or Jupyter:

```bash
pip install gdown
```

```python
import gdown

# Folder link
dataset_url = 'https://drive.google.com/drive/folders/1bMjyLQRYaugvy2LyqDcSuez2n_hkVwDu?usp=sharing'

# Use gdown to download (requires the folder to be public)
!gdown --folder $dataset_url -O ./alzheimer_dataset
```

Then set dataset path in the notebook:

```python
dataset_path = './alzheimer_dataset/OriginalDataset'
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
gdown
```

Install quickly with:

```bash
pip install -r requirements.txt
# or for Colab inside a cell
!pip install lime gdown
```

---

## Quick start (Colab / Jupyter)

1. Open `Alzheimer(Multi_Class+LIME).ipynb` in Colab or Jupyter.
2. Download the dataset using the link above or via `gdown`.
3. Mount Google Drive (if using Colab) so the notebook can read/write model checkpoints and data.

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Run cells in order. The notebook will:

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
* Class imbalance is addressed with `sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', ...)`.

---

## Evaluation & Inference

After training, the notebook:

* Runs `model.predict()` on `test_generator`
* Prints a classification report (`precision`, `recall`, `f1-score`) and plots a confusion matrix

To run inference on a single image:

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

---

## LIME explainability

The notebook includes a `predict_fn(imgs)` wrapper and a `run_lime(pil_img, pred_class)` helper that:

* Uses `lime.lime_image.LimeImageExplainer()`
* Generates a mask and overlays it with `skimage.segmentation.mark_boundaries`

Usage:

```python
lime_img = run_lime(pil_img, pred_class)
plt.imshow(lime_img)
plt.title(f"LIME\n{label}")
plt.show()
```

---


## License

This repository is released under the **MIT License**. Change as appropriate.

---

## Acknowledgements

* The notebook uses `MobileNetV2` and the `lime` library.
* Dataset courtesy of the shared Google Drive link above.

---

**Author** : [Imran Mansha](https://www.linkedin.com/in/imranmansha/)

