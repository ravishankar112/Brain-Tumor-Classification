# Brain Tumor Classification

A deep learning-based application for classifying brain MRI images into four categories: **Glioma, Meningioma, No Tumor, and Pituitary**. The trained model is integrated with a Streamlit interface for simple image-based prediction and result visualization.

## Project Overview

This project was developed as an end-to-end machine learning application covering dataset preparation, transfer learning, model fine-tuning, evaluation, and deployment.

The classification model is based on **VGG19 with ImageNet pre-trained weights**. The final model was fine-tuned on a balanced brain MRI dataset and then integrated into a Streamlit web application.

## Key Features

- Four-class brain MRI image classification
- VGG19 transfer learning and fine-tuning
- Image preprocessing at 224 × 224 resolution
- Class-wise probability scores
- Streamlit-based web interface
- Prediction report generation
- Model and dataset information
- User feedback form
- Basic image quality validation

## Classification Categories

| Category | Model Output |
|---|---|
| Glioma | Tumor category |
| Meningioma | Tumor category |
| No Tumor | No target tumor category detected |
| Pituitary | Tumor category |

## Dataset

The dataset contains **7,200 MRI images** divided into training, validation, and testing sets.

| Split | Number of Images |
|---|---:|
| Training | 5,600 |
| Validation | 1,120 |
| Testing | 1,600 |
| **Total** | **7,200** |

The four classes are balanced in the original training and testing directories.

## Model

The final classifier uses VGG19 as a convolutional feature extractor with a custom classification head.

### Architecture

```text
Input MRI Image
      ↓
Resize to 224 × 224
      ↓
Pixel Normalization (/255.0)
      ↓
VGG19 Feature Extractor
      ↓
Global Average Pooling
      ↓
Dense Layer (256, ReLU)
      ↓
Dropout (0.5)
      ↓
Dense Layer (4, Softmax)
      ↓
Predicted Class
```

### Training Configuration

| Parameter | Configuration |
|---|---|
| Architecture | VGG19 |
| Pre-trained weights | ImageNet |
| Input size | 224 × 224 × 3 |
| Pooling | Global Average Pooling |
| Dense layer | 256 neurons |
| Dropout | 0.5 |
| Output classes | 4 |
| Output activation | Softmax |
| Optimizer | SGD |
| Learning rate | 0.0001 |
| Momentum | 0.9 |
| Nesterov | Enabled |
| Fine-tuning | Selected Block 5 convolution layers |

## Model Performance

The final model achieved **84.62% test accuracy** on 1,600 previously unseen test images.

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Glioma | 0.90 | 0.69 | 0.78 |
| Meningioma | 0.78 | 0.72 | 0.75 |
| No Tumor | 0.86 | 0.98 | 0.92 |
| Pituitary | 0.85 | 0.98 | 0.91 |
| **Overall Accuracy** | | | **84.62%** |

## Application Workflow

```text
Upload MRI Image
       ↓
Image Quality Check
       ↓
Image Preprocessing
       ↓
VGG19 Model
       ↓
Softmax Prediction
       ↓
Class Probabilities
       ↓
Prediction Report
```

## Streamlit Application

The trained model is deployed through a Streamlit interface.

The application contains four main sections:

### Dashboard
Provides a summary of the project, dataset, model, technologies, and workflow.

### MRI Classifier
Users can upload an MRI image and view the predicted category with probability scores.

### Model Information
Displays the model architecture, training configuration, dataset information, and evaluation results.

### Feedback
Provides a simple form for submitting feedback about the application.

## Technologies

- Python
- TensorFlow
- Keras
- VGG19
- NumPy
- Pandas
- Pillow
- Streamlit

## Project Structure

```text
Brain-Tumor-Classification/
│
├── app.py
├── brain_tumor_vgg19_final.keras
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Brain-Tumor-Classification.git
cd Brain-Tumor-Classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the application

```bash
streamlit run app.py
```

The application will then be available through the local Streamlit URL shown in the terminal.

## Requirements

The `requirements.txt` file contains the main dependencies:

```text
streamlit
tensorflow
numpy
pandas
Pillow
```

## Future Development

The project can be extended with:

- Grad-CAM based visual explanations
- More diverse MRI datasets
- Additional tumor categories
- Improved MRI/non-MRI image validation
- Model comparison with newer architectures
- Cloud deployment
- Mobile-friendly access
- Better confidence calibration
- Suspicious-region localization

## Limitations

The model is trained for the four categories represented in the dataset. Images outside the training distribution may produce unreliable predictions.

The current image validation step is a basic quality check and should not be treated as a medical imaging verification system.

## Medical Disclaimer

This application is intended for **educational and research purposes only**. It is not a medical diagnostic system and should not be used to make clinical decisions.

A prediction from this model must not replace examination, diagnosis, or treatment advice from a qualified healthcare professional.

## Developer

**Ravi Shankar Kumar**  
Computer Science and Engineering (Data Science)  
Government Engineering College, Sheohar

---

If you find this project useful for learning or research, you are welcome to explore the implementation and build upon it.
