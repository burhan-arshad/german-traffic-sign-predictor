# 🚦 GTSRB Traffic Sign Recognition

A deep learning project for recognizing and classifying German traffic signs using a Convolutional Neural Network (CNN).

The model is trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset and can classify images into **43 different traffic sign categories**.

## 🌐 Live Demo

🚀 **Try the live application:**

[![Open Live Demo] <https://german-traffic-sign-predictor-burhan.streamlit.app/>

---

## 📌 Project Overview

Traffic sign recognition is an important computer vision problem with applications in:

* 🚗 Autonomous driving
* 🛣️ Driver assistance systems
* 🚦 Intelligent transportation systems
* 📷 Real-time traffic monitoring
* 🤖 Computer vision systems

This project uses a custom CNN to learn visual features from traffic sign images and classify them into one of 43 traffic sign classes.

The complete workflow includes:

**Dataset → EDA → Preprocessing → Data Augmentation → CNN → Training → Evaluation → Deployment**

---

## 🧠 Model Architecture

The model is built using TensorFlow/Keras.

```text
Input Image
   │
   ▼
48 × 48 × 3
   │
   ▼
Data Augmentation
   │
   ├── Random Rotation
   ├── Random Zoom
   └── Random Contrast
   │
   ▼
Rescaling (1/255)
   │
   ▼
Conv2D (32 filters)
   │
   ▼
MaxPooling2D
   │
   ▼
Conv2D (64 filters)
   │
   ▼
MaxPooling2D
   │
   ▼
Conv2D (128 filters)
   │
   ▼
MaxPooling2D
   │
   ▼
Flatten
   │
   ▼
Dense (128)
   │
   ▼
Dropout (0.5)
   │
   ▼
Dense (43)
   │
   ▼
Softmax
   │
   ▼
Traffic Sign Class
```

---

## 📊 Dataset

The project uses the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

### Dataset Details

* **43 traffic sign classes**
* Training images organized by class
* Separate test set
* Test labels provided through `Test.csv`
* Images resized to `48 × 48`
* RGB images

The dataset contains different traffic signs including:

* Speed limits
* No entry
* Stop
* Yield
* Priority road
* Road work
* Pedestrian crossing
* Traffic signals
* Keep right/left
* Roundabout
* Dangerous curves
* And many more

---

## 🔍 Exploratory Data Analysis

Before training, the dataset was analyzed to understand:

* Class distribution
* Sample traffic sign images
* Image dimensions
* Class imbalance
* Dataset structure

The training dataset contains an imbalance between different traffic sign classes.

To reduce the effect of class imbalance, **class weights** were calculated and supplied during model training.

---

## ⚙️ Data Preprocessing

Images are resized to:

```text
48 × 48 × 3
```

The model performs normalization internally using:

```python
Rescaling(1./255)
```

### Data Augmentation

The training pipeline uses:

* Random Rotation
* Random Zoom
* Random Contrast

Horizontal flipping was intentionally avoided because flipping traffic signs can change their semantic meaning.

---

## 🏋️ Training

The dataset was divided into:

* **80% Training**
* **20% Validation**

The split was performed using stratification to maintain class proportions across training and validation sets.

The test dataset was kept separate and was used only for final evaluation.

### Training Configuration

| Parameter         | Value                           |
| ----------------- | ------------------------------- |
| Image Size        | 48 × 48                         |
| Batch Size        | 32                              |
| Number of Classes | 43                              |
| Optimizer         | Adam                            |
| Loss Function     | Sparse Categorical Crossentropy |
| Epochs            | Up to 30                        |
| Dropout           | 0.5                             |
| Early Stopping    | Yes                             |
| Model Checkpoint  | Yes                             |
| Class Weights     | Yes                             |

---

## 📈 Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification Report
* Confusion Matrix

The final evaluation is performed on the held-out GTSRB test dataset.

### Results

| Metric                   |               Score |
| ------------------------ | ------------------: |
| Test Accuracy            | **ADD_RESULT_HERE** |
| Best Validation Accuracy | **ADD_RESULT_HERE** |

> Update these values after the final model evaluation.

---

## 🚀 Streamlit Application

The trained CNN is deployed using **Streamlit**.

The application allows users to:

1. Upload a traffic sign image.
2. Preprocess the image.
3. Run the trained CNN.
4. Display the predicted traffic sign.
5. Display prediction confidence.
6. Display the top 5 predictions.

### Example

```text
Upload Image
      ↓
Image Preprocessing
      ↓
CNN Model
      ↓
43-Class Prediction
      ↓
Predicted Traffic Sign
      ↓
Confidence + Top 5 Predictions
```

---

## 🗂️ Project Structure

```text
gtsrb-traffic-sign-recognition/
│
├── app.py
├── best_traffic_sign_model.keras
├── requirements.txt
├── README.md
├── .gitignore
│
└── notebook/
    └── traffic_sign_recognition.ipynb
```

The dataset is intentionally excluded from the repository because it is unnecessary for running the deployed application.

---

## 🛠️ Technologies Used

### Programming

* Python

### Machine Learning / Deep Learning

* TensorFlow
* Keras
* Scikit-learn

### Data Processing

* NumPy
* Pandas
* Pillow

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit
* Streamlit Community Cloud

### Development

* Google Colab
* Jupyter Notebook
* Git
* GitHub

---

## 💻 Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd gtsrb-traffic-sign-recognition
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

## 🌐 Deployment

This application can be deployed using Streamlit Community Cloud.

Basic deployment workflow:

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
Install requirements.txt
       ↓
Load best_traffic_sign_model.keras
       ↓
Run app.py
       ↓
Public Web Application
```

Streamlit Community Cloud supports deploying directly from GitHub and provides a public URL for the deployed application.

---

## 🎯 Future Improvements

Possible improvements for future versions include:

* Increasing image resolution from 48×48 to 64×64
* Using Region of Interest (ROI) cropping
* Experimenting with Batch Normalization
* Improving CNN architecture
* Transfer learning with pretrained models
* Advanced class imbalance techniques
* Model explainability using Grad-CAM
* Real-time webcam traffic sign recognition
* Mobile deployment
* Real-time video traffic sign detection

---

## 📚 Learning Outcomes

Through this project, the following concepts were implemented:

* Image classification
* Convolutional Neural Networks
* TensorFlow/Keras
* Data augmentation
* Image preprocessing
* Stratified train/validation splitting
* Class imbalance handling
* Early stopping
* Model checkpointing
* Confusion matrix analysis
* Classification reports
* Model deployment
* Streamlit application development

---

## 👨‍💻 Author

**Burhan Arshad**

BS Computer Science Student
Machine Learning & AI Enthusiast

### Portfolio

[GitHub](YOUR_GITHUB_PROFILE_URL)

[LinkedIn](YOUR_LINKEDIN_URL)

---

## ⭐ Project

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.
