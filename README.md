# 👥 Crowd Density Estimation using P2PNet

<p align="center">

**An AI-powered computer vision system for crowd counting, point-based person localization, and density heatmap generation.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-ff4b4b?logo=streamlit)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5c3ee8?logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-Educational-green)](#-license)

</p>

---

## 🎥 Project Demo

> Upload a crowd image, run AI-powered analysis, visualize detected people, and generate an interpretable crowd density heatmap.

<!-- Replace this with your GitHub-hosted demo video URL -->

https://github.com/Codeabc27/CrowdCounting-P2PNet/blob/main/assets/Recording%202026-07-22%20102742.mp4

---

## 📌 Overview

**Crowd Density Estimation using P2PNet** is a deep learning-based computer vision application designed to estimate the number of people present in crowded environments.

The system uses **P2PNet**, a point-based crowd counting architecture, to predict individual person locations instead of relying on traditional bounding-box detection.

The detected person points are then processed to:

* Estimate the total crowd count
* Visualize individual detection points
* Classify the crowd density level
* Generate a visual density heatmap
* Provide an interactive web-based analysis interface

The application is built with **PyTorch** for deep learning inference and **Streamlit** for the user interface.

---

## 🎯 Problem Statement

Traditional object detection approaches often struggle in highly crowded environments because of:

* Severe person-to-person occlusion
* Overlapping individuals
* Extremely dense scenes
* Small and partially visible people
* Complex camera angles

Bounding-box-based detectors may fail to accurately distinguish individuals in dense crowds.

This project addresses the problem using a **point-based crowd counting approach**, where each detected person is represented by a point coordinate.

---

## 💡 Solution

The system follows a simple and efficient pipeline:

```text
┌──────────────────────┐
│   Upload Crowd Image │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Image Preprocessing  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│       P2PNet         │
│  Deep Learning Model │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Person Point         │
│ Predictions          │
└───────┬───────┬──────┘
        │       │
        ▼       ▼
┌───────────┐ ┌───────────────┐
│ Crowd     │ │ Density       │
│ Count     │ │ Heatmap       │
└─────┬─────┘ └───────┬───────┘
      │               │
      └───────┬───────┘
              ▼
     ┌──────────────────┐
     │ Streamlit Result │
     │    Dashboard     │
     └──────────────────┘
```

---

## ✨ Key Features

### 👥 Crowd Counting

Estimates the number of people present in the uploaded image using P2PNet point predictions.

### 📍 Point-Based Person Localization

Visualizes predicted person locations directly on the original image.

### 🔥 Density Heatmap Generation

Generates a heatmap that visually represents areas of higher crowd concentration.

### 📊 Crowd Density Classification

The system classifies crowd density based on the estimated count.

| Estimated Count | Density Classification |
| --------------: | ---------------------- |
|               0 | No Crowd               |
|            1–19 | Low Density            |
|           20–49 | Medium Density         |
|           50–99 | High Density           |
|            100+ | Very High Density      |

### ⚡ CPU and GPU Inference

The application automatically selects the available computation device:

* NVIDIA CUDA GPU
* CPU

### 🖥️ Interactive Web Interface

The Streamlit application provides:

* Image upload
* Image preview
* Crowd analysis
* Count visualization
* Density classification
* Detection point visualization
* Heatmap generation
* Heatmap download

---

## 🧠 Model Architecture

### P2PNet

P2PNet is a point-based crowd counting model designed specifically for dense crowd scenes.

Unlike conventional object detection models that predict bounding boxes, P2PNet predicts points representing individual people.

### Traditional Object Detection

```text
Image
  │
  ▼
Bounding Box Detection
  │
  ▼
Person Boxes
  │
  ▼
Count
```

### P2PNet Approach

```text
Image
  │
  ▼
P2PNet
  │
  ▼
Person Point Predictions
  │
  ▼
Point Count
  │
  ▼
Crowd Density Estimation
```

This approach is particularly useful when individuals are heavily occluded or packed closely together.

---

## 🏗️ System Architecture

```text
                 ┌─────────────────────┐
                 │   User Uploads      │
                 │   Crowd Image       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Streamlit Frontend  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Image Preprocessing │
                 │   PIL / OpenCV      │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │    P2PNet Model     │
                 │    PyTorch Runtime  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Point Predictions   │
                 └──────────┬──────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     ┌────────────────┐          ┌────────────────┐
     │ Crowd Counting │          │ Heatmap Engine  │
     └───────┬────────┘          └───────┬────────┘
             │                           │
             └─────────────┬─────────────┘
                           ▼
                 ┌─────────────────────┐
                 │ Result Visualization│
                 │  & Download Output  │
                 └─────────────────────┘
```

---

## 📁 Project Structure

```text
CrowdCounting-P2PNet/
│
├── app.py
│   └── Streamlit application and user interface
│
├── inference.py
│   └── Model loading and inference pipeline
│
├── heatmap.py
│   └── Crowd density heatmap generation
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
├── .gitignore
│   └── Ignored files and directories
│
├── assets/
│   └── Project screenshots and demo assets
│
└── weights/
    └── SHTechA.pth
        └── Trained model checkpoint
```

> **Note:** Large model weight files should not be committed directly to GitHub.

---

## 🛠️ Technology Stack

| Category                | Technology   |
| ----------------------- | ------------ |
| Programming Language    | Python       |
| Deep Learning Framework | PyTorch      |
| Model Architecture      | P2PNet       |
| Computer Vision         | OpenCV       |
| Numerical Computing     | NumPy        |
| Image Processing        | Pillow       |
| Visualization           | Matplotlib   |
| Web Application         | Streamlit    |
| Scientific Computing    | SciPy        |
| Version Control         | Git & GitHub |

---

## ⚙️ Installation

### Prerequisites

Make sure the following are installed:

* Python 3.10 or higher
* Git
* pip
* Optional: NVIDIA GPU with CUDA support

---

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/CrowdCounting-P2PNet.git
```

Navigate to the project directory:

```bash
cd CrowdCounting-P2PNet
```

---

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

#### Linux/macOS

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dependencies

Example `requirements.txt`:

```text
torch
torchvision
numpy
opencv-python
Pillow
streamlit
matplotlib
scipy
```

Install manually if required:

```bash
pip install torch torchvision numpy opencv-python Pillow streamlit matplotlib scipy
```

---

## 🧠 Model Weights

The application requires a trained P2PNet model checkpoint.

Expected path:

```text
weights/SHTechA.pth
```

Expected structure:

```text
weights/
└── SHTechA.pth
```

The model is loaded during inference using the project inference pipeline.

> ⚠️ Model weights are excluded from GitHub because large binary files can significantly increase repository size.

### Model Weight Setup

1. Download the trained model checkpoint.
2. Create the `weights/` directory.
3. Place the model file inside it.
4. Verify that the filename matches the expected path.

---

## ▶️ Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 🖥️ Application Workflow

### Step 1: Upload Image

Upload a crowd image in one of the supported formats:

* JPG
* JPEG
* PNG

### Step 2: Preview

The uploaded image is displayed in the Streamlit interface.

### Step 3: Analyze

Click:

```text
Analyze Crowd
```

The P2PNet model performs inference on the uploaded image.

### Step 4: View Results

The system displays:

* Estimated number of people
* Crowd density classification
* Processing device
* Detection point visualization
* Density heatmap

### Step 5: Download Output

The generated heatmap can be downloaded as:

```text
crowd_density_heatmap.png
```

---

## 📊 Output Example

Example output:

```text
Estimated People: 87

Density Level: High Density

Processing Device: CUDA
```

The application produces two primary visual outputs:

### Detection Point Visualization

Each detected individual is represented by a point on the original image.

### Density Heatmap

The heatmap represents the spatial concentration of detected people.

Higher-intensity regions indicate areas with greater estimated crowd concentration.

---

## 📈 Density Classification Logic

The current classification system is based on the estimated crowd count:

```text
Count = 0
        │
        ▼
    No Crowd

Count = 1–19
        │
        ▼
   Low Density

Count = 20–49
        │
        ▼
 Medium Density

Count = 50–99
        │
        ▼
  High Density

Count >= 100
        │
        ▼
Very High Density
```

---

## 🔬 Technical Pipeline

```text
1. Image Upload
       │
       ▼
2. Image Decoding
       │
       ▼
3. Image Preprocessing
       │
       ▼
4. Tensor Conversion
       │
       ▼
5. P2PNet Inference
       │
       ▼
6. Point Prediction Filtering
       │
       ▼
7. Crowd Count Calculation
       │
       ▼
8. Density Classification
       │
       ▼
9. Heatmap Generation
       │
       ▼
10. Result Visualization
```

---

## 🚀 Future Roadmap

### Phase 1 — Application Improvements

* [ ] Improved result dashboard
* [ ] Image comparison mode
* [ ] Custom density thresholds
* [ ] Enhanced visualization controls

### Phase 2 — Real-Time Monitoring

* [ ] Video upload support
* [ ] Webcam integration
* [ ] CCTV stream support
* [ ] Real-time crowd counting

### Phase 3 — Intelligent Alerting

* [ ] Crowd threshold alerts
* [ ] Overcrowding notifications
* [ ] Region-based monitoring
* [ ] Safety monitoring dashboard

### Phase 4 — Production Deployment

* [ ] REST API integration
* [ ] Docker containerization
* [ ] Cloud deployment
* [ ] Model optimization
* [ ] GPU inference optimization
* [ ] Monitoring and logging

---

## ⚠️ Limitations

The performance of the model may be affected by:

* Low-resolution images
* Severe image blur
* Extreme occlusion
* Unusual camera perspectives
* Poor lighting conditions
* Domain differences between training and deployment images

For production use, the model should be evaluated and potentially fine-tuned on data representative of the target environment.

---

## 🔒 Security and Data Privacy

This application processes uploaded images for inference.

For production deployment:

* Avoid storing uploaded images unnecessarily.
* Implement secure file validation.
* Restrict maximum upload size.
* Use secure deployment practices.
* Avoid logging sensitive image data.
* Apply appropriate access controls.

---

## 🧪 Testing Recommendations

The system should be tested using images with different:

* Crowd densities
* Camera angles
* Lighting conditions
* Image resolutions
* Occlusion levels
* Background environments

Recommended evaluation metrics for crowd counting models include:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

---

## 🤝 Contributing

Contributions are welcome.

### Contribution Workflow

```bash
# Fork the repository

# Create a feature branch
git checkout -b feature/your-feature

# Make your changes

# Commit your changes
git commit -m "Add: your feature description"

# Push the branch
git push origin feature/your-feature

# Open a Pull Request
```

Please ensure that contributions:

* Follow clean coding practices.
* Include appropriate documentation.
* Do not commit model weights or sensitive files.
* Maintain project functionality.

---

## 📄 License

This project is intended for educational, research, and demonstration purposes.

---

## 👨‍💻 Author

### SASI R

**AI/ML Engineer | Artificial Intelligence & Data Science**

Interested in:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Computer Vision
* Generative AI
* Data Science
---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

Your support is appreciated!

---

<p align="center">

**Built with Python, PyTorch, P2PNet, and Streamlit**

</p>
