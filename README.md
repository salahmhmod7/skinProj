# 🩺 Skin Lesion Classification — DermaScan AI

An end-to-end **Deep Learning and Computer Vision application** for the classification of dermatological skin lesions using the **HAM10000 dataset**.

The system classifies dermoscopic images into **7 diagnostic categories** and provides a probability distribution and confidence score for each prediction. The trained models are exposed through an API and can be deployed using Docker.

> **Disclaimer:** This project is intended for educational and research purposes and is **not a medical diagnostic tool**. Predictions should not be used as a substitute for evaluation by a qualified healthcare professional.

---

## 📖 Description

**DermaScan AI** is a computer vision project built with **PyTorch** for automated skin lesion image classification.

The system processes dermoscopic images through a deep learning inference pipeline that includes image preprocessing, model inference, probability estimation, and diagnostic class mapping.

The project uses the **HAM10000** dataset and supports classification across seven common categories of skin lesions, including melanoma and basal cell carcinoma.

The project also includes a modular inference pipeline and a containerized API for serving model predictions.

---

## 🎯 Diagnostic Classes

The model classifies dermoscopic images into the following seven categories:

| Code    | Diagnosis                                       |
| ------- | ----------------------------------------------- |
| `akiec` | Actinic keratoses and intraepithelial carcinoma |
| `bcc`   | Basal cell carcinoma                            |
| `bkl`   | Benign keratosis-like lesions                   |
| `df`    | Dermatofibroma                                  |
| `mel`   | Melanoma                                        |
| `nv`    | Melanocytic nevi                                |
| `vasc`  | Vascular lesions                                |

---

## ✨ Features

### 🖼️ Image Classification

* Accepts dermoscopic skin lesion images.
* Performs image preprocessing and normalization.
* Predicts probabilities across 7 diagnostic categories.
* Returns the predicted class and confidence score.

### 🧠 Deep Learning

* Built with **PyTorch** and **Torchvision**.
* Includes multiple trained model variants.
* Supports experimentation with different model architectures and training configurations.
* Includes clinical model variants for experimentation.

### 🔬 Inference Pipeline

The inference logic is separated into dedicated modules:

```text
Image
  ↓
Preprocessing
  ↓
Model
  ↓
Softmax Probabilities
  ↓
Predicted Class
  ↓
Confidence Score
```

### 🌐 API Integration

* Provides an API for model inference.
* Separates the ML model from the application layer.
* Makes the trained model accessible to external applications.

### 🐳 Containerized Deployment

* Includes a `Dockerfile`.
* Includes Docker Compose configuration.
* Provides a reproducible deployment environment.

### 🇪🇬 Arabic Output Support

The inference pipeline can provide:

* Arabic diagnostic descriptions.
* Predicted class.
* Confidence score.
* Human-readable medical information.

---

## 🛠️ Tech Stack

| Category                   | Technology       |
| -------------------------- | ---------------- |
| Programming Language       | Python 3.11+     |
| Deep Learning              | PyTorch          |
| Computer Vision            | Torchvision      |
| Data Processing            | NumPy, Pandas    |
| Machine Learning Utilities | Scikit-learn     |
| API                        | FastAPI / Flask  |
| Experimentation            | Jupyter Notebook |
| Containerization           | Docker           |
| Orchestration              | Docker Compose   |
| Dataset                    | HAM10000         |

---

## 📂 Project Structure

```text
skinProj/
│
├── data/
│   ├── HAM10000_images_part_1/
│   │   └── *.jpg
│   │
│   ├── HAM10000_images_part_2/
│   │   └── *.jpg
│   │
│   ├── best_model.pth
│   ├── best_model_v2.pth
│   ├── clinical_best_model.pth
│   ├── clinical_final_model.pth
│   ├── convext_clinical_best.pth
│   │
│   └── *.csv
│       └── Dataset metadata and train/validation/test splits
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   │   └── API endpoints
│   │
│   ├── infer.py
│   │   └── Image preprocessing and inference
│   │
│   └── model.py
│       └── Model architecture
│
├── archive.zip
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── Untitled-1.ipynb
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/salahmhmod7/skinProj.git

cd skinProj
```

### 2. Create a Virtual Environment

#### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

#### macOS / Linux

```bash
python -m venv .venv

source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Training & Experimentation

The project includes a Jupyter Notebook containing the experimentation and model development workflow.

Open:

```text
Untitled-1.ipynb
```

using **VS Code**, **Jupyter Notebook**, or **JupyterLab**.

The notebook can be used to explore:

* Dataset preprocessing
* Data analysis
* Image transformations
* Model training
* Validation
* Model evaluation
* Model comparison
* Inference experiments

---

## 🌐 Running the API

To start the API locally:

```bash
python src/api.py
```

Depending on the API implementation, the service can then be accessed through the configured host and port.

---

## 🐳 Running with Docker

Docker is the recommended approach for running the application in a reproducible environment.

### Build and Start

```bash
docker-compose up --build
```

To run the services in the background:

```bash
docker-compose up --build -d
```

To stop the containers:

```bash
docker-compose down
```

---

## 🔄 End-to-End Pipeline

The complete system follows this workflow:

```text
                  ┌─────────────────────┐
                  │  Dermoscopic Image  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Image Preprocessing │
                  │ Resize / Normalize  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   PyTorch Model     │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   Class Logits      │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Softmax Probabilities│
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Predicted Diagnosis │
                  │ + Confidence Score  │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │      API Response   │
                  └─────────────────────┘
```

---

## 📊 Model Output

A prediction contains information such as:

```json
{
  "prediction": "mel",
  "confidence": 0.91,
  "description": "Melanoma"
}
```

The exact response structure depends on the implementation in `src/api.py`.

---

## 🔮 Future Improvements

Potential improvements for the project include:

* [ ] Add Grad-CAM / explainable AI visualization.
* [ ] Add lesion localization or segmentation.
* [ ] Add model performance metrics to the API.
* [ ] Add confidence calibration.
* [ ] Add uncertainty estimation.
* [ ] Add model versioning.
* [ ] Add automated API testing.
* [ ] Add comprehensive validation and monitoring.
* [ ] Add a dedicated web interface.
* [ ] Add experiment tracking.
* [ ] Add CI/CD pipeline.
* [ ] Deploy the API to a cloud platform.
* [ ] Add automated model evaluation.

---

## ⚠️ Medical Disclaimer

This project is developed for **educational, research, and software engineering purposes**.

The predictions generated by the models are not intended to provide medical diagnosis, treatment recommendations, or clinical decisions.

A qualified healthcare professional should evaluate any real medical concern.

---

## 👨‍💻 Author

**Salah Mahmoud**

* GitHub: [@salahmhmod7](https://github.com/salahmhmod7)
* Email: [flaysefeco.1@gmail.com](mailto:flaysefeco.1@gmail.com)

---

## ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub!
