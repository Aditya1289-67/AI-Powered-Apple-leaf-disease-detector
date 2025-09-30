# 🌿 Apple Leaf Disease Detector

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered web application that identifies diseases in apple leaves using a Convolutional Neural Network (CNN). This tool helps farmers and gardeners get an instant, accurate diagnosis to facilitate early treatment.

---

## ✨ Demo

![web app](https://github.com/user-attachments/assets/a7e673a6-2d40-4dbc-bf66-2ee39149ca68)
![diagnosis result](https://github.com/user-attachments/assets/6e46235a-7a79-4353-9b9e-9815f60d7a9b)
![ai advice](https://github.com/user-attachments/assets/27b449f5-9ff7-4205-8f0d-53ce16e5fc8e)



## 📖 Table of Contents

- [About The Project](#-about-the-project)
- [🚀 Features](#-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [🧠 Model Architecture](#-model-architecture)
- [💾 Dataset](#-dataset)
- [🏁 Getting Started](#-getting-started)
- [🔧 Usage](#-usage)
- [📂 Project Structure](#-project-structure)
- [📄 License](#-license)
- [📬 Contact](#-contact)

## 📌 About The Project

This project aims to provide a simple, accessible, and efficient solution for detecting common diseases in apple leaves. By leveraging the power of deep learning, the application can analyze an image of a leaf and predict whether it is healthy or affected by diseases like Apple Scab, Black Rot, or Cedar Apple Rust. The user-friendly interface is built with Streamlit, making it easy for anyone to upload an image and receive an instant diagnosis.

## 🚀 Features

-   **User-Friendly Web Interface:** Simple and intuitive UI for easy navigation.
-   **Image Upload:** Allows users to upload leaf images directly from their device.
-   **Real-time Prediction:** Provides instant classification of the leaf's condition.
-   **Displays Confidence Score:** Shows the model's confidence in its prediction.
-   **Cross-Platform:** Accessible from any web browser on desktop or mobile.

## 🛠️ Tech Stack

-   **Backend & ML:** Python
-   **Deep Learning Framework:** TensorFlow / Keras
-   **Web Framework:** Streamlit
-   **Image Processing:** Pillow, OpenCV
-   **Numerical Operations:** NumPy

.

🧠 Model Architecture
The core of this application is a custom Convolutional Neural Network (CNN), designed and trained from scratch specifically for this project. The model was trained on a dedicated dataset of apple leaf images to accurately distinguish between healthy leaves and those affected by common diseases.

Its architecture is built to learn the specific visual
http://googleusercontent.com/image_generation_content/1


## 💾 Dataset

The model was trained on a public dataset containing thousands of images of healthy and diseased apple leaves. This dataset provides a robust foundation for training computer vision models for agricultural applications.

-   **Source:** *Leave this space to add the source, e.g., Kaggle, PlantVillage, etc.*
-   **Link:** **`https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset`**
-   ## 🧠 Model & Training Options

You can get the model for this project in two ways:

1.  **Train from Scratch:** Download the full dataset using the link in the Dataset section.
    - **Important:** For training, you must use **only** the images located within the `apple` sub-folder of the dataset.

2.  **Use the Pre-trained Model:** For convenience, you can download the ready-to-use trained model directly.
    - **Model Download Link:** 'https://drive.google.com/file/d/1HSHrhT7MD-D1jHXdQui_TmtXByNY6gZk/view?usp=sharing`
This comprehensive dataset typically includes the following classes:
* Healthy
* Apple Scab
* Black Rot
* Cedar Apple Rust

## 🏁 Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

Make sure you have Python 3.9+ and pip installed on your system.
-   [Python](https://www.python.org/downloads/)
-   [pip](https://pip.pypa.io/en/stable/installation/)

### Installation


1.  **Create and activate a virtual environment (recommended):**
    ```sh
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

3.  **Set up your secrets:**
    - Create a file at `.streamlit/secrets.toml`.
    - Add any necessary API keys or secret credentials to this file. See the [Streamlit Docs](https://docs.streamlit.io/library/advanced-features/secrets-management) for more info.

4.  **Download the trained model file** (e.g., `apple_disease_model.h5`) and place it in the `models/` directory.

## 🔧 Usage

To run the Streamlit application locally, execute the following command in your terminal from the project's root directory:

```sh
streamlit run app2.py
```

Open your web browser and navigate to `http://localhost:8501`.

## 📂 Project Structure

```
apple-leaf-disease-detector/
│
├── 📁 .streamlit/
│   └── 🔑 secrets.toml        # Secret keys (e.g., API keys)
│
├── 📁 data/
│   └── .gitkeep             # Placeholder for data files/scripts
│
├── 📄 app.py                # Main Streamlit application script
├── 📁 models/
│   └── 🧠 apple_disease_model.h5  # The trained H5 model file
│
├── 📁 assets/                # For images, logos, etc.
│   └── 🖼️ logo.png
│
├── 📄 requirements.txt      # List of Python dependencies
├── 📄 .gitignore             # Files to be ignored by Git
└── 📄 README.md              # Project documentation (this file)
```


## 📬 Contact

Your Name - https://www.linkedin.com/in/adityanandkhot-x2?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app
