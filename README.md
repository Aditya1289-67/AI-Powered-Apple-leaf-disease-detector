# 🌿 Apple Leaf Disease Detector

An AI-powered web application that identifies diseases in apple leaves using a Convolutional Neural Network (CNN). This tool helps farmers and gardeners get an instant, accurate diagnosis to facilitate early treatment.

---

## ✨ Demo

![web app](https://github.com/user-attachments/assets/70f67d80-b221-4b47-bdb6-05c726844077)
![diagnosis result](https://github.com/user-attachments/assets/ec545c6b-dabc-40f6-a78d-e09cbc359b22)
![ai advice](https://github.com/user-attachments/assets/924b5a1e-1711-42c8-8390-37e800b311da)




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

## 🧠 Model Architecture

The core of this application is a **Convolutional Neural Network (CNN)**. The model was trained on a diverse dataset of apple leaf images to learn distinctive features for each disease class. The architecture is based on a pre-trained model like **EfficientNetB0**, fine-tuned for this specific classification task to achieve high accuracy.

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

3.  **Download the trained model file** (e.g., `apple_disease_model.h5`) and place it in a `models/` directory within the project folder. *(Note: You may need to provide a link for others to download your model if it's too large for GitHub).*

## 🔧 Usage

To run the Streamlit application locally, execute the following command in your terminal from the project's root directory:

```sh
streamlit run app.py
```

Open your web browser and navigate to `http://localhost:8501`.

## 📂 Project Structure

```
apple-leaf-disease-detector/
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


## 📬 Contact: khotaditya375@gmail.com

Adityanand Khot - https://www.linkedin.com/in/adityanandkhot-x2?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app 
Project Link: [https://github.com/your-username/apple-leaf-disease-detector](https://github.com/your-username/apple-leaf-disease-detector)
