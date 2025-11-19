📰 Fake News Detection Web App

A lightweight, fast, and easy Fake News Detector using TF-IDF + Logistic Regression + Gradio UI

🚀 Overview

This project is a simple and fast Fake News Detection web application that you can run easily in Google Colab or locally on your system.
It combines:

TF-IDF Vectorizer for text processing

Logistic Regression for classification

Gradio Web Interface for a clean and interactive UI

Perfect for beginners learning Machine Learning, students working on projects, or anyone wanting a quick fake news classifier.

📂 Features

✅ One-cell runnable code
✅ Paste your dataset path and train instantly
✅ Gradio Web Interface (with shareable public link in Colab)
✅ Handles missing/invalid labels automatically
✅ Automatic fallback dataset if no CSV found
✅ Very fast training time
✅ Supports Windows, Colab, and Linux

📁 Dataset Requirements

Your CSV file must have:

Column	Description
text	News article or statement
label	0 = Fake, 1 = Real

Accepted label formats:

0, 1

fake, real

true, false

Automatic cleaning will convert them to 0 and 1.

📌 How to Use (Google Colab)
1. Upload your dataset to Colab or Google Drive
2. Paste its path in the code

Inside the script, find this part:

DATASET_PATH = "### >>> PASTE YOUR DATASET PATH HERE <<<"


Replace it with your actual file path, e.g.:

DATASET_PATH = "/content/WELFake_Dataset.csv"

3. Run the code
4. Gradio will generate a public URL

You will see something like:

Running on public URL: https://xxxx.gradio.app


Click the link to open your web app.

🧠 How It Works

Load Dataset

If the dataset path is invalid, a small demo dataset is used.

Preprocessing

Lowercasing

Removing empty values

Converting labels to 0/1

Training

Uses TF-IDF (2000 max features)

Logistic Regression (very fast and accurate)

Prediction

Model predicts Fake or Real with a confidence score.

Gradio Interface

Enter news text

Get predictions instantly

🧪 Sample Prediction Output
Prediction: Fake
Confidence: 0.87

📦 Installation (Local System)
1. Create environment
pip install pandas scikit-learn gradio numpy

2. Run the script
python fakenews02.py

🖥️ Project Structure
├── fakenews02.py      # Main script
├── README.md          # Documentation
└── dataset.csv        # (Your dataset)

🛠️ Technologies Used

Python

Pandas

NumPy

Scikit-Learn

Gradio

Logistic Regression

TF-IDF Vectorization

📈 Future Improvements (Optional)

Upgrade to BERT / DistilBERT

Add dataset cleaner

Add visualization (wordcloud, confusion matrix)

Deploy to HuggingFace Spaces

Add real-time news scraping

🤝 Contributing

Pull requests are welcome!
For major changes, open an issue first to discuss your ideas.

📜 License

This project is licensed under the MIT License.
