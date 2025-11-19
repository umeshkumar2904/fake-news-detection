# --- ONE-CELL SIMPLE FAKE-NEWS APP FOR GOOGLE COLAB ---
# 1) Paste your dataset path below (see DATASET_PATH)
# 2) Run this cell (it will train quickly and open a Gradio app)


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio is not installed. Run `pip install gradio` to enable the UI.")

# ====== >>> PASTE YOUR DATASET PATH HERE (exactly replace the string) <<<<< ======
# Examples:
# "/content/WELFake_Dataset.csv"
# "/content/drive/MyDrive/fake_news.csv"
DATASET_PATH = r"C:\Users\dell\OneDrive\Desktop\fake news detection\new one\WELFake_Dataset.csv"
# ==================================================================================

def load_dataset(path):
    if not path or path.startswith("###") or not os.path.exists(path):
        # fallback tiny demo dataset (runs instantly)
        demo = {
            'text': [
                "Scientists discover new exoplanet with conditions similar to Earth",
                "Government secretly replaced city water with mind-control serum, report says",
                "Local sports team wins championship after dramatic comeback",
                "Study proves eating chocolate every day increases lifespan by 70 percent",
            ],
            'label': [1, 0, 1, 0]
        }
        print("Using demo dataset (paste a valid path into DATASET_PATH to use your CSV).")
        return pd.DataFrame(demo)
    # Attempt to read CSV with python engine for better parsing of malformed lines
    # on_bad_lines='warn' is used to log warnings instead of raising an error for bad lines.
    df = pd.read_csv(path, engine='python', on_bad_lines='warn')
    return df

# Load
df = load_dataset(DATASET_PATH)

# Normalize common column names
if 'content' in df.columns and 'text' not in df.columns:
    df = df.rename(columns={'content':'text'})
if 'label' not in df.columns and 'target' in df.columns:
    df = df.rename(columns={'target':'label'})
if 'text' not in df.columns:
    raise ValueError("Dataset must have a 'text' or 'content' column. Rename it and retry.")

# Keep only needed columns and drop nulls
df = df[['text', 'label']].dropna(subset=['text', 'label']) # Ensure 'label' column has no NaNs initially if possible

# If labels are strings like 'fake'/'real', convert to 0/1
if df['label'].dtype == object:
    df['label'] = df['label'].astype(str).str.lower().map({'fake':0,'false':0,'0':0,'real':1,'true':1,'1':1})

# After mapping, there might be new NaNs if some labels didn't match. Drop them.
df = df.dropna(subset=['label'])

# Quick speed hack: if dataset is huge, sample up to 4000 rows for fast training
MAX_ROWS = 4000
if len(df) > MAX_ROWS:
    df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
    print(f"Dataset sampled to {MAX_ROWS} rows for faster training.")

df['text'] = df['text'].astype(str).str.lower()

X = df['text'].values
y = df['label'].astype(int).values

# Train/test split (small fallback if tiny dataset)
test_size = 0.2 if len(df) > 4 else 0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y))>1 else None)

# Fast pipeline: small TF-IDF + Logistic Regression
pipeline = make_pipeline(
    TfidfVectorizer(max_features=2000, ngram_range=(1,2)),
    LogisticRegression(max_iter=300, solver='liblinear')
)

print("Training (this should be quick)...")
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Done. Test accuracy: {acc:.3f}  —  You can improve this by using a larger dataset or transformers.")

# Gradio app
def predict(text):
    if not text or str(text).strip()=="":
        return "Error: empty input", 0.0
    prob = pipeline.predict_proba([text])[0]
    classes = pipeline.named_steps['logisticregression'].classes_
    # find prob for class 1 (real) if present
    if 1 in classes:
        idx_real = list(classes).index(1)
        real_prob = float(prob[idx_real])
        fake_prob = 1.0 - real_prob
        label = "Real" if real_prob >= 0.5 else "Fake"
        confidence = round(max(real_prob, fake_prob), 3)
    else:
        idx = int(np.argmax(prob))
        label = str(classes[idx])
        confidence = round(float(prob[idx]), 3)
    return label, confidence

if GRADIO_AVAILABLE:
    ui = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(lines=6, placeholder="Paste news/article text here..."),
        outputs=[gr.Textbox(label="Prediction"), gr.Number(label="Confidence")],
        title="Quick Fake News Detector (TF-IDF + LR)",
        description="Replace DATASET_PATH at the top with your CSV path (must have 'text' and 'label' columns)."
    )
    print("Launching Gradio app (share=True creates a public URL)...")
    ui.launch(share=True)
else:
    print("Running in CLI mode. Type text and press Enter (blank to exit).")
    try:
        while True:
            text = input("Enter text: ").strip()
            if not text:
                break
            label, confidence = predict(text)
            print(f"Prediction: {label} | Confidence: {confidence}")
    except KeyboardInterrupt:
        pass