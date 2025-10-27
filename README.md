## 🧠 Alzheimer MRI Image Classifier

An intelligent Deep Learning Web App that classifies Alzheimer’s disease stages from brain MRI images using FastAI, Hugging Face Hub, and Gradio.
The model predicts the cognitive state (e.g., Non-Demented, Very Mild, Mild, Moderate Dementia) from an uploaded MRI scan.

---

## 🚀 Features

```text
🧩 Built with FastAI for fast and efficient transfer learning

☁️ Model hosted on Hugging Face Hub (Kutti-AI/alzheimer)

🖼️ Clean Gradio web interface for live predictions

⚡ Quick deployment locally or in the cloud (e.g., Hugging Face Spaces)

🧠 Classifies multiple dementia stages accurately
```
---

## 🧠 Model Details

Framework: FastAI (PyTorch backend)

Type: Convolutional Neural Network (CNN)

Dataset: Alzheimer MRI Dataset (NonDemented → Moderate Demented)

Exported Model File: model.pkl

Hugging Face Repository: Kutti-AI/alzheimer

---

## 🧩 Project Structure
```text
alzheimer-classifier/
│
├── app.py               # Gradio web app (main file)
├── tileshop.jpg
├── verymild.jpeg
├── moderate.png          # images
├── md.PNG
├── requirements.txt     # Python dependencies
├── README.md            # Documentation
```
----

## ⚙️ Installation

# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/alzheimer-classifier.git
cd alzheimer-classifier

# 2️⃣ Install dependencies
pip install -r requirements.txt


or manually:

pip install fastai gradio huggingface_hub pillow

# 🔑 Hugging Face Authentication

To download your model from the Hugging Face Hub securely:

Option 1: Set environment variable
export hf_token="your_huggingface_token"

Option 2: Inside Python
import os
os.environ["hf_token"] = "your_huggingface_token"

Option 3: .env file (optional)
hf_token=your_huggingface_token

# 🧰 Run the App

Launch the Gradio app locally:

python app.py


Once launched, you’ll see:

Running on local URL:  http://127.0.0.1:7860


Open the URL in your browser to use the classifier.

---

## 🖼️ Example Images

The app includes example MRI scans:

tileshop.jpg — Non Demented

verymild.jpeg — Very Mild Demented

moderate.png — Moderate Demented

md.PNG — Mild Demented

You can also upload your own MRI brain scan for prediction.

## 🧾 Code Overview

# ✅ Load the Model
model_path = hf_hub_download(repo_id="Kutti-AI/alzheimer", filename="model.pkl")
learn = load_learner(model_path)

# ✅ Define the Prediction Function
def classify_image(im):
    learn.model.eval()
    if not isinstance(im, Image.Image):
        im = Image.fromarray(im)
    with learn.no_bar():
        pred, idx, probs = learn.predict(im)
    return dict(zip(learn.dls.vocab, map(float, probs)))

# ✅ Create the Gradio Interface
intf = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    examples=[
        ['tileshop.jpg'],
        ['verymild.jpeg'],
        ['moderate.png'],
        ['md.PNG']
    ]
)
intf.launch()

## 🧾 requirements.txt
```text
fastai==2.7.13
fastcore==1.5.55
torch==2.1.2
torchvision==0.16.2
transformers==4.40.2
datasets==2.13.1
numpy==1.24.4
pandas==2.2.3
matplotlib==3.7.2
spacy==3.8.7
gradio==4.44.1
timm==1.0.21
```
--- 

## 🌐 Deployment Options
# 🚀 Hugging Face Spaces

Just upload:

app.py

requirements.txt

The app will automatically deploy and become publicly accessible.

# 💻 Localhost / Cloud VM
python app.py


Then open the link printed in your terminal.

# 🧿 License

This project is licensed under the MIT License — feel free to use, modify, and share it responsibly.

---

## ✍️ Author

👤 Husen (Kutti-AI)
💌 Machine Learning | Deep Learning | LLM Enthusiast
🌐 Hugging Face Profile

“AI isn’t just about intelligence — it’s about compassion, when used to heal minds.”
— Husen (Kutti-AI) 🧠✨

---
