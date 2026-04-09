# 🔥 Real-Time Fire Detection System

> A deep learning system that detects fire in real-time using live webcam feeds,
> compares 4 CNN architectures, and sends instant Gmail alerts with buzzer sound.

---

## 📸 Screenshots

> Add your UI screenshots here

---

## 🎯 Overview

Fire detection using deep learning is a critical AI application that enables 
real-time fire identification from camera feeds — without human monitoring.

This system trains and compares **4 CNN architectures** on a fire/non-fire 
image dataset and deploys the best model to analyze live webcam feeds, 
triggering instant alerts when fire is detected.

---

## ✨ Features

- 🎥 Real-time webcam fire detection
- 🧠 4 CNN models trained & compared (VGG16, ResNet50, MobileNetV2, Xception)
- ✅ **92% accuracy** on best model
- 📧 Instant Gmail notification on fire detection
- 🔔 Audible buzzer alert
- 📊 Live monitoring dashboard (React.js)
- 🗄️ MongoDB logging for detection history

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| Frontend | React.js, HTML5, CSS3 |
| Backend | Python, Flask, REST APIs |
| AI / ML | TensorFlow, VGG16, ResNet50, MobileNetV2, Xception |
| Database | MongoDB |
| Alerts | Gmail API, Web Audio API |

---

## 🧠 How It Works

Fire Image Dataset
↓
Preprocessing (RGB→Gray, Resize 32x32, Labeling)
↓
Train/Test Split
↓
Train 4 CNN Models (VGG16, ResNet50, MobileNetV2, Xception)
↓
Select Best Model (92% Accuracy)
↓
Live Webcam Feed → Model Inference
↓
Fire Detected? → Gmail Alert + Buzzer

---

## 📊 Model Comparison

| Model | Accuracy | Loss |
|-------|----------|------|
| VGG16 | fill yours | fill yours |
| ResNet50 | fill yours | fill yours |
| MobileNetV2 | fill yours | fill yours |
| Xception | fill yours | fill yours |
| ✅ **Best Model** | **92%** | - |

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/fire-detection-system.git
cd fire-detection-system

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install frontend dependencies
cd client
npm install

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your Gmail and MongoDB credentials

# 5. Run backend
python app.py

# 6. Run frontend (new terminal)
cd client
npm start
```

---

## 🔧 Environment Variables

Create a `.env` file in root directory:

```env
GMAIL_USER=your_email@gmail.com
GMAIL_PASSWORD=your_app_password
MONGO_URI=your_mongodb_connection_string
```

---

## 📁 Project Structure



<img width="1591" height="714" alt="image" src="https://github.com/user-attachments/assets/1fce2210-0c96-406c-8ddd-89ef735aa540" />

