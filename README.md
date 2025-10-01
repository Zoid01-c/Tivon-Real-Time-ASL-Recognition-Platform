# Tivon – Real-Time ASL Recognition Web App

Tivon is a **real-time American Sign Language (ASL) recognition platform** that helps users learn and practice ASL using just a webcam. Powered by **deep learning** and **computer vision**, Tivon runs entirely in your browser with a focus on **privacy, accessibility, and user engagement**.

---

## ✨ Key Features

* **Real-Time Recognition:** Detect ASL signs instantly via webcam
* **Interactive Learning:** Explore the ASL alphabet with visual guides
* **Practice Tools:** Exercises and quizzes to improve signing skills
* **Progress Tracking:** Analytics to monitor your learning journey
* **Responsive Design:** Works on desktop, tablet, and mobile
* **Privacy First:** All recognition happens locally; no data leaves your device

---

## ℹ️ Why “Tivon”?

* **Pronunciation:** “Tih-von”
* **Breakdown:** Combines **“talk” (ti)** and **“vocal” (von)**, symbolizing communication through signs
* **Why it fits:** Simple, friendly, and approachable with a modern tech vibe
* **Uniqueness:** “Tivon” is an invented word with no existing online footprint in this context

---

## 🖼 Screenshot Gallery

Click any thumbnail to view the full screenshot:

<table>
<tr>
<td align="center">
<a href="ScreenShot/Home.png"><img src="ScreenShot/Home.png" width="150" alt="Home Screen"></a><br>Home Screen
</td>
<td align="center">
<a href="ScreenShot/Identifying screen.png"><img src="ScreenShot/Identifying screen.png" width="150" alt="Recognition Screen"></a><br>Recognition Screen
</td>
<td align="center">
<a href="ScreenShot/About Asl.png"><img src="ScreenShot/About Asl.png" width="150" alt="About ASL"></a><br>About ASL
</td>
</tr>
<tr>
<td align="center">
<a href="ScreenShot/About product.png"><img src="ScreenShot/About product.png" width="150" alt="About Product"></a><br>About Product
</td>
<td align="center">
<a href="ScreenShot/Helpful link.png"><img src="ScreenShot/Helpful link.png" width="150" alt="Helpful Links"></a><br>Helpful Links
</td>
<td></td>
</tr>
</table>

> All thumbnails are clickable for a larger view, creating an interactive gallery effect.

---

## 🎥 Demo Video  

You can watch the demo here:  


---

https://github.com/user-attachments/assets/7875c605-e7b8-417b-96d8-bb8e7a6e119e



## ⚙️ How It Works

Tivon combines:

* **MediaPipe:** For accurate hand tracking
* **Custom Deep Learning Model:** To classify ASL signs (must be provided locally)
* **Flask Backend & WebSockets:** For real-time communication
* **Frontend:** Interactive HTML, CSS, and JavaScript interface

**⚠️ Note:** The ASL model is **not included** in this repository. You must provide your **own trained model**. All recognition is performed **locally** to ensure maximum privacy.

---

## 🛠 Technologies

* **Frontend:** HTML5, CSS3, JavaScript, Bootstrap
* **Backend:** Python, Flask, OpenCV, MediaPipe
* **Machine Learning:** TensorFlow, Keras
* **Database:** SQLite

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* Webcam
* Modern browser with JavaScript enabled
* Trained ASL model (local file)

### Installation

```bash
git clone https://github.com/yourusername/tivon.git
cd tivon
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place your **trained ASL model** in the `aslsavedmodelfinetunedv3/` folder.

### Running the App

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📬 Contact

Created by **[Khan Rehan Jakir]** – reach out with questions or suggestions!
