# 🚦 Real-Time Mobile Phone Usage Detection & Automated Ticketing (Raspberry Pi + YOLOv8)

This project detects **two-wheeler riders using mobile phones** in real time, extracts the **vehicle number plate using OCR**, and automatically sends a **violation email** with evidence.  
It also displays system status messages on an **LCD screen** connected to the Raspberry Pi.

---

## 📌 Features
- ✅ **YOLOv8** model trained for detecting mobile usage & number plates.  
- ✅ **Tesseract OCR** for number plate text extraction.  
- ✅ **Automated Email Alert** with violation details & image evidence.  
- ✅ **LCD Display Integration** to show real-time system messages.  
- ✅ **Raspberry Pi Deployment** with USB webcam support.  

---

## 🖼️ System Workflow
1. Capture live image/video using a USB camera.  
2. Run YOLOv8 detection on the frame.  
3. If a rider is detected **using a mobile phone**:
   - Extract number plate region.  
   - Process with **Tesseract OCR**.  
   - Display violation details on **LCD**.  
   - Send an **email alert** with attached evidence image.  
4. If no violation → display **No Violation** on LCD.  

---

## 🛠️ Hardware Requirements
- Raspberry Pi 4 (or 3B+)  
- USB Webcam  
- 16x2 LCD with I²C or GPIO interface  
- 12V Adapter / Power Supply  
- Connecting Wires & Breadboard  

---

## 💻 Software Requirements
- Python 3.9+  
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- OpenCV (`cv2`)  
- NumPy  
- PyTesseract  
- RPi.GPIO  
- RPLCD  
- smtplib (for email)  

---

## ⚡ Installation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
pip install ultralytics opencv-python numpy pillow pytesseract RPLCD RPi.GPIO

# Install Tesseract OCR
sudo apt install tesseract-ocr -y

📂 Project Structure
├── runs/detect/train5/weights/best.pt   # Trained YOLOv8 model
├── deep/
│   ├── 32.jpg                           # Sample input image
│   ├── detected_plate.jpg               # Cropped plate region
│   ├── processed_plate.jpg              # Preprocessed plate
│   ├── final_detected_image.jpg         # Output with bounding boxes
├── traffic_monitor.py                   # Main Python script
├── README.md                            # Project Documentation

▶️ Running the Project
python3 detection.py


LCD Display Messages:

"Traffic Monitor" → System booting

"System Ready" → Ready for detection

"No Violation" → No rider using mobile

"Violation Found" → Mobile usage detected

"Email Sent ✅" → Email sent successfully

📧 Email Alert

The system sends an email with:

Subject: "Traffic Violation Detected - <Number Plate>"

Body: "A violation has been detected. Number plate: <Number Plate>"

Attachment: Final detected image

🎥 Demo
video
images 

👨‍💻 Authors
VishnuVardhan Manjula
M. Vamsi Krishna
P. Venu Kumar
Dr. C. Kumar

📜 License

This project is for educational & research purposes.
Commercial use requires permission from the authors
