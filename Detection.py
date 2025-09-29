import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
from PIL import Image, ImageFont, ImageDraw
import os
import re
import smtplib
from email.message import EmailMessage

# ✅ Set Tesseract OCR path (Ensure Tesseract is installed on Raspberry Pi)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# ✅ Load YOLO Model
model = YOLO(r"/home/pi/deep/runs/detect/train5/weights/best.pt")

# ✅ Font paths
font_paths = ["/home/pi/deep/FE.TTF", "/home/pi/deep/Arial.ttf"]  # Added an alternative font

# ✅ Load Image
image_path = r"/home/pi/deep/32.jpg"
img = cv2.imread(image_path)
if img is None:
    print("Error: Image not found!")
    exit()

# ✅ Resize Image
img = cv2.resize(img, (979, 999))

# ✅ YOLO Detection
results = model.predict(img, conf=0.1, iou=0)

riders_with_mobile = []
number_plates = []

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0].item()
        class_id = int(box.cls[0])

        if confidence > 0.2:
            if class_id == 1:  # Mobile usage detection
                riders_with_mobile.append((x1, y1, x2, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, "Mobile in use", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            elif class_id == 2:  # Number plate detection
                number_plates.append((x1, y1, x2, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# ✅ Extract & Process Number Plate
for (rx1, ry1, rx2, ry2) in riders_with_mobile:
    closest_plate = None
    min_distance = float("inf")

    for (px1, py1, px2, py2) in number_plates:
        distance = abs(ry2 - py1)
        if distance < min_distance:
            min_distance = distance
            closest_plate = (px1, py1, px2, py2)

    if closest_plate:
        px1, py1, px2, py2 = closest_plate
        plate_region = img[py1:py2, px1:px2]

        # ✅ Save Plate Image
        plate_path = r"/home/pi/deep/detected_plate.jpg"
        cv2.imwrite(plate_path, plate_region)

        # ✅ Preprocessing for OCR
        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)  # Reduce noise while keeping edges sharp
        sharpened = cv2.addWeighted(gray, 1.5, filtered, -0.5, 0)  # Sharpening the image
        _, otsu_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ✅ Resize for better OCR accuracy
        resized = cv2.resize(otsu_thresh, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # ✅ Save Processed Image
        processed_path = r"/home/pi/deep/processed_plate.jpg"
        cv2.imwrite(processed_path, resized)

        # ✅ OCR Extraction using Tesseract
        custom_config = "--psm 7 --dpi 300 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = pytesseract.image_to_string(resized, config=custom_config)

        # ✅ Character Correction Mapping
        char_map = str.maketrans("OZ8SB", "0425B")
        corrected_text = text.translate(char_map).strip()

        # ✅ Validate Against Plate Format
        matches = re.findall(r'[A-Z]{2}\d{2}[A-Z]{2}\d{4}', corrected_text)
        extracted_text = matches[0] if matches else corrected_text

        print(f"✅ Extracted Plate using Tesseract: {extracted_text}")

        # ✅ Draw text with custom font
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        for font_path in font_paths:
            try:
                font = ImageFont.truetype(font_path, 60)  # Increased font size
                break
            except IOError:
                print(f"⚠️ Font not found: {font_path}")
                font = ImageFont.load_default()

        draw.text((px1, py1 - 80), extracted_text, (255, 255, 255), font=font)

        # ✅ Convert back to OpenCV format
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ✅ Save final image with bounding boxes
final_image_path = r"/home/pi/deep/final_detected_image.jpg"
cv2.imwrite(final_image_path, img)

# ✅ Send Email with Image
def send_email(image_path, extracted_text):
    sender_email = "embedded44@gmail.com"
    sender_password = "alqplnjlmgbyoxst"
    receiver_email = "manjulavishnu9052@gmail.com"

    subject = f"Traffic Violation Detected - {extracted_text}"
    body = f"A violation has been detected. Number plate: {extracted_text}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg.set_content(body)

    # Attach the image
    with open(image_path, "rb") as img_file:
        msg.add_attachment(img_file.read(), maintype="image", subtype="jpeg", filename="violation.jpg")

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("✅ Email Sent Successfully")
    except Exception as e:
        print(f"❌ Error Sending Email: {e}")

send_email(final_image_path, extracted_text) for this code add # === GPIO & LCD Setup ===

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

lcd = CharLCD(pin\_rs=25, pin\_e=24, pins\_data=\[23, 17, 18, 22], numbering\_mode=GPIO.BCM, cols=16, rows=2)

def lcd\_display\_message(message, delay=2):
"""Displays a message on the LCD and prints to the IDLE shell."""
lcd.clear()
print(f"📢 {message}")  # Display in IDLE Shell
if len(message) > 16:
lcd.write\_string(message\[:16])
lcd.cursor\_pos = (1, 0)
lcd.write\_string(message\[16:32])  # Second line
else:
lcd.write\_string(message)
time.sleep(delay) and when code run display message in lcd and idle shell lcd_display_message("Traffic Monitor", 3)
lcd_display_message("System Ready", 2)  lcd_display_message("No Violation", 2)lcd_display_message("Email Sent ✅", 3)  

