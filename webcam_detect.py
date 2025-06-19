import cv2
import os
import base64
import time
import uuid
import pygame
import tempfile
import pyttsx3
import json

import numpy as np

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from gtts import gTTS
from ultralytics import YOLO

import firebase_admin
import pyrebase
from firebase_admin import credentials, firestore, initialize_app
from firebase_admin import auth

# ============================ #
# 🔧 Inisialisasi Komponen
# ============================ #

# Inisialisasi Firebase
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Firebase Auth
config = {
    "apiKey": "AIzaSyB5rmFV7lT7xn89s3E-0lkJTQEQr8bXTEg",
    "authDomain": "rambu-id.firebaseapp.com",
    "databaseURL": "https://rambu-id-default-rtdb.asia-southeast1.firebasedatabase.app",
    "storageBucket": "rambu-id.appspot.com"
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()

# Anonymous login
user = auth.sign_in_anonymous()

# Load mapping label → kategori dari file JSON eksternal
with open('label_kategori.json', 'r', encoding='utf-8') as f:
    label_to_category = json.load(f)

# Ambil waktu sekarang di zona Asia/Jakarta
waktu_jakarta = datetime.now(ZoneInfo("Asia/Jakarta"))

# Load model YOLOv8
model = YOLO('rambuid.pt')

# Folder ikon
icons_dir = 'icons'

# Text-to-Speech
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Kecepatan bicara

def speak_label(label):
    sentence = f"Terdeteksi rambu {label}"
    tts = gTTS(text=sentence, lang='id')
    
    # Buat file temporer
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        temp_mp3 = fp.name

    # Beri waktu file ditulis
    time.sleep(0.2)
    
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(temp_mp3)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)  # tunggu sampai selesai

        pygame.mixer.music.unload()  # pastikan pygame selesai pakai file    

    finally:
        if os.path.exists(temp_mp3):
            try:
                os.remove(temp_mp3)
            except PermissionError:
                print(f"[Warning] Tidak bisa hapus {temp_mp3}, masih dipakai.")

# Buka webcam
cap = cv2.VideoCapture(0)

# Variabel kontrol pengiriman
last_sent_time = 0
send_interval = 5  # dalam detik
last_label_sent = None

# ============================ #
# 📤 Kirim Deteksi ke Firestore
# ============================ #

def is_valid_label(label):
    if not isinstance(label, str):
        return False
    label = label.strip()
    if len(label) == 0 or len(label) > 100:
        return False
    return True

def is_valid_frame(frame):
    return frame is not None and isinstance(frame, (np.ndarray,))

def send_detection_to_firestore(label, kategori, x1, y1, x2, y2, frame):
    user_id = auth.current_user
    # user_id = "uuidDitulisDisini"  #ID user Hardcoded
    # user_id = user['localId']      #ID user Lokal
    label = label.strip().lower()

    if not is_valid_label(label):
        print(f"Label tidak valid: '{label}', tidak dikirim.")
        return 
    
    if not is_valid_frame(frame):
        print("Frame tidak valid.")
        return

     # Format waktu zona   
    timestamp = datetime.now(ZoneInfo("Asia/Jakarta")).isoformat()
    formatted_time = waktu_jakarta.strftime("%d-%m-%Y %H:%M:%S") + " WIB"

    # Encode frame ke base64
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')

    # Pastikan size gambar tidak berlebihan
    if len(jpg_as_text) > 800000:
        print("Gambar terlalu besar untuk Firestore.")
        return

    data = {
        "userId": user_id,
        "label": label,
        "kategori": kategori,
        "timestamp": timestamp,
        "tanggal": formatted_time,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "image_base64": jpg_as_text
    }

    try:
        doc_id = str(uuid.uuid4())
        db.collection("detections").document(doc_id).set(data)
        print(f"[Firebase] Dikirim: {label} @ {timestamp}")
    except Exception as e:
        print(f"[Error Firebase] {e}")

# Teks
def draw_wrapped_text_with_background(img, text, origin, font, scale, text_color, thickness, max_width, bg_color=(0, 0, 0), alpha=0.5):
    """
    Gambar teks terbungkus dengan latar belakang semi-transparan.
    - origin: posisi awal (x, y)
    - bg_color: warna latar belakang (B, G, R)
    - alpha: transparansi latar (0.0 - 1.0)
    """
    words = text.split()
    lines = []
    current_line = ''

    # Bungkus teks
    for word in words:
        test_line = f"{current_line} {word}".strip()
        (w, h), _ = cv2.getTextSize(test_line, font, scale, thickness)
        if w <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)

    x, y = origin
    line_height = h + 10  # spasi antar baris
    total_height = line_height * len(lines)
    max_line_width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines)

    # Buat latar belakang semi-transparan
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 5, y - h - 5), (x + max_line_width + 5, y + total_height - h + 5), bg_color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Gambar teks
    for line in lines:
        cv2.putText(img, line, (x, y), font, scale, text_color, thickness)
        y += line_height

# ============================ #
# 🎯 Loop Deteksi Utama
# ============================ #

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membuka webcam.")
        break

    # Prediksi menggunakan model
    results = model.predict(source=frame, conf=0.5, save=False, imgsz=320, device='cpu')
    # results = model.predict(source=frame, conf=0.5, save=False, imgsz=416)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        last_sent_label = None

        for box, cls_id in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(cls_id)]

            # Ambil Kategori
            kategori = label_to_category.get(label.lower().replace(" ", "-"), "Tidak Diketahui")
            if kategori == "Tidak Diketahui":
                print(f"[Warning] Kategori tidak ditemukan untuk label: {label}")

            # Ambil waktu sekarang
            current_time = time.time()

            # --- Normalisasi label ---
            label_norm = label.lower().replace(' ', '-').strip()

            # Path ikon
            icon_path = os.path.join(icons_dir, f"{label_norm}.png")

            # Gambar bounding box dan label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Tambahkan kategori di bawah label
            cv2.putText(frame, f"Kategori: {kategori}", (x1, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # Tampilkan ikon jika ada
            if os.path.exists(icon_path):
                icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
                if icon is not None:
                    icon = cv2.resize(icon, (100, 100))

                    # Posisi ikon
                    x_offset, y_offset = 10, 10
                    y1_icon, y2_icon = y_offset, y_offset + icon.shape[0]
                    x1_icon, x2_icon = x_offset, x_offset + icon.shape[1]

                    # Cek frame cukup besar
                    if y2_icon <= frame.shape[0] and x2_icon <= frame.shape[1]:
                        if icon.shape[0] > frame.shape[0] or icon.shape[1] > frame.shape[1]:
                            print("Ukuran ikon lebih besar dari frame, dilewati.")
                            continue
                        if icon.shape[2] == 4:
                            alpha_s = icon[:, :, 3] / 255.0
                            alpha_l = 1.0 - alpha_s
                            for c in range(0, 3):
                                frame[y1_icon:y2_icon, x1_icon:x2_icon, c] = (
                                    alpha_s * icon[:, :, c] +
                                    alpha_l * frame[y1_icon:y2_icon, x1_icon:x2_icon, c]
                                )
                        else:
                            frame[y1_icon:y2_icon, x1_icon:x2_icon] = icon

                        # Tulis teks di bawah ikon
                        # Tentukan posisi teks agar tidak keluar dari frame
                        text = f"Ini adalah rambu {label}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2
                        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

                        # Coba tampilkan teks di bawah ikon, jika muat
                        if y2_icon + 30 + text_height < frame.shape[0]:
                            text_y = y2_icon + 30
                        else:
                            # Kalau tidak muat, tampilkan di atas ikon
                            text_y = y1_icon - 10
                            if text_y - text_height < 0:
                                text_y = y1_icon  # fallback ke posisi ikon jika terlalu atas

                        # Periksa jika teks terlalu lebar untuk frame
                        max_width = frame.shape[1] - 20  # beri margin 10 px kiri-kanan
                        draw_wrapped_text_with_background(
                            frame,
                            text,
                            origin=(10, text_y),
                            font=cv2.FONT_HERSHEY_SIMPLEX,
                            scale=0.7,
                            text_color=(0, 255, 255),
                            thickness=2,
                            max_width=frame.shape[1] - 20,
                            bg_color=(0, 0, 0),        # Hitam
                            alpha=0.6                 # Transparansi
                        )                        
                        if text_width > max_width:
                            while text_width > max_width and len(text) > 3:
                                text = text[:-4] + "..."  # potong dan beri elipsis
                                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)                       
                    else:
                        print("Frame terlalu kecil untuk menempelkan ikon.")
                else:
                    print(f"Gagal load ikon: {icon_path}")
            else:
                print(f"Tidak ditemukan ikon: {icon_path}")

            print(f"[DEBUG] Label hasil deteksi: '{label}'")

            # Kirim ke Firestore hanya jika label berubah atau waktu sudah lewat dan ucapkan jika perlu
            if label != last_label_sent or (current_time - last_sent_time) >= send_interval:
                # Salin frame untuk dikirim (agar tidak konflik dengan anotasi yang belum selesai)
                frame_to_send = frame.copy()
                # Tambahkan anotasi ke frame_to_send
                cv2.rectangle(frame_to_send, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_to_send, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_to_send, f"Kategori: {kategori}", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)  
                formatted_time = waktu_jakarta.strftime("%d-%m-%Y %H:%M:%S") + " WIB"             
                speak_label(label)                
                send_detection_to_firestore(label, kategori, x1, y1, x2, y2, frame)
                label = label.strip().lower()
                print(f"🔧 Mengirim ke Firestore: label={label}, kategori={kategori}, coords=({x1},{y1},{x2},{y2})")
                last_label_sent = label
                last_sent_time = current_time
                print(f"Dikirim ke Firebase: {label} @ {datetime.now().isoformat()}")

    # Tampilkan hasil
    cv2.imshow('Webcam Deteksi Rambu', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================ #
# 🚪 Bersih-bersih
# ============================ #

cap.release()
cv2.destroyAllWindows()
