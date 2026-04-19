import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Professional ECG Rectifier", layout="wide")
st.title("📏 ECG Rectifier & Standardizer AI")
st.write("Carica una foto e l'IA ritaglierà, raddrizzerà e ricostruirà l'ECG su carta millimetrata standard.")

# 1. Caricamento file
uploaded_file = st.file_uploader("Carica la foto dell'ECG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Conversione per OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    height, width = img.shape[:2]

    # --- FASE 1: Rilevamento Contorni e Rettifica Prospettica ---
    st.write("### 1. Rilevamento e Rettifica Prospettica")
    
    # Pre-processing per trovare i contorni del foglio
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    
    # Trova il contorno più grande (presumibilmente il foglio ECG)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    rect_coords = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4: # Abbiamo trovato un quadrilatero
            rect_coords = approx
            break
            
    if rect_coords is not None:
        # Ordina le coordinate: top-left, top-right, bottom-right, bottom-left
        pts = rect_coords.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        # Calcola le dimensioni del nuovo rettangolo piatto
        (tl, tr, br, bl) = rect
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        dst_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        dst_height = max(int(height_a), int(height_b))
        
        # Applica la trasformazione prospettica
        dst = np.array([
            [0, 0],
            [dst_width - 1, 0],
            [dst_width - 1, dst_height - 1],
            [0, dst_height -
