import streamlit as st
import numpy as np
import cv2

st.set_page_config(page_title="Professional ECG Rectifier", layout="wide")
st.title("📏 ECG Rectifier & Standardizer AI")

uploaded_file = st.file_uploader("Carica la foto dell'ECG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # --- FASE 1: Rilevamento Contorni e Rettifica ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    rect_coords = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            rect_coords = approx
            break
            
    if rect_coords is not None:
        pts = rect_coords.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        (tl, tr, br, bl) = rect
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        dst_width = max(int(width_a), int(width_b))
        
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        dst_height = max(int(height_a), int(height_b))
        
        # Correzione qui: Matrice chiusa correttamente
        dst = np.array([
            [0, 0],
            [dst_width - 1, 0],
            [dst_width - 1, dst_height - 1],
            [0, dst_height - 1]], dtype="float32")
            
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (dst_width, dst_height))
        
        # --- FASE 2: Estrazione e Griglia a Puntini ---
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # Filtro adattivo per isolare il nero
        binary_trace = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 12)
        
        # Creazione carta millimetrata (Sfondo bianco, puntini neri)
        grid_paper = np.ones((dst_height, dst_width, 3), dtype="uint8") * 255
        dot_spacing = 15 
        for y in range(0, dst_height, dot_spacing):
            for x in range(0, dst_width, dot_spacing):
                cv2.circle(grid_paper, (x, y), 1, (180, 180, 180), -1) 

        # Sovrapposizione traccia nera
        grid_paper[binary_trace == 255] = [0, 0, 0] 
        
        col1, col2 = st.columns(2)
        col1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Originale")
        col2.
