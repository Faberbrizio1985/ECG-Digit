import streamlit as st
import numpy as np
import cv2

st.set_page_config(page_title="Universal ECG Digitizer", layout="wide")
st.title("🩺 Universal ECG Digitizer AI")
st.write("Supporta: Foto su carta (Rosa/BN) e Screenshot da Monitor (Telemetria)")

uploaded_file = st.file_uploader("Carica Foto o Screenshot", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # --- LOGICA DI ELABORAZIONE ADATTIVA ---
    
    # 1. Analisi dello sfondo: è un monitor o carta?
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    is_dark_bg = np.mean(gray) < 127 # Se la media è bassa, è un monitor
    
    # 2. Pre-processing specifico
    if is_dark_bg:
        st.info("Rilevato sfondo scuro (Monitor/Telemetria)")
        # Esaltiamo i colori brillanti (verde/giallo della traccia)
        processed = cv2.bitwise_not(gray) # Invertiamo per rendere la traccia nera su bianco
    else:
        st.info("Rilevato sfondo chiaro (Carta millimetrata)")
        # Usiamo il canale della luminosità per ignorare il rosa della carta
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        processed = l # Il canale L ignora quasi del tutto il rosa/rosso

    # 3. Pulizia profonda (Filtro Bilaterale per preservare i bordi della traccia)
    denoised = cv2.bilateralFilter(processed, 9, 75, 75)
    
    # 4. Soglia Adattiva Intelligente
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 21, 15)

    # 5. Rimozione Griglia (Morfologia Lineare)
    # Rimuoviamo oggetti piccoli (i puntini della carta o rumore del monitor)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    final_trace = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # --- VISUALIZZAZIONE ---
    col1, col2 = st.columns(2)
    col1.image(img_rgb, caption="Input Originale")
    col2.image(final_trace, caption="Traccia Estratta")

    # 6. Conversione in Dati (Vettorializzazione)
    if st.button("Genera Dati Digitali"):
        # Scansioniamo l'immagine colonna per colonna
        height, width = final_trace.shape
        signal = []
        for x in range(width):
            pixels = np.where(final_trace[:, x] == 255)[0]
            if len(pixels) > 0:
                # Prendiamo il valore medio dei pixel neri (centro della traccia)
                signal.append(height - np.mean(pixels)) 
            else:
                signal.append(np.nan) # Punto mancante
        
        st.line_chart(signal)
        st.download_button("Scarica Dati (CSV)", "Time,Value\n" + "\n".join([f"{i},{v}" for i,v in enumerate(signal)]), "ecg_data.csv")
