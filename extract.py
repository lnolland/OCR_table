import cv2
import pdfplumber
import numpy as np
import json
import os
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\\Program Files\\Tesseract-OCR\\tessdata\\"

def extract_tables_from_pdf(pdf_path, output_folder):
    """ Extrait les tables d'un PDF et applique l'OCR en structurant les données """
    os.makedirs(output_folder, exist_ok=True)
    extracted_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            img_path = os.path.join(output_folder, f"page_{i+1}.png")
            pix = page.to_image(resolution=300)
            pix.save(img_path, format="PNG")
            
            # Charger l'image avec OpenCV
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Appliquer un seuillage adaptatif
            thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 15, 4)
            
            # Détecter les contours des tables
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            tables_detected = []
            
            for j, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                if w > 100 and h > 50:  # Filtrer les petits bruits
                    table_img = img[y:y+h, x:x+w]
                    table_path = os.path.join(output_folder, f"table_{i+1}_{j+1}.png")
                    cv2.imwrite(table_path, table_img)
                    
                    # OCR avec Tesseract en mode table
                    custom_config = '--psm 6 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-"'
                    text = pytesseract.image_to_string(table_img, config=custom_config)
                    
                    # Convertir le texte en tableau structuré
                    rows = text.split("\n")
                    structured_table = [row.split() for row in rows if row.strip()]
                    
                    tables_detected.append({"page": i+1, "table": j+1, "content": structured_table})
    
    # Sauvegarde en JSON
    with open(os.path.join(output_folder, "extracted_tables.json"), "w", encoding="utf-8") as f:
        json.dump(tables_detected, f, indent=4, ensure_ascii=False)
    
    print("Extraction terminée. Données sauvegardées dans extracted_tables.json")

# Exemple d'utilisation
extract_tables_from_pdf("test.pdf", "output")
