import cv2
import numpy as np
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import os
import pytesseract
import pandas as pd
import json
import re

# Configuration de Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# Configuration de Poppler
os.environ["PATH"] += os.pathsep + r"H:\\Desktop\\projet_semestre\\NotreProjet\\poppler\\poppler-24.08.0\\Library\\bin"

def merge_contours(contours, threshold=50):
    """Fusionne les contours qui se chevauchent ou sont proches."""
    if not contours:
        return []
    
    merged = []
    for cnt in contours:
        x, y, w, h = cnt
        merged_flag = False
        
        for i, (mx, my, mw, mh) in enumerate(merged):
            if (x < mx + mw + threshold and x + w > mx - threshold and
                y < my + mh + threshold and y + h > my - threshold):
                new_x = min(x, mx)
                new_y = min(y, my)
                new_w = max(x + w, mx + mw) - new_x
                new_h = max(y + h, my + mh) - new_y
                merged[i] = (new_x, new_y, new_w, new_h)
                merged_flag = True
                break
        
        if not merged_flag:
            merged.append(cnt)
    
    return merged

def detect_tables(image):
    """Détecte les tables et fusionne les fragments détectés."""
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel_h = np.ones((1, 50), np.uint8)
    kernel_v = np.ones((50, 1), np.uint8)
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
    
    table_mask = cv2.add(horizontal_lines, vertical_lines)
    morph = cv2.dilate(table_mask, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_contours = [(cv2.boundingRect(contour)) for contour in contours if cv2.boundingRect(contour)[2] > 200 and cv2.boundingRect(contour)[3] > 100]
    merged_contours = merge_contours(detected_contours, threshold=30)
    
    return merged_contours

def extract_table_data(image, table_contours):
    """Extrait le texte des zones des tables détectées et structure en tableau."""
    extracted_data = []
    for (x, y, w, h) in table_contours:
        roi = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, config='--psm 6')
        lines = text.split("\n")
        structured_table = [re.sub(r"\s{2,}", "|", line.strip()).split("|") for line in lines if line.strip()]
        extracted_data.append(structured_table)
    return extracted_data

def process_pdf(pdf_path, output_folder, output_json):
    """Convertit le PDF en images, détecte les tables et extrait leurs données."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    images = convert_from_path(pdf_path)
    structured_tables = {}
    table_counter = 1
    
    for i, img in enumerate(images):
        img_path = os.path.join(output_folder, f"page_{i+1}.png")
        img.save(img_path, "PNG")
        img_cv = cv2.imread(img_path)
        tables = detect_tables(img_cv)
        extracted_data = extract_table_data(img_cv, tables)
        
        for table in extracted_data:
            if table:
                title = table[0][0] if table else f"Tableau {table_counter}"
                structured_tables[title] = table[1:]
                table_counter += 1
    
    # Sauvegarde des données dans un fichier JSON
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(structured_tables, json_file, ensure_ascii=False, indent=4)
    
    return structured_tables

# Exemple d'utilisation
pdf_path = "test.pdf"  # Remplace avec le chemin correct
output_folder = "output_images"
output_json = "structured_extracted_data.json"
data_extracted = process_pdf(pdf_path, output_folder, output_json)

# Affichage des résultats
for title, table in data_extracted.items():
    print(f"{title} :")
    for row in table:
        print(row)
