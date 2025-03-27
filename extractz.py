def process_pdf(pdf_path, output_folder, output_json):
    import os
    import re
    import cv2
    import numpy as np
    import json
    import matplotlib.pyplot as plt
    from pdf2image import convert_from_path
    from paddleocr import PaddleOCR

    print("[OCR] Initialisation de PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', det=True, rec_algorithm='CRNN')

    print("[OCR] Configuration de Poppler...")
    import shutil
    os.environ["PATH"] += os.pathsep + r"H:\poppler-24.08.0\Library\bin"
    print("[Poppler] PATH configuré :", os.environ["PATH"])
    print("[Check] pdftoppm trouvé :", shutil.which("pdftoppm"))
    print("[OCR] Initialisation PaddleOCR...")
    ocr = PaddleOCR(use_angle_cls=True, lang='en', det=True, rec_algorithm='CRNN')

    print("[OCR] Lecture du fichier PDF :", pdf_path)
    if not os.path.exists(pdf_path):
        print("Le fichier n'existe pas !")
    images = convert_from_path(pdf_path, dpi=300)
    

    def merge_contours(contours, threshold=50):
        if not contours: return []
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
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.equalizeHist(img_gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel_h = np.ones((1, 50), np.uint8)
        kernel_v = np.ones((50, 1), np.uint8)
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
        table_mask = cv2.add(horizontal_lines, vertical_lines)
        morph = cv2.dilate(table_mask, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 200 and cv2.boundingRect(c)[3] > 100]
        merged = merge_contours(detected, threshold=30)
        return [(max(x-10, 0), max(y-10, 0), w+20, h+20) for (x, y, w, h) in merged]

    def extract_text_from_roi(roi):
        results = ocr.ocr(roi, cls=True)
        return [[entry[1][0] for entry in line] for line in results]

    def clean_and_format_cell(cell):
        if not cell or cell in {"1", "(1", "(!)"}: return None
        cell = cell.replace(" ", "")
        if re.match(r'^\d{1,3}\.\d$', cell) or re.match(r'^\d{2,3}\.\d{1,3}$', cell):
            return float(cell)
        if re.match(r'^\d{2,3}$', cell):
            return float(cell[:-1] + '.' + cell[-1])
        if re.match(r'^\d{4}$', cell):
            return cell
        return cell

    def format_table_data(data):
        return [[clean_and_format_cell(cell) for cell in row] for row in data]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("[OCR] Lecture du PDF...")
    images = convert_from_path(pdf_path, dpi=300)
    structured_tables = {}

    for i, img in enumerate(images):
        print(f"[OCR] Traitement de la page {i+1}...")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        tables = detect_tables(img_cv)
        page_data = {}
        for j, (x, y, w, h) in enumerate(tables):
            print(f"  -> Extraction du tableau {j+1}")
            roi = img_cv[y:y+h, x:x+w]
            img_name = f"extracted_table_{j+1}.png"
            cv2.imwrite(os.path.join(output_folder, img_name), roi)
            ocr_data = extract_text_from_roi(roi)
            page_data[f"Tableau {j+1}"] = format_table_data(ocr_data)
        structured_tables[f"Page {i+1}"] = page_data

    print("[OCR] Sauvegarde du fichier JSON...")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(structured_tables, f, ensure_ascii=False, indent=4)

    print("[OCR] Extraction terminée avec succès.")
    return structured_tables
