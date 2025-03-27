from flask import Flask, render_template, request
from extractz import process_pdf
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_script():
    print("Le bouton a bien été cliqué !")
    try:
        if 'pdf_file' not in request.files:
            return "Aucun fichier PDF fourni", 400

        pdf = request.files['pdf_file']
        if pdf.filename == '':
            return "Nom de fichier invalide", 400

        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf.filename)
        pdf.save(pdf_path)

        print("\u27a1\ufe0f Lancement de process_pdf() avec:", pdf_path)
        process_pdf(pdf_path, "output_images", "structured_extracted_data.json")
        print("Traitement terminé")
        return "Traitement terminé avec succès !"
    except Exception as e:
        print("Une erreur est survenue :", e)
        return f"Erreur : {str(e)}", 500

if __name__ == '__main__':
    print("Serveur en cours de démarrage...")
    app.run(debug=True)
