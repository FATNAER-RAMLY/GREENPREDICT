import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = r'C:\Users\pc\Desktop\model\static\UPLOAD_FOLDER'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# Charger le modèle et les noms de classe
MODEL_PATH = os.path.join('model', r'C:\Users\pc\Desktop\model\llll.h5')
model = load_model(MODEL_PATH)

with open(os.path.join('model', r'C:\Users\pc\Desktop\model\class_indices.json'), 'r') as f:
    class_names = json.load(f)

# Fonction pour vérifier les extensions de fichiers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route principale
@app.route('/', methods=['GET', 'POST'])
def index():
    results = None  # Initialiser results à None

    if request.method == 'POST':
        # Vérifier si un fichier a été téléchargé
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)  # Enregistrer le fichier dans le dossier uploads

            # Prétraitement de l'image
            img = Image.open(filepath).resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prédiction
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            predicted_class_name = class_names[str(predicted_class)]
            probabilities = predictions[0]

            # Préparer les résultats pour l'affichage
            results = {
                "filename": filename,
                "predicted_class": predicted_class_name,
                "probabilities": {class_names[str(i)]: float(probabilities[i]) for i in range(len(class_names))}
            }

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)