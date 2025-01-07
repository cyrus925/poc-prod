import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from flask import Flask, request, render_template_string
import logging

from predict.predict.run import TextPredictionModel

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# HTML 
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial; margin: 40px; }
        input[type=text] { width: 300px; padding: 5px; }
        input[type=submit] { padding: 5px 15px; }
        .result { margin-top: 20px; }
    </style>
</head>
<body>
    <h2>Entrez votre texte pour prédire les tags</h2>
    <form method="POST">
        <input type="text" name="text" placeholder="Votre texte ici...">
        <input type="submit" value="Prédire">
    </form>
    {% if predictions %}
    <div class="result">
        <h3>Prédictions:</h3>
        <p>{{ predictions }}</p>
    </div>
    {% endif %}
</body>
</html>
'''

# Variable globale pour stocker le modèle
model = None

def load_model():
    """Charge le modèle"""
    global model
    try:
        model_path = "../../train/models/2024-12-30-01-22-38"  # A ajuster selon le modèle qu'on veut
        model = TextPredictionModel.from_artefacts(model_path)
        logger.info("Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {str(e)}")
        raise RuntimeError("Impossible de charger le modèle")

@app.route('/', methods=['GET', 'POST'])
def home():
    predictions = None
    if request.method == 'POST':
        text = request.form['text']
        if model is None:
            return "Erreur: Modèle non chargé"
        try:
            predictions = model.predict([text])[0]
        except Exception as e:
            return f"Erreur lors de la prédiction : {str(e)}"
    
    return render_template_string(HTML, predictions=predictions)

if __name__ == "__main__":
    load_model()  # Charger le modèle au démarrage
    app.run(host='0.0.0.0', port=8001, debug=True) 