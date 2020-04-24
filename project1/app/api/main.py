import sys
sys.append('~/Development/uvadev/nlp2/project1')
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS, cross_origin
from models.make_vae import make_vae
from app_config import config

model_paths = {
    'vae_vanilla': 'vae_best_mu0_wd1_fb-1.pt',
    'vae_high_freebits': 'vae_best_mu5_wd1_fb8.pt',
    'vae_md_freebits':'vae_best_mu5_wd1_fv4.pt'
}

loaded_models = { key: make_vae(config, True, val) for key, val in model_paths.items() }

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)

@app.route('', methods=['GET'])
def init():
    return loaded_models.keys()

@app.route('/predict', methods=['POST'])
def predict():
    # Validate the request body contains JSON
    if request.is_json:

        # Parse the JSON into a Python dictionary
        req = request.get_json()

        # Print the dictionary
        print(req)

        # Return a string along with an HTTP status code
        # Replace with prediction
        return jsonify({'prediction': "Lorem Ipsum Sic Dolor Amet!"}), 200

    return "Not JSON", 400

if __name__ == '__main__':
    app.run(debug=True)
