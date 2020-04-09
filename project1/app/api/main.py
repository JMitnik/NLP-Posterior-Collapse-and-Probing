from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS, cross_origin

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)

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
