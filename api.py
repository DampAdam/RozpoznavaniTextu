from flask import Flask, request, jsonify
from inference_sdk import InferenceHTTPClient
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import cv2
import pytesseract as pt
import inference

clr = lambda:os.system('cls')

clr()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="5ILgTDnuiaokCkMBc5Fs"
)

model = inference.get_model("dpdtextrecognition/3")


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
CORS(app, origins="*")

@app.route('/getScreenText', methods=['POST'])
def getScreenTExt():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowedFile(file.filename):
        print("yes")
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        predictions = CLIENT.infer(filepath, model_id="dpdtextrecognition/3")['predictions']
        if len(predictions) == 0:
            return jsonify({"text": ""})
        x, y, width, height, conf = int(predictions[0]['x']), int(predictions[0]['y']), int(predictions[0]['width']), int(predictions[0]['height']), predictions[0]['confidence']
        x0 = int(x - width/2)
        y0 = int(y - height/2)
        x1 = x0+width
        y1 = y0+height
        text = pt.image_to_string(cv2.imread(filepath)[y0:y1, x0:x1], config='--psm 6 --oem 3')
        return jsonify({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text, "confidence": conf})
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route("/test")
def test():
    return jsonify({"message": "Hello World!"})

def allowedFile(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)