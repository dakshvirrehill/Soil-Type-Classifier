from flask import Flask, render_template, request
import cv2
from PIL import Image
import numpy as np
from keras.models import load_model


def init():
    """Load Model and Start Classifier System"""
    global model
    model = load_model('model/trainedModel.h5')


def preprocess_image(img):
    """Preprocess uploaded image before sending for prediction"""
    img = np.array(img)
    img = img[:, :, ::-1].copy()
    img = cv2.resize(img, (32, 32))
    return np.expand_dims(img, axis=0)


app = Flask(__name__, template_folder='templates')


@app.route('/SoilClassifier')
def startApp():
    """Offer Front page of application to the user"""
    return render_template('index.html')


@app.route('/getResult', methods=['POST'])
def getResult():
    """Uses the uploaded image to predict soil type and return result"""
    if request.method == 'POST':
        soil_image = Image.open(request.files['image'].stream).convert('RGB')
        soil_image = preprocess_image(soil_image)
        result_val = np.argmax(model.predict([soil_image])).item()
        result = {"result": result_val, "status": 200}
        return result
    return {"status": 500}

if __name__ == '__main__':
    init()
    app.run(host='0.0.0.0',port=8080)
