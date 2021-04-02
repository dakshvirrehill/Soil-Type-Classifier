from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

def init():
    global model,model_classes
    model = load_model('model/trainedModel.h5')
    model_classes = ['Alluvial, Black, Red, Clay']

def preprocess_image(img):
    #add preprocessing of image from model notebook
    return img

app = Flask(__name__, template_folder='templates')

@app.route('/SoilClassifier')
def startApp():
   return render_template('index.html')

@app.route('/getResult',methods=['POST'])
def getResult():
    if request.method == 'POST':
        soil_image = Image.open(request.files['image'].stream).convert('L')
        soil_image = preprocess_image(soil_image)
        result = {"result":model_classes[np.argmax(model.predict([soil_image])),"status":200]}
        return result
    return {"status":500}



if __name__ == '__main__':
    init()
    app.run(host='0.0.0.0', port=4567)