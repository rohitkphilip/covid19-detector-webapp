import os
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
#from gevent.pywsgi
from flask import Flask, redirect, request, render_template

# import tensorflow.keras.preprocessing.image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model

from config import MODEL_PATH

app = Flask(__name__)

trainedModel = load_model(MODEL_PATH)
print(trainedModel.summary())

def make_predictions(image_path, model):

    # input_image = image.load_img(image_path, target_size = (64, 64))
    input_image = image.load_img(image_path, target_size = (224, 224))
    input_image = image.img_to_array(input_image)
    input_image = np.expand_dims(input_image, axis = 0)

    output = model.predict(input_image)

    return output


@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        print("image uploaded")
        input_file = request.files['file']

        #save the file
        basepath = os.path.dirname(__file__)
        input_file_path = os.path.join(basepath, secure_filename(input_file.filename))
        input_file.save(input_file_path)

        output = make_predictions(input_file_path, trainedModel)

        print("PREDICTION")
        print(output[0][0])
        print(output[0][1])

        if output[0][0] > 0.4:
            test_result = "COVID"
        else:
            test_result = "Normal"

        os.remove(input_file_path)

        return test_result
    
    return None


if __name__ == '__main__':
    app.run(debug = True)




