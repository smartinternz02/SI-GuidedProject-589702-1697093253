
# # import os
# # import base64
# # from flask import Flask, render_template, request

# # app = Flask(__name__)

# # # Define a function to generate captions (replace this with your actual caption generation logic)
# # def generate_captions(image_path):
# #     captions = ["Caption 1", "Caption 2", "Caption 3"]
# #     return captions

# # # Default home page or route
# # @app.route('/')
# # def home():
# #     return render_template('/index.html')

# # @app.route('/Prediction')
# # def Prediction():
# #     return render_template('/caption.html')

# # @app.route('/PredictionCaption', methods=["POST"])
# # def upload():
# #     if request.method == "POST":
# #         file = request.files['image']
# #         if file:
# #             basepath = os.path.dirname(__file__)
# #             filepath = os.path.join(basepath, 'uploads', file.filename)
# #             file.save(filepath)
# #             captions = generate_captions(filepath)

# #     with open(filepath, 'rb') as uploadedfile:
# #         image_base64 = base64.b64encode(uploadedfile.read()).decode()

# #     return render_template('caption.html', prediction=captions, image=image_base64)

# # if __name__ == '__main__':
# #     app.run(debug=True, port=11000)




# # from flask import Flask, render_template, url_for, request
# # import os
# # import base64
# # import numpy as np
# # from tensorflow.keras.models import Model
# # from tensorflow.keras.preprocessing import image
# # from tensorflow.keras.preprocessing.image import load_img, img_to_array
# # from tensorflow.python.ops.gen_array_ops import concat
# # from tensorflow.keras.models import load_model 
# # from tensorflow.keras.preprocessing.text import Tokenizer
# # from tensorflow.keras.preprocessing.sequence import pad_sequences
# # from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# # #Loading the model
# # model = load_model("best_model.h5", compile=False)
# # app = Flask(__name__)
# # # default home page or route
# # def generate_captions(image_path):
# #     # Load and preprocess the image
# #     img = load_img(image_path, target_size=(224, 224))
# #     img = img_to_array(img)
# #     img = preprocess_input(img)
# #     img = np.expand_dims(img, axis=0)

# #     # Use your model to generate captions
# #     captions = []  # Store generated captions here

# #     # Example code for generating captions (replace with your actual model)
# #     for i in range(5):  # Generate up to 5 captions
# #         caption = f"Generated caption {i + 1}"  # Replace with your caption generation logic
# #         captions.append(caption)

# #     return captions
# # @app.route('/')
# # def home():
# #     return render_template("index.html")

# # @app.route('/Prediction')
# # def Prediction():
# #     return render_template('caption.html')

# # @app.route('/PredictCaption', methods=["GET","POST"])
# # def upload():
# #     if request.method == "POST":
# #         file  = request.files['image']
# #         basepath = os.path.dirname(__file__)
# #         print("current path", basepath)
# #         filepath = os.path.join(basepath, 'uploads', file.filename)
# #         print("upload folder is", filepath)
# #         file.save(filepath)
# #         captions = generate_captions(filepath)
# #     with open(filepath, 'rb') as uploadedfile:
# #         img_base64 = base64.b64encode(uploadedfile.read()).decode()
# #     return render_template('caption.html', prediction=str(captions), image=img_base64)

# # if __name__ == '__main__':
# #     app.run(debug=True, port= 1100)
# from pickle import load
# from numpy import argmax
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.models import Model 
# from tensorflow.keras.models import load_model
# import os 
# from flask import Flask, render_template, request
# from werkzeug.utils import secure_filename 
# from gevent.pywsgi import WSGIServer


# app = Flask (__name__)
# @app.route('/')
# def home():
#     return render_template("index.html")

# @app.route('/Prediction')
# def Prediction():
#     return render_template('caption.html')

# @app.route('/PredictCaption',methods = ['GET', 'POST']) 
# def upload():
#     if request.method == 'POST':
#         f = request.files['image']
#         print("current path")
#         basepath= os.path.dirname(__file__) 
#         print("current path", basepath)
#         filepath = os.path.join(basepath,"uploads",f.filename)
#         print("upload folder is", filepath)
#         f.save(filepath)
#         text = modelpredict(filepath)
#         return text
# def extract_features (filename):
#     print('features extracted')
#     model= VGG16()
#     model.layers.pop()
#     model = Model(inputs = model.inputs, outputs = model.layers [-1].output)
#     image = load_img(filename, target_size=(224, 224))
#     print('image loaded')
#     image = img_to_array(image)
#     print (image)
#     image = image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))
#     image = preprocess_input(image)
#     feature = model.predict(image, verbose=0)
#     print('model predicted')
#     return feature

# def word_for_id(integer, tokenizer):
#     for word,index in tokenizer.word_index.items():
#         if index == integer:
#             return word
#     return 
# def generate_desc (model, tokenizer, photo, max_length): 
#     print("generate description")
#     in_text= 'startseq'
#     for i in range (max_length):
#         sequence = tokenizer.texts_to_sequences ([in_text])[0]
#         sequence = pad_sequences ([sequence], maxlen=max_length)
#         print('sequence')
#         yhat = model.predict ( [photo, sequence], verbose=0)
#         yhat= argmax(yhat)
#         word = word_for_id(yhat, tokenizer)
#         if word is None:
#             break
#         in_text += ' ' + word
#         if word == 'endseq':
#             break
#         print(in_text)
#         return in_text

# def modelpredict(filepath):
#     tokenizer = load(open(r"C:\\Users\\Pavan\\Downloads\\ImageCaptionGeneration\\tokenizer.pkl",'rb'))
#     max_length = 34
#     model = load_model(r'C:\\Users\\Pavan\\Downloads\\ImageCaptionGeneration\\caption.h5')
#     print('model loaded')
#     photo = extract_features (filepath)
#     description = generate_desc (model, tokenizer, photo, max_length)
#     return description

        

        
# if __name__=="__main__":
#     app.run(debug = False)

from pickle import load
from numpy import argmax
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/Prediction')
def Prediction():
    return render_template('caption.html')

@app.route('/PredictCaption', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, "uploads", secure_filename(f.filename))
        f.save(filepath)
        text = modelpredict(filepath)
        return text

def extract_features(filename):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        # in_text += ' ' + word
        if word == 'endseq':
            break
        in_text += ' ' + word
        # Remove 'startseq' from the beginning and 'endseq' from the end
    in_text = in_text.replace('startseq', '').replace('endseq', '').strip()
    return in_text

def modelpredict(filepath):
    tokenizer = load(open("tokenizer.pkl", 'rb'))
    max_length = 34
    model = load_model("caption.h5")
    photo = extract_features(filepath)
    description = generate_desc(model, tokenizer, photo, max_length)
    return description

if __name__ == "__main__":
    app.run(debug=True)

