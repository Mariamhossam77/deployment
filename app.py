
from flask import Flask, render_template, request
from ultralytics import YOLO
from PIL import Image
import os

app = Flask(__name__)
def predict_disease(image_path):
    model = YOLO('./runs/classify/train7/weights/best.pt')
    with Image.open(image_path) as img:
        img = img.resize((255, 255))
    results = model(img, show=True)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    prediction = names_dict[probs.index(max(probs))]
    return f'We regret to inform you that you have {prediction} disease '


@app.route('/')
def home():
    return render_template('index.html', prediction=None)


@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']
    image_path = 'temp_image.jpg'
    file.save(image_path)
    prediction = predict_disease(image_path)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)