from DogBreedsApp import app
import json, plotly
from flask import render_template, request
import classifier_functions as clf

@app.route('/')
@app.route('/index')
def index():

    message = 'Hello World!'

    return render_template('index.html',
                           message = message)

@app.route('/', methods=['POST'])
def form_post():
    text = request.form['text']
    img_path = text
    isHuman = clf.face_detector(img_path)
    isDog = clf.dog_detector(img_path)
    if isDog:
        breed = clf.predict_breed(img_path)
        message = 'I think this is a {}'.format(breed)
    elif isHuman:
        breed = clf.predict_breed(img_path)
        message = 'I think this is a human, but it bears a certain resemblance with {}'.format(breed)
    else:
        message = 'This is neither a dog nor a human.'
    return render_template('index.html', message=message, img_path=img_path)
