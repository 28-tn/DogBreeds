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
    if isHuman:
        message = 'I think this is a human.'
    else:
        message = 'I do not think this is a human.'
    return render_template('index.html', message=message, img_path=img_path)
