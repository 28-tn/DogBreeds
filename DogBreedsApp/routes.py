from DogBreedsApp import app
from flask import render_template, request
import classifier_functions as clf

@app.route('/')
@app.route('/index')
def index():
    return render_fun()

@app.route('/', methods=['POST'])
def form_post():
    text = request.form['text']
    img_path = text
    return render_fun(img_path)

def render_fun(img_path='https://upload.wikimedia.org/wikipedia/commons/0/03/Kurzhaardackel.jpg'):
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
