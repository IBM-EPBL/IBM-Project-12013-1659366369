
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from cloudant.client import Cloudant

# basepath = os.path.dirname(__file__)
# print(basepath)
print(os.getcwd())
model = load_model(r'model/Updated-Xception-diabetic-retinopathy.h5')
client = Cloudant.iam("ee288fae-8bbe-416f-b9a5-6b0c59bd45de-bluemix",
                      "MOuW96ZLdrjw4lHFV15pX7ZmIrSgs4VJqjS95Wp5D1VS", connect=True)
myDB = client.create_database('retinopathy')
app = Flask(__name__)

# pages


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/register')
def register():
    return render_template('register.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/registerUser', methods=['post'])
def registerUser():
    x = [x for x in request.form.values()]
    print(x)
    data = {
        '_id': x[1],
        'name': x[0],
        'pass': x[2]
    }
    print(data)
    query = {'_id': {'$eq': data['_id']}}

    docs = myDB.get_query_result(query)
    print(docs)

    if (len(docs.all()) == 0):
        url = myDB.create_document(data)
        return render_template('register.html', pred='Registration successful, please login with your details')
    else:
        return render_template('register.html', pred="you are already registered. please login with your credential")


@app.route('/loginUser', methods=['POST'])  # type: ignore
def loginUser():
    print(request.form)
    user = request.form['email']
    passw = request.form['password']
    print(user, passw)

    query = {'_id': {'$eq': user}}
    docs = myDB.get_query_result(query)
    print(docs)

    if (len(docs.all()) == 0):
        return render_template('login.html', pred="username not found")
    else:
        if (user == docs[0][0]['_id'] and passw == docs[0][0]['pass']):
            return redirect(url_for('prediction'))
        else:
            print('Invalid user')


@app.route('/predictImage', methods=['POST'])
def predictImage() -> object:
    # print(request.files)
    f = request.files['image']
    basepath = os.path.dirname(__file__)
    print(basepath)
    filepath = os.path.join(basepath, 'uploads', f.filename)  # type: ignore
    print(filepath)
    f.save(filepath)

    img = image.load_img(filepath, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    img_data = preprocess_input(x)
    print('finish1')
    _prediction = np.argmax(model.predict(img_data), axis=1)
    print('finish2')
    index = ['No Diabetic Retinopathy', 'Mild DR',
             'Moderate DR', 'Severe DR', 'Proliferative DR']

    result = str(index[_prediction[0]])
    print(result)
    return render_template('prediction.html', prediction=result)


# main driver function
if __name__ == '__main__':
    app.run(debug=False)
