import numpy as np
from flask import Flask, request, render_template, url_for, flash,redirect
from flask_sqlalchemy import SQLAlchemy
#from flask_bcrypt import Bcrypt
import pickle

#creat an app object
app = Flask(__name__)
   
#login details

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
#bcrypt = Bcrypt(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
#...............   
    
#load the trained & tested model
NAD_model = pickle.load(open('models/NAD_model.pkl', 'rb'))
ITD_model = pickle.load(open('models/ITD_model.pkl', 'rb'))

#random generation of ITD data features for testing purposes
# Generate 9018 random input numbers between 0 and 1
input_values = np.random.rand(9018)
# Convert the numpy array to a list
input_values_list = input_values.tolist()
#................................................................

#home rout for initial webpage
@app.route('/')
def home():
    return render_template("login.html")

#login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if(user.password, password):
            flash('Login successful!', 'success')
            return render_template('index.html')
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

#Register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        #hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

    
#using post to allow form submission based on input
#redirect to predict page
@app.route("/NAD_predict", methods=['POST'])
def NAD_predict():
    #convert string inputs into floating numb.
    int_features = [float(x) for x in request.form.values()]
    #covert to the form [[a,b]] for input
    features = [np.array(int_features)]
    #features must be in a form of [[a,b]]
    prediction = NAD_model.predict(features)
    output = round(prediction[0], 2)
    if(output > -1):
        return render_template('index.html', NAD_prediction_text='Your network is safe from attacks... {}'.format(output), prediction_class='no-threat')
    else:
        return render_template('index.html', NAD_prediction_text='Your network is under attack.. {}'.format(output), prediction_class='threat')
        

@app.route("/ITD_predict", methods=['POST'])
def ITD_predict():
    #convert string inputs into floating numb.
    features = [np.array(input_values_list)]
    #features must be in a form of [[a,b]]
    prediction = ITD_model.predict(features)
    output = round(prediction[0], 2)
    if(output < -1):
      return render_template('index.html', ITD_prediction_text='No Insider Threat Detected {}'.format(output), prediction_class='no-threat')
    else:
      return render_template('index.html', ITD_prediction_text='Insider Threat Activity Detected {}'.format(output), prediction_class='threat')

#So if we want to run our code right here, we can check if __name__ == __main__
if __name__ == "__main__":
     with app.app_context():
         db.create_all()
     app.run(debug=True) 