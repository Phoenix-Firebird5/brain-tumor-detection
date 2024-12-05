from distutils.log import debug
from fileinput import filename
from os import environ
from flask import *
import datetime
import mysql.connector
import hashlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import os
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
import io
from PIL import Image


labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']

image_size = 150
effnet = EfficientNetB0(weights='efficientnetb0_notop.h5',include_top=False,input_shape=(image_size,image_size,3))

model = effnet.output
model = tf.keras.layers.GlobalAveragePooling2D()(model)
model = tf.keras.layers.Dropout(rate=0.5)(model)
model = tf.keras.layers.Dense(4,activation='softmax')(model)
model = tf.keras.models.Model(inputs=effnet.input, outputs = model)
#model.summary()
model.compile(loss='categorical_crossentropy',optimizer = 'Adam', metrics= ['accuracy'])

model.load_weights("brain-tumor-weights.h5")


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="141519",
  database="braindiseases"
)
mycursor = mydb.cursor()

app = Flask(__name__)
app.secret_key = "abc"

@app.route('/')  
def main():  
    return render_template("index.html")  

@app.route('/about')  
def about():  
    return render_template("about.html")

@app.route('/contact')  
def contact():  
    return render_template("contact.html")

@app.route('/signin')  
def signin():  
    return render_template("signin.html")

@app.route('/signout')  
def signout():
    del(session['response'])
    return render_template("signin.html")

@app.route('/PRegistration')  
def PRegistration():  
    return render_template("PRegistration.html")

@app.route('/CheckDisease')  
def CheckDisease():  
    return render_template("CheckDisease.html")


@app.route('/Pregsuccess', methods = ['POST'])  
def Pregsuccess():  
    if request.method == 'POST':
        FNam = request.form['FNam']
        LNam = request.form['LNam']
        Mobile = request.form['Mobile']
        Emailid = request.form['Emailid']
        
        mess=""
        if not FNam:
            mess=mess+"Enter First Name, "
        if not re.match("^[a-zA-Z]*$", FNam):
            mess=mess+"Enter valid First Name, "
        if not LNam:
            mess=mess+"Enter Last Name, "
        if not re.match("^[a-zA-Z]*$", LNam):
            mess=mess+"Enter valid Last Name, "
        if not Emailid:
            mess=mess+"Enter Email, "
        if not re.match("^[\w\.\+\-]+\@[\w]+\.[a-z]{2,3}$", Emailid):
            mess=mess+"Enter valid Email, "
        if not Mobile:
            mess=mess+"Enter Mobile No, "
        if not re.match("^[0-9]{10}$", Mobile):
            mess=mess+"Enter valid Mobile No., "

        if not mess:
            sql="INSERT INTO patient(Fname,Lname,Email,Mobile,Result) VALUES (%s,%s,%s,%s,%s)"
            val=(FNam,LNam,Emailid,Mobile,"")
            mycursor.execute(sql,val)
            mydb.commit()
            Respon=make_response("")
            if mycursor.rowcount==1:
                Respon=make_response("<font color='#0000FF'>Patient Registration Successfully.!</font><br><br>")
            else:
                Respon=make_response("<font color='#FF0000'>Patient Registration Fail.!</font><br><br>")
        else:
            Respon=make_response("<font color='#FF0000'>Patient Registration Fail -"+mess+"</font><br><br>")

        return Respon
        #return render_template("Acknowledgement.html", name = filed+f.filename)


@app.route('/success', methods = ['POST'])  
def success():
    remess=""
    Respon=make_response("")
    if request.method == 'POST':
        patientid = request.form['PID']
        mycursor.execute("SELECT * FROM patient where pid='"+patientid+"'")
        rcc=len(mycursor.fetchall())
        if mycursor.rowcount>=1:
            f1 = request.files['OriginalPhoto']
            filed1=datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')+"O_"+f1.filename
            f1.save("static/tmp/"+filed1)

            img = Image.open("static/tmp/"+filed1)
            opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = cv2.resize(opencvImage,(150,150))
            img = img.reshape(1,150,150,3)
            p = model.predict(img)
            p = np.argmax(p,axis=1)[0]
            Tumortype=""
            if p==0:
                Tumortype='Glioma Tumor'
            elif p==1:
                Tumortype='No Tumor'
            elif p==2:
                Tumortype='Meningioma Tumor'
            else:
                Tumortype='Pituitary Tumor'
                
            if p!=1:
                sql="update patient set Result='"+Tumortype+"' where pid='"+patientid+"'"
                mycursor.execute(sql)
                mydb.commit()
                remess=Tumortype+" Detected."
                Respon=make_response("<font color='#0000FF'>"+remess+".!</font><br><img src='"+"static/tmp/"+filed1+"' style='width:300px;'><br>")
            elif p==1:
                remess=Tumortype+" Detected."
                Respon=make_response("<font color='#0000FF'>"+remess+".!</font><br><img src='"+"static/tmp/"+filed1+"' style='width:300px;'><br>")
            else:
                Respon=make_response("<font color='#FF0000'>Fail To Detect.!</font><br><br>")

        else:
            Respon=make_response("<font color='#FF0000'>Patient Record Not Found.!</font><br><br>")        

    return Respon


@app.route('/login', methods = ['POST'])  
def login():
    Respon=make_response("")
    if request.method == 'POST':
        UserEmail = request.form['UserEmail']
        UserPass = request.form['UserPass']
        if UserEmail=='admin' and UserPass=='admin':
            session['response']='adminsession'
            Respon=make_response("<font color='#FF0000'>Login Successfully.!</font><script>location.href=\"/Mainpage\";</script>")
        else:
            Respon=make_response("<font color='#FF0000'>Login Fail Check Email and Password.!</font><br><br>")

    return Respon

@app.route('/Mainpage', methods=['GET'])  
def Mainpage():
    htmData=""
    Respon=make_response("")
    if 'response' in session:  
        s = session['response'];

        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM patient")
        myresult = mycursor.fetchall()
        htmData=htmData+'<table class="table"><thead><tr><th>ID</th><th>Name</th><th>Email</th><th>Mobile</th><th>Result</th></tr></thead><tbody>'
        for row in myresult:                
            htmData=htmData+'<tr><td class="image">'+str(row[0])+'</td><td class="name">'+row[1]+' '+row[2]+'</td><td class="name">'+row[3]+'</td><td class="name">'+row[4]+'</td><td class="name">'+row[5]+'</td></tr>'

        htmData=htmData+'</tbody></table>'
        Respon = make_response(render_template("Mainpage.html", HTMLData = htmData))
    else:
        Respon=make_response("<script>location.href=\"/signin\";</script>");

    return Respon
            
@app.route('/shutdown')
def shutdown():
    SystemExit.exit()
    os.exit(0)
    return
   
if __name__ == '__main__':
   HOST = environ.get('SERVER_HOST', 'localhost')
   try:
      PORT = int(environ.get('SERVER_PORT', '5555'))
   except ValueError:
      PORT = 5555
   app.run(HOST, PORT)
   #app.run(debug=True)
