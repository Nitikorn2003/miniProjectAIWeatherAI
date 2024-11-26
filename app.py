from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import joblib
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # โฟลเดอร์สำหรับเก็บไฟล์อัพโหลด

# ตรวจสอบว่าโฟลเดอร์ uploads มีอยู่หรือไม่ ถ้าไม่มีก็สร้างขึ้น
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# โหลดโมเดลที่ถูกบันทึกไว้
model = joblib.load('Wearther_model.pkl')

@app.route('/')
def home():
    return render_template('index.html', 
                           mean_absolute_error=None, 
                           mean_squared_error=None,
                           root_mean_squared_error=None,
                           r2_score=None,
                           average_temperature=None)  # เพิ่มตัวแปรค่าเฉลี่ยอุณหภูมิ

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # โหลดชุดข้อมูลจากไฟล์ CSV
        dataset = pd.read_csv(filepath)

        # ตรวจสอบคอลัมน์ที่มีอยู่ในข้อมูล
        required_columns = ["MinTemp", "MaxTemp", "Precip"]
        for column in required_columns:
            if column not in dataset.columns:
                return render_template('index.html', 
                                       mean_absolute_error=None, 
                                       mean_squared_error=None,
                                       root_mean_squared_error=None,
                                       r2_score=None,
                                       average_temperature=None,
                                       error="Error: Missing required column: {}".format(column))

        # จัดการข้อมูลที่เป็นตัวเลขและลบค่า NaN
        dataset["MinTemp"] = pd.to_numeric(dataset["MinTemp"], errors='coerce')
        dataset["MaxTemp"] = pd.to_numeric(dataset["MaxTemp"], errors='coerce')
        dataset["Precip"] = pd.to_numeric(dataset["Precip"], errors='coerce')
        
        dataset.dropna(subset=["MinTemp", "MaxTemp", "Precip"], inplace=True)

        # กำหนดตัวแปรนำเข้า (input features) และตัวแปรเป้าหมาย (output variable)
        x = dataset[["MinTemp", "MaxTemp", "Precip"]].values  
        y = dataset["MaxTemp"].values.reshape(-1,)  

        # ทำนายค่าจากข้อมูลใหม่โดยใช้โมเดลที่บันทึกไว้
        y_pred = model.predict(x)

        # สร้าง DataFrame สำหรับเปรียบเทียบค่าจริงและค่าที่ทำนาย
        df = pd.DataFrame({'Actually': y, 'Predicted': y_pred})

        # คำนวณค่าเฉลี่ยของอุณหภูมิจริงและที่ทำนาย
        average_temperature = df['Actually'].mean()  # ค่าเฉลี่ยของอุณหภูมิจริง

        # สร้างกราฟเพื่อแสดงค่าจริงกับค่าที่ทำนาย
        df1 = df.head(30)
        df1.plot(kind="bar", figsize=(10, 5))
        plt.title("Actual vs Predicted Values")
        plt.xlabel("Index")
        plt.ylabel("Temperature")
        plt.xticks(rotation=0)

        # บันทึกกราฟลงในไฟล์
        plt.savefig('static/plot.png')
        plt.close()  # ปิดกราฟ

        # คำนวณและแสดงเมตริกสำหรับประเมินผล
        mean_absolute_error = metrics.mean_absolute_error(y, y_pred)
        mean_squared_error = metrics.mean_squared_error(y, y_pred)
        root_mean_squared_error = np.sqrt(mean_squared_error)
        r2_score = metrics.r2_score(y, y_pred)

        return render_template('index.html',
                               mean_absolute_error=mean_absolute_error,
                               mean_squared_error=mean_squared_error,
                               root_mean_squared_error=root_mean_squared_error,
                               r2_score=r2_score,
                               average_temperature=average_temperature)  # ส่งค่าเฉลี่ยอุณหภูมิไปยังเทมเพลต

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
