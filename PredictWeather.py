import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import joblib
import matplotlib.pyplot as plt

# 1. โหลดชุดข้อมูลจากไฟล์ CSV
dataset = pd.read_csv("Weather.csv")

# 2. แปลงคอลัมน์ให้เป็นตัวเลข และจัดการค่าที่หายไป
dataset["MinTemp"] = pd.to_numeric(dataset["MinTemp"], errors='coerce')
dataset["MaxTemp"] = pd.to_numeric(dataset["MaxTemp"], errors='coerce')
dataset["Precip"] = pd.to_numeric(dataset["Precip"], errors='coerce')
dataset.dropna(subset=["MinTemp", "MaxTemp", "Precip"], inplace=True)

# 3. กำหนดตัวแปรนำเข้า (input features) และตัวแปรเป้าหมาย (output variable)
x = dataset[["MinTemp", "MaxTemp", "Precip"]].values  
y = dataset["MaxTemp"].values.reshape(-1,)  

# 4. แบ่งข้อมูลเป็นชุดฝึก (training set) และชุดทดสอบ (test set)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# 5. สร้างและฝึกโมเดล Random Forest regression
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(x_train, y_train)

# 6. บันทึกโมเดลลงไฟล์หลังจากการเทรนเสร็จสิ้น
joblib.dump(model, 'Wearther_model.pkl')
print("โมเดลถูกบันทึกเรียบร้อยแล้ว")

# 7. ทำนายค่าจากชุดทดสอบ
y_pred = model.predict(x_test)

# 8. สร้าง DataFrame สำหรับเปรียบเทียบค่าจริงและค่าที่ทำนาย
df = pd.DataFrame({'Actually': y_test, 'Predicted': y_pred})

# 9. สร้างกราฟเพื่อแสดงค่าจริงกับค่าที่ทำนาย
df1 = df.head(40)
df1.plot(kind="bar", figsize=(10, 5))
plt.title("Actual vs Predicted Values")
plt.xlabel("Index")
plt.ylabel("Temperature")
plt.xticks(rotation=0)
plt.show()

# 10. คำนวณและแสดงเมตริกสำหรับประเมินผล
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("R² Score:", metrics.r2_score(y_test, y_pred))