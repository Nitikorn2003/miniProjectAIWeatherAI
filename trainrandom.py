import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import joblib
import random

# 1. สุ่มข้อมูลที่ต้องการใช้แทนการโหลดจากไฟล์ CSV
# สร้างข้อมูลสุ่มขนาด 1000 แถว โดยใช้ค่า MinTemp, MaxTemp, Precip ที่สุ่มขึ้นมา
np.random.seed(42)  # กำหนด seed เพื่อให้ผลลัพธ์สุ่มเหมือนกันทุกครั้งที่รัน
num_samples = 1000

MinTemp = np.random.uniform(low=-10, high=30, size=num_samples)  # อุณหภูมิต่ำสุด
MaxTemp = MinTemp + np.random.uniform(low=5, high=20, size=num_samples)  # อุณหภูมิสูงสุด
Precip = np.random.uniform(low=0, high=200, size=num_samples)  # ปริมาณน้ำฝน

# รวมข้อมูลทั้งหมดเป็น matrix
x = np.column_stack((MinTemp, MaxTemp, Precip))
y = MaxTemp  # ใช้ MaxTemp เป็นตัวแปรเป้าหมาย

# ฟังก์ชันสำหรับการเทรนโมเดลโดยใช้ random_state สุ่มค่า
def train_and_evaluate(random_state):
    # 2. แบ่งข้อมูลเป็นชุดฝึก (training set) และชุดทดสอบ (test set) ด้วย random_state ที่สุ่ม
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

    # 3. สร้างและฝึกโมเดล Random Forest regression
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(x_train, y_train)

    # 4. บันทึกโมเดลลงไฟล์หลังจากการเทรนเสร็จสิ้น
    joblib.dump(model, f'Wearther_model.pkl')

    # 5. ทำนายค่าจากชุดทดสอบ
    y_pred = model.predict(x_test)

    # 6. คำนวณและเก็บค่าเมตริกสำหรับประเมินผล
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_test, y_pred)
    
    return mae, mse, rmse, r2

# ลูปทำซ้ำหลายรอบ (สุ่มค่า random_state และเทรน)
num_iterations = 1000 # จำนวนรอบที่ต้องการสุ่มเทรน
results = []

for i in range(num_iterations):
    random_state = random.randint(0, 1000)  # สุ่มค่า random_state
    mae, mse, rmse, r2 = train_and_evaluate(random_state)
    results.append((random_state, mae, mse, rmse, r2))
    print(f"Iteration {i+1}: Random State: {random_state}, MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}")

# แสดงผลสรุปหลังจากทำการสุ่มเทรน
print("\nSummary of all iterations:")
for i, (random_state, mae, mse, rmse, r2) in enumerate(results):
    print(f"Iteration {i+1}: Random State: {random_state}, MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R²: {r2}")
