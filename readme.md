# Laporan Proyek Machine Learning

### Nama : Alya Nur Oktapiani

### Nim : 211351015

### Kelas : Pagi B

## Domain Proyek

Web App yang dikembangkan untuk memudahkan seorang profesional dalam mengambil keputusan apakah seorang pasien mengidap penyakit ginjal kronis. Karena banyak nilai-nilai yang harus diisi dan hanya bisa didapatkan setelah melakukan pengecekan hasil urine di lab, maka disarankan agar web app ini hanya digunakan oleh/atau saat bersama seorang profesional.

## Business Understanding

Dengan semakin banyaknya orang yang terkena penyakit ginjal dikarnakan pola hidup sekarang yang tidak seimbang dan banyak bergadang, maka semakin banyak pula pasien-pasien yang berdatangan pada doctor/ahli profesional. Maka dari itu aplikasi web ini bisa membantu pada profesional dengan cepat mendiagnosa pasien sehingga semakin banyak orang bisa diberikan pengobatan yang tepat.

### Problem Statements
-   Semakin banyaknya pasien sehingga bisa membuat dokter kewalahan dalam mendiagnosa satu satu pasiennya.

### Goals
-   Memudahkan dan mempercepat proses diagnosa dokter terhadap pasien sehingga lebih banyak pasien bisa dirawat/ditangani dengan tepat

## Data Understanding

Datasets yang saya gunakan di sini bernama chronic kidney disease yang saya dapatkan dari kaggle.com. Datasets ini sudah bersih alias sudah layak digunakan untuk dijadikan model maka saya hanya akan melakukan data visualizing dan modeling tanpa melakukan data cleansing. Datasets ini memiliki 14 kolom dengan 400 baris data, berikut adalah tautan untuk mengakses datasetsnya : <br>

[chronic kidney disease](https://www.kaggle.com/datasets/abhia1999/chronic-kidney-disease).

### Variabel-variabel pada chronic kidney disease adalah sebagai berikut:

-   Bp : merupakan tekanan darah/Blood Pressure.
-   Sg : merupakan Specific Gravity/Relative Density/Massa relatif dari urine pasien.
-   Al : merupakan jumlah level albumin pada urine pasien.
-   Su : merupakan jumlah level gula pada urine pasien.
-   Rbc : merupakan red blood cell/cell darah merah pada urine pasien.
-   Bu : merupakan blood urea pada urine pasien.
-   Sc : merupakan serum creatinine yang ada pada urine pasien.
-   Sod : merupakan sodium/zat garam yang ada pada urine pasien.
-   Pot : merupakan pottasium pada urine pasien.
-   Hemo : merupakan hemoglobin pada urine pasien.
-   Wbcc : merupakan jumlah darah putih pada pasien.
-   Rbcc : merupakan jumlah darah merah pada pasien.
-   Htn : merupakan status pasien apakah mengidap hiper tensi atau tidak.
-   Class : merupakan status pasien apakah mengidap penyakit kronis ginjal.

## Data Preparation
Pada bagian ini yang akan saya lakukan adalah mengvisualisasi dan menganalisis datasets yang dipilih, karena dataset ini sudah bersih dan semua kolom sudah sesuai (numerik) untuk diproses maka saya hanya akan mengimpor dan menjadikannya model. Setelah mengvisualisasikannya,
``` bash
#pertama adalah mengimpor file kaggle agar bisa mengunduh datasetnya
from google.colab import files
files.upload()
```
Langkah selanjut membuat folder untuk menyimpan filenya,
``` bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
Lalu kita lanjut dengan mengunduh datasets,
``` bash
!kaggle datasets download -d abhia1999/chronic-kidney-disease
```
Selanjutnya membuat folder baru lalu mengekstrak file yang tadi diunduh ke dalam folder tersebut,
``` bash
!mkdir chronic-kidney-disease
!unzip chronic-kidney-disease.zip -d chronic-kidney-disease
!ls chronic-kidney-disease
```
Langkah selanjutnya adalah mengimpor semua library yang akan digunakan,
``` bash
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```
Lalu membaca file csv yang tadi diekspor dan melihat 5 data pertama pada datasetsnya,
``` bash
data = pd.read_csv("chronic-kidney-disease/new_model.csv")
data.head()
```
Kita juga akan melihat 5 data pertama yang tidak mengidap penyakit ginjal kronis,
``` bash
data[data["Class"] == 0].head()
```
langkah selanjutnya adalah melihat tipe data dari setiap kolomnya,
``` bash
data.info()
```
Seperti yang terlihat dari hasilnya, semua tipe datanya sudah numerik (float dan int), ini sudah bisa diproses oleh suatu algorithma,
``` bash
data.describe()
```
Dengan ini kita bisa melihat jumlah data yang ada serta mean, min dan maxnya, selanjutnya kita akan lihat apakah datanya memiliki nilai null, untuk berjaga-jaga saja,
``` bash
sns.heatmap(data.isnull())
```
![download](https://github.com/alyatoon/chronickidney_prediksi/assets/149295614/03dece55-5767-4e00-b1fe-b08dd9427dc9)
Merah semua bertanda bahwa tidak ada nilai null,
``` bash
corr_matrix = data.corr()
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, ax=ax)
plt.show()
```
![download](https://github.com/alyatoon/chronickidney_prediksi/assets/149295614/5f3c055e-fa23-452f-abed-8d11f61cbf23) <br>
Diatas merupakan korelasi antar kolom, selanjutnya kita akan melihat visualisasi setiap kolom kategorial,
``` bash
categorical_columns = ['Sg', 'Al', 'Su', 'Rbc', 'Htn', 'Class']

sns.set(style='whitegrid')

plt.figure(figsize=(10, 6))
for i, cat_var in enumerate(categorical_columns, 1):
    plt.subplot(2, 3, i)
    sns.countplot(data=data, x=cat_var, color='Teal', alpha=0.7)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel(cat_var, fontsize=8)
    plt.ylabel('')

plt.tight_layout()
plt.show()
```
![download](https://github.com/alyatoon/chronickidney_prediksi/assets/149295614/ef617419-19df-4bf4-b602-eb70d3a93ca9) <br>
Selanjutnya kolom numeriknya,
``` bash
numerical_columns = ['Bp', 'Bu', 'Sc', 'Sod', 'Pot', 'Hemo', 'Wbcc', 'Rbcc']

sns.set(style='whitegrid')
plt.figure(figsize=(10,6))
for i, var in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=data, x=var, kde=True, bins=20, color='Teal')
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.xlabel(var, fontsize=8)
    plt.ylabel('')
plt.tight_layout()
plt.show()
```
![download](https://github.com/alyatoon/chronickidney_prediksi/assets/149295614/e7dfd386-169b-4dc9-9c92-035d1145c67b) <br>
Visualisasi dan explorasi datasets sudah dilakukan, langkah selanjutnya adalah modeling.
## Modeling
Algorithma yang saya gunakan disini adalah Logistic Regression, salah satu algorithma yang sangat umum digunakan untuk melakukan klasifikasi suatu data. Karena kita ingin melakukan pengklasifikasian apakah seseorang tersebut mengidap penyakit ginjal atau tidak maka kita akan gunakan ini, <br>
Langkah pertama tentunya menentukan fitur dan target yang diinginkan
```
X = data.drop('Class', axis=1)
y = data['Class']
```
Lalu memasukkan semua library yang akan digunakan,
``` bash
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
```
Dilanjut dengan melakukan train_test_split dengan test 20% dan 80% train menggunakan datasetnya seperti berikut,
``` bash
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
Selanjutnya membuat model Logistic Regression dengan iterasi 500 kali dan melakukan data fitting dan membuat X_pred,
```
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
X_pred = model.predict(X_test)
```
Langkah selanjutnya mari kita melihat score yang didapatkan!,
``` bash
accuracy = accuracy_score(X_pred, y_test)
print("Skor akurasi:", accuracy)
```
Score yang kita dapatkan adalah 97,5% cukup tinggi!, mari kita coba dengan menggunakan data palsu yang kita buat,
``` bash
# Contoh data pasien terkena penyakit ginjal
# input_data =  np.array([70.0,1.005,4.0,0.0,1.0,56.0,3.8,111.00,2.50,11.2,6700.0,3.90,1.0])
# Contoh data pasien tidak terkena penyakit ginjal
input_data = np.array([80.0,1.025,0.0,0.0,1.0,10.0,1.2,135.0,5.0,15.0,10400.0,4.5,0.0])

input_data_reshaped = input_data.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if (prediction[0] == 1) :
    print("Pasien terkena penyakit ginjal kronis")
else :
    print("Pasien tidak terkena penyakit ginjal kronis")
```
Hasilnya adalah 0 dengan pesan "Pasien tidak terkena penyakit ginjal kronis". <br>
Kita akan menyimpan hasil modelnya dengan menggunakan pickle untuk digunakan pada streamlit nanti.
```
import pickle

filename = 'chronic-kidney.sav'
pickle.dump(model, open(filename, 'wb'))
```
## Evaluation
Karena hasil yang diharapkan merupakan kategorial/klasifikasi, kita akan menggunakan metrik evaluasi presisi dan bukan akurasi(yang lebih umum digunakan untuk hasil angka). Kita juga akan menggunakan recall, kode yang saya gunakan untuk Matrix Evaluation ini adalah seperti berikut,
``` bash
precision= precision_score(y_test,y_pred)
recall= recall_score(y_test,y_pred)

print("Precision dari logistic regression: ", precision)
print("Recall dari logistic regression :", recall)
```
Hasil dari model yang diciptkan adalah presisi 100% dan recall 96% yang mana ini cukup tinggi dan memiliki kemampuan untuk memgkategorikan sesuatu dengan tepat.

## Deployment
[Aplikasi Web Ginjal Kronis Alya](https://ginjalkronis-prediksi-alya.streamlit.app/)
![image](https://github.com/alyatoon/chronickidney_prediksi/assets/149295614/5415b452-60f9-4c6d-8ee6-349ee0234796)
