# Smoke detection

## Domain Proyek

Yang melatar belakangi saya membuat project ini adalah karena indonesia menjadi negara dengan mayoritas masyarakatnya perokok aktif dan banyak tempat yang   seharusnya dilarang merokok namun seringkali perokok tetap membakar rokoknya ditempat tersebut.   
- reference 1: https://www.tribunnews.com/internasional/2021/06/02/indonesia-peringkat-ke-3-dan-jepang-ke-7-terbanyak-perokok-di-dunia
- reference 2: https://www.suara.com/health/2021/05/30/132226/indonesia-masuk-10-negara-penyumbang-perokok-terbanyak-di-dunia

- kenapa masalah ini harus diselesaikan? karena banyak dari masyarakat indonesia sering melanggar peraturan dilarang merokok diarea tersebut mengakibatkan orang yang berada di area tersebut dapat terkena dampaknya juga bahkan dapat melebihi dampak yang dihasilkan dari perokok itu sendiri. orang-orang yang berada di no smoking area mengasumsikan kalo diarea tersebut tidak ada perokok sehingga orang-orang yang tidak ingin terkena asap rokok memilih tempat tersebut. selain itu model machine learning ini juga dapat digunakan untuk mendeteksi asap kebakaran jika didalam rumah atau ruangan terdapat asap. salah satu solusi yang dapat saya berikan adalah dengan cara membuat sebuah model machine learning untuk mendeteksi adanya asap ditempat tersebut, yang nantinya model machine learning tersebut dapat dikembangkan ke perangkat IoT agar dapat memfasilitasi no smoking area, ruangan dan rumah. jika perangkat IoT tersebut mendeteksi asap nantinya terdapat sebuah pemberitahuan seperti alarm. 
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
  
  Reference from paper: [Perbandingan effect asap rokok pada smokers dan non-smokers](https://www.sciencedirect.com/science/article/abs/pii/S030057120500117X) 

## Business Understanding

Pada bagian ini, Saya perlu menjelaskan proses klarifikasi masalah.

### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Masyarakat indonesia yang merokok disembarang tempat tanpa memperdulikan sekitar
- Asap kebakaran sulit untuk diprediksi jika secara manual

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Membuat sebuah model machine learning yang dapat mendeteksi asap disuatu tempat atau ruangan dengan akurat berdasarkan kriteria tertentu
- Memberikan rasa aman kepada masyarakat indonesia yang ingin menghidari asap rokok disuatu tempat

    ### Solution statements
    - dapat menggunakan 1 algoritma machine learning yaitu Logistic Regression dan dari algoritma machine learning tersebut kita akan improve recall scorenya menggunakan hyperparameter tuning GridSearchCV.

## Data Understanding
Link download dataset: https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada dataset Smoke detection adalah sebagai berikut:
- UTC: Time when experiment was performed
- Temperature[C]: Temperature of surroundings, measured in celcius
- Humidity[%]: Air humidity during the experiment
- TVOC[ppb]: Total Volatile Organic Compounds, measured in ppb (parts per billion)
- eCO2[ppm]: CO2 equivalent concentration, measured in ppm (parts per million)
- Raw H2: The amount of Raw Hydrogen [Raw Molecular Hydrogen; not compensated (Bias, Temperature etc.)] present in surroundings
- Raw Ethanol: The amount of Raw Ethanol present in surroundings
- Pressure[hPa]: Air pressure, Measured in hPa
- PM1.0: Paticulate matter of diameter less than 1.0 micrometer
- PM2.5: Paticulate matter of diameter less than 2.5 micrometer
- NC0.5: Concentration of particulate matter of diameter less than 0.5 micrometer
- NC1.0: Concentration of particulate matter of diameter less than 1.0 micrometer
- NC2.5: Concentration of particulate matter of diameter less than 2.5 micrometer
- CNT: Sample Count. Fire Alarm(Reality) If fire was present then value is 1 else it is 0
- Fire Alarm: 1 means Positive and 0 means Not Positive

### Data Loading:
- Langkah pertama import library yang kita butuhkan untuk kasus kali ini:
```
# libraries for data manipulation and calculation math
import numpy as np
import pandas as pd

# libraries evaluation
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV

# libraries for preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# libraries for machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# libraries for data visualization
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# libraries for ignore warning after run code
import warnings
warnings.filterwarnings('ignore')
```
- Selanjutnya import dataset dan menampilkan 5 data teratas:
```
df = pd.read_csv('/content/smoke_detection_iot.csv',index_col = False)
df.head()
```
- Kita dapat menghapus feature 'Unnamed: 0' Kemudian melihat dimensi dari dataset ini:
```
df = df.drop(columns='Unnamed: 0')
df.shape
```

### Exploratory Data Analysis - Deskripsi Variabel:
Exploratory data analysis atau sering disingkat EDA merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. Teknik ini biasanya menggunakan bantuan statistik dan representasi grafis atau visualisasi.

Cakupan proses EDA sangat luas. Namun, secara umum, Anda dapat melakukan proses EDA untuk menjawab beberapa pertanyaan berikut:

1. Apa saja jenis variabel pada dataset?
2. Bagaimana distribusi variabel dalam dataset?
3. Apakah ada missing value?
4. Apakah ada fitur yang tidak berguna (redundant)?
5. Bagaimana korelasi antara fitur dan target?


- Melakukan pengecekkan informasi pada tiap variabel pada dataset kita: 
```
df.info()
```
> Seluruh feature pada dataset kita bertipe numeric.

- Melakukan pengecekkan deskripsi statistik pada dataset kita untuk mengetahui apakah terdapat anomalies:
```
df.describe()
```
> Tidak ada keanehan dari basic stats diatas.

Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:

1. Count  adalah jumlah sampel pada data.
2. Mean adalah nilai rata-rata.
3. Std adalah standar deviasi.
4. Min yaitu nilai minimum setiap kolom. 
5. 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama. 
6. 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
7. 75% adalah kuartil ketiga.
8. Max adalah nilai maksimum.

### Exploratory Exploratory Data Analysis - Menangani Missing Value dan Outliers:
- Untuk Melihat missing value pada dataset kita:
```
Total = df.isnull().sum().sort_values(ascending=False)          

Percent = (df.isnull().sum()*100/df.isnull().count()).sort_values(ascending=False)   

missing_data = pd.concat([Total, Percent], axis = 1, keys = ['Total', 'Percentage of Missing Values'])    
missing_data
```
> Tidak terdapat missing value pada dataset kita.

- Untuk melihat outliers pada dataset kita: 
```
cols = df.columns
for i, col in enumerate(cols):
  print("Column:",col)
  plt.figure()
  sns.boxplot(x=df[col])
  plt.show()
```
> Dengan melihat konsistensi nilai-nilai outlier dapat diartikan bahwa hal tersebut bukan karena kesalahan manusia saat menghitung.

### Exploratory Data Analysis - Univariate Analysis:

### Exploratory Data Analysis - Multivariate Analysis:

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

