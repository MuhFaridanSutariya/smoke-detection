# Smoke detection

![4](https://user-images.githubusercontent.com/88027268/203010892-b35284b1-4779-4764-923b-6b94c065a446.jpeg)
Gambar 1. Illustrasi asap rokok

## Domain Proyek

Yang melatar belakangi pembuatan project ini adalah karena indonesia menjadi negara dengan mayoritas masyarakatnya perokok aktif dan banyak tempat yang seharusnya dilarang merokok namun seringkali perokok tetap membakar rokoknya ditempat tersebut. 

- kenapa masalah ini harus diselesaikan? karena banyak dari masyarakat indonesia sering melanggar peraturan dilarang merokok diarea tersebut mengakibatkan orang yang berada di area tersebut dapat terkena dampaknya juga bahkan dapat melebihi dampak yang dihasilkan dari perokok itu sendiri. orang-orang yang berada di *no smoking area* mengasumsikan kalo diarea tersebut tidak ada perokok sehingga orang-orang yang tidak ingin terkena asap rokok memilih tempat tersebut. selain itu model *machine learning* ini juga dapat digunakan untuk mendeteksi asap kebakaran jika didalam rumah atau ruangan terdapat asap. salah satu solusi yang dapat saya berikan adalah dengan cara membuat sebuah model *machine learning* untuk mendeteksi adanya asap ditempat tersebut, yang nantinya model *machine learning* tersebut dapat dikembangkan ke perangkat *IoT* agar dapat memfasilitasi diberbagai tempat. jika perangkat *IoT* tersebut mendeteksi asap nantinya terdapat sebuah pemberitahuan seperti *alarm*. 
 
## Business Understanding

Pada bagian ini, akan menjelaskan proses klarifikasi masalah.

### Problem Statements

<img width="960" alt="1" src="https://user-images.githubusercontent.com/88027268/203011051-26a43e4f-4235-467b-9dda-4213f07b131d.png">
Gambar 2. Illustrasi asap rokok

Menjelaskan pernyataan masalah latar belakang:
- Masyarakat indonesia yang merokok disembarang tempat tanpa memperdulikan sekitar.
- Asap kebakaran sulit untuk diprediksi jika secara manual.

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Membuat sebuah model *machine learning* yang dapat mendeteksi asap disuatu tempat atau ruangan yang berpotensi membahayakan manusia dengan akurat.
- Memberikan rasa aman kepada masyarakat indonesia yang ingin menghidari asap rokok disuatu tempat.

    ### Solution statements
    - dapat menggunakan satu algoritma *machine learning* yaitu *Logistic Regression* dan dari algoritma *machine learning* tersebut kita akan improve recall scorenya menggunakan hyperparameter tuning *GridSearchCV*.

## Data Understanding
Link download dataset: https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset  

### Variabel-variabel pada dataset Smoke detection adalah sebagai berikut:
- *UTC: Time when experiment was performed
- *Temperature[C]: Temperature of surroundings, measured in celcius
- *Humidity[%]: Air humidity during the experiment
- *TVOC[ppb]: Total Volatile Organic Compounds, measured in ppb (parts per billion)
- *eCO2[ppm]: CO2 equivalent concentration, measured in ppm (parts per million)
- *Raw H2: The amount of Raw Hydrogen [Raw Molecular Hydrogen; not compensated (Bias, Temperature etc.)] present in surroundings
- *Raw Ethanol: The amount of Raw Ethanol present in surroundings
- *Pressure[hPa]: Air pressure, Measured in hPa
- *PM1.0: Paticulate matter of diameter less than 1.0 micrometer
- *PM2.5: Paticulate matter of diameter less than 2.5 micrometer
- *NC0.5: Concentration of particulate matter of diameter less than 0.5 micrometer
- *NC1.0: Concentration of particulate matter of diameter less than 1.0 micrometer
- *NC2.5: Concentration of particulate matter of diameter less than 2.5 micrometer
- *CNT: Sample Count. Fire Alarm(Reality) If fire was present then value is 1 else it is 0
- *Fire Alarm: 1 means Positive and 0 means Not Positive

### Data Loading:
Langkah pertama import *library** yang dibutuhkan untuk kasus kali ini:

*Library* yang akan diimport adalah *library* yang berhubungan untuk memanipulasi data, data visualisasi, *preprocessing*, pengembangan *machine learning*, evaluasi dan mematikan warning yang didapat setelah menjalankan code. 

Selanjutnya membaca dataset dan menampilkan 5 data teratas:
|   	| UTC        	| Temperature[C] 	| Humidity[%] 	| TVOC[ppb] 	| eCO2[ppm] 	| Raw H2 	| Raw Ethanol 	| Pressure[hPa] 	| PM1.0 	| PM2.5 	| NC0.5 	| NC1.0 	| NC2.5 	| CNT 	|
|---	|------------	|----------------	|-------------	|-----------	|-----------	|--------	|-------------	|---------------	|-------	|-------	|-------	|-------	|-------	|-----	|
| 0 	| 1654733331 	| 20.0           	| 57.36       	| 0         	| 400       	| 12306  	| 18520       	| 939.735       	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 0   	|
| 1 	| 1654733332 	| 20.015         	| 56.67       	| 0         	| 400       	| 12345  	| 18651       	| 939.744       	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 1   	|
| 2 	| 1654733333 	| 20.029         	| 55.96       	| 0         	| 400       	| 12374  	| 18764       	| 939.738       	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 2   	|
| 3 	| 1654733334 	| 20.044         	| 55.28       	| 0         	| 400       	| 12390  	| 18849       	| 939.736       	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 3   	|
| 4 	| 1654733335 	| 20.059         	| 54.69       	| 0         	| 400       	| 12403  	| 18921       	| 939.744       	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 4   	|

Kita dapat menghapus fitur 'Unnamed: 0' Kemudian melihat dimensi dari dataset ini. lalu melihat dimensi dari dataset kita.
```
print("Row: {}, Columns: {}".format(df.shape[0], df.shape[1]))
```
![3](https://user-images.githubusercontent.com/88027268/203011353-91f2f557-8737-4427-8106-575c485db4dd.jpg)
Gambar 3. Dimensi dataset
> Dataset ini terdiri dari 62630 data dan 10 columns.

### Exploratory Data Analysis - Deskripsi Variabel:
*Exploratory data analysis* atau sering disingkat EDA merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. Teknik ini biasanya menggunakan bantuan statistik dan representasi grafis atau visualisasi.

Cakupan proses EDA sangat luas. Namun, secara umum, Anda dapat melakukan proses EDA untuk menjawab beberapa pertanyaan berikut:

1. Apa saja jenis variabel pada dataset?
2. Bagaimana distribusi variabel dalam dataset?
3. Apakah ada *missing value*?
4. Apakah ada fitur yang tidak berguna (*redundant*)?
5. Bagaimana korelasi antara fitur dan target?


Melakukan pengecekkan informasi pada tiap variabel pada dataset: 
```
df.info()
```
![5](https://user-images.githubusercontent.com/88027268/203011977-257ab80d-f659-45b0-a0ac-479b8e9d8ee0.jpg)
Gambar 4. Keterangan dari tiap fitur pada dataset
> Seluruh fitur pada dataset kita bertipe numeric.



Melakukan pengecekkan deskripsi statistik pada dataset untuk mengetahui apakah terdapat anomali:
```
df.describe()
```
![6](https://user-images.githubusercontent.com/88027268/203012023-bd10d1c8-0334-47d3-a537-d538b52f7379.jpg)
Gambar 5. Kalkulasi deskripsi statistik dari dataset 
> Tidak ada keanehan dari deskripsi statistik diatas.

*Function describe()* memberikan informasi statistik pada masing-masing kolom, antara lain:

1. *Count*  adalah jumlah sampel pada data.
2. *Mean* adalah nilai rata-rata.
3. *Std* adalah standar deviasi.
4. *Min* yaitu nilai minimum setiap kolom. 
5. *25%* adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama. 
6. *50%* adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
7. *75%* adalah kuartil ketiga.
8. *Max* adalah nilai maksimum.

### Exploratory Exploratory Data Analysis - Menangani Missing Value dan Outliers:

![7](https://user-images.githubusercontent.com/88027268/203012078-75fde3d1-5d66-457a-8bb7-21dc021897ad.jpg)
Gambar 5. Missing value pada dataset
> Tidak terdapat missing value pada dataset.

![8](https://user-images.githubusercontent.com/88027268/203012132-ca1b43c3-1202-4ecb-b1f6-f0d366029902.jpg)
![9](https://user-images.githubusercontent.com/88027268/203012160-f8eb630f-6770-4f48-baf0-7fae887c03b1.jpg)
![10](https://user-images.githubusercontent.com/88027268/203012189-b189c06d-6701-40bd-b466-e4561e06a474.jpg)
![11](https://user-images.githubusercontent.com/88027268/203012218-724f7176-142e-428f-b095-fd845096f658.jpg)
![12](https://user-images.githubusercontent.com/88027268/203012254-f7c28332-b38a-4507-8ca1-3517aa0071b5.jpg)
Gambar 6. Outlier pada dataset
> Dengan melihat konsistensi nilai-nilai *outlier* dapat diartikan bahwa hal tersebut bukan karena kesalahan manusia saat menghitung.

### Exploratory Data Analysis - Univariate Analysis:
Melihat proporsi dari nilai variabel target pada dataset:

![13](https://user-images.githubusercontent.com/88027268/203012313-e90c3936-0252-4286-88e2-24e833ddd769.png)
Gambar 7. Proporsi pada fitur target

> Terjadi data *imbalance* pada variabel target. Permasalahan tersebut dapat diselesaikan dengan cara melakukan pendekatan *undersampling* dan *uppersampling* pada variabel target atau dengan cara lain yaitu memilih *metric* yang tepat seperti *recall*, *precision* dan *F1 Score*. Tidak dapat menggunakan *metric* *accuracy* karena dapat menyebabkan bias pada saat *scoring* model.

### Exploratory Data Analysis - Multivariate Analysis:
Melihat *kernel density estimation* (KDE) plot dari tiap variabel:

Melihat hubungan humidity dengan variabel target menggunakan KDE
![14](https://user-images.githubusercontent.com/88027268/203012378-5e964184-d1d6-45c9-a4b2-6c03ff518b0b.png)
Gambar 8. KDE Humidity vs Fire Alarm
> Selama *experiment* kemungkinan *fire alarm* ditempat yang memiliki kelembapan cukup tinggi. karena puncak *density* pada visualisasi diatas mencakup kelembapan > 40%


Melihat hubungan temperature dengan variabel target menggunakan KDE
![15](https://user-images.githubusercontent.com/88027268/203012429-d73bc0cc-2430-48be-9b13-d85fb0520b05.png)
Gambar 9. KDE Temperature vs Fire Alarm
> Selama *experiment* kemungkinan *fire alarm* ditempat yang memiliki *temperature* sekitar 20 celcius

Melihat hubungan Pressure dengan variabel target menggunakan KDE
![16](https://user-images.githubusercontent.com/88027268/203012552-edf1b456-3842-4f82-99d6-9fc148f003fd.png)
Gambar 10. KDE Pressure vs Fire Alarm
> Berdasarkan puncak *density* dari visualisasi diatas dikita lihat bahwa semakin tinggi *Pressure* maka kemungkinan untuk *fire alarm* berbunyi semakin besar juga

Melihat hubungan Raw H2 dengan variabel target menggunakan KDE
Gambar 11. KDE Raw H2 vs Fire Alarm
![17](https://user-images.githubusercontent.com/88027268/203012750-ae9d8d1d-246f-4b93-9de5-87a05891b00c.png)
> Berdasarkan puncak *density* dari visualisasi diatas dapat dilihat bahwa *Raw H2* pada *yes fire* dan *no fire* memiliki rentang yang serupa yaitu 12500 - 1340

Melihat hubungan Raw Ethanol dengan variabel target menggunakan KDE
Gambar 12. KDE Raw Ethanol vs Fire Alarm
![18](https://user-images.githubusercontent.com/88027268/203012805-be948f4d-9b96-4978-92cd-d35da0cdb0ce.png)
> Berdasarkan puncak *density* dari visualisasi diatas dapat dilihat bahwa *yes fire* memiliki kecenderungan berada di jumlah *Raw Ethanol* sekitar 19500 - 20500 dan *no fire* memiliki kecenderungan berada di jumlah *Raw Ethanol* sekitar 20000 - 21000

Melihat korelasi antara tiap fitur:

![19](https://user-images.githubusercontent.com/88027268/203012863-b8347aed-42c5-486e-8851-e547ef663457.png)
Gambar 13. Korelasi ditiap fitur pada dataset
- Semua kolom *'PM's* dan *'NC's* memiliki korelasi yang tinggi dengan sesama kolom tersebut
- Tidak ada fitur yang berkorelasi tinggi dengan fitur target. *Humidity*, *Pressure* dan *Raw H2* adalah fitur yang memiliki korelasi positif namun tidak tinggi dan sisanya adalah fitur yang berkorelasi rendah dengan fitur targetnya.

## Data Preparation
![22](https://user-images.githubusercontent.com/88027268/203012905-826bd461-2a07-43c8-b63b-129c621f49aa.jpg)
Gambar 14. Illustrasi Data Preparation

Pada bagian ini terdapat empat tahap persiapan data, yaitu:

- *Feature Selection*
- Reduksi dimensi dengan *Principal Component Analysi*s (PCA)
- Pembagian dataset dengan fungsi train_test_split dari library sklearn
- Standarisasi

### Feature Selection
*Feature selection* adalah proses mengurangi jumlah fitur atau variabel input dengan memilih fitur-fitur yang dianggap paling relevan terhadap model.

- Fitur yang sangat didominasi oleh satu nilai saja akan dibuang pada tahap ini. karena fitur yang didominasi satu nilai saja tidak berarti untuk *machine learning*.
```
for col in df.columns.tolist():
    print(df[col].value_counts().count())
    print('\n')
```
![20](https://user-images.githubusercontent.com/88027268/203013002-6875dab8-de7e-455c-8147-8fa6cc0175d7.jpg)
Gambar 15. Distribusi value tiap fitur
> Tidak ada fitur dengan satu nilai saja maka tidak ada fitur yang harus dibuang. 

- menghapus kolom *UTC* karena tidak berpengaruh pada model *machine learning* sehingga hal ini dapat memudahkan *machine learning* dalam pencari pola dari dataset.
```
df = df.drop(columns='UTC')
```

### Principal Component Analysis (PCA)
*PCA* bekerja menggunakan metode aljabar linier. Ia mengasumsikan bahwa sekumpulan data pada arah dengan *varians* terbesar merupakan yang paling penting (utama). *PCA* umumnya digunakan ketika variabel dalam data memiliki korelasi yang tinggi. Korelasi positif yang tinggi ini menunjukkan data yang berulang atau *redundant*.

Fitur yang akan dilakukan PCA adalah *PM1.0, PM2.5, NC0.5, NC1.0 dan NC2.5*. Karena fitur tersebut saling berkorelasi positif dan cukup tinggi.

```
pca = PCA(n_components=5, random_state=42)
pca.fit(df[['PM1.0', 'PM2.5', 'NC0.5', 'NC1.0','NC2.5']])
princ_comp = pca.transform(df[['PM1.0', 'PM2.5', 'NC0.5', 'NC1.0','NC2.5']])
```
- parameter *n_components* adalah jumlah komponen atau dimensi seperti dikasus ini 5 yaitu *'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0' dan 'NC2.5'*
- parameter *random_state* berfungsi untuk mengontrol *random number generator* yang digunakan. Parameter ini berupa bilangan *integer* dan nilainya bebas. Pada kasus ini, Menerapkan *random_state = 42*. Berapa pun nilai *integer* yang ditentukan. selama itu bilangan *integer*, ia akan memberikan hasil yang sama setiap kali dilakukan pemanggilan fungsi.

```
pca.explained_variance_ratio_.round(3)
```
![23](https://user-images.githubusercontent.com/88027268/203013240-27bb6097-ed9f-48b6-8804-f3957fc4e6c5.jpg)
Gambar 16. Hasil PCA

> Arti dari *output* di atas adalah, 0.9% informasi pada kelima fitur *'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0' dan 'NC2.5'* terdapat pada *Principal Component* pertama. Sedangkan sisanya, sebesar 0.1%, 0.0%, 0.0% dan 0.0% terdapat pada *Principal Component* kedua, ketiga, keempat dan kelima.

Dari hasil diatas dapat dipertahankan pada *Principal Component* pertama saja untuk menggantikan kelima fitur yang telah direduksi sebelumnya lalu kita beri nama fitur ini dengan *'dimension'*

```
pca = PCA(n_components=1, random_state=42)
pca.fit(df[['PM1.0', 'PM2.5', 'NC0.5', 'NC1.0','NC2.5']])
df['dimension'] = pca.transform(df.loc[:, ('PM1.0', 'PM2.5', 'NC0.5', 'NC1.0','NC2.5')]).flatten()
df.drop(['PM1.0', 'PM2.5', 'NC0.5', 'NC1.0','NC2.5'], axis=1, inplace=True)
df.head()
```
![24](https://user-images.githubusercontent.com/88027268/203013279-8120c2e2-af49-4f8b-a921-5e6ac4c57ff2.jpg)
Gambar 17. Overview dataset

### Pembagian dataset

Selanjutnya dapat melakukan *split data* ke dalam beberapa bagian yaitu *data train* dan *data test*. kita melakukan beberapa transformasi pada *data train* sedangkan *data test* kita gunakan sebagai data uji yang diasumsikan adalah data baru sama halnya nanti di *production*. karna *data test* adalah data baru sehingga kita tidak diperkenankan untuk melakukan transformasi apapun, jika kita melakukan transformasi pada *data test* itu akan menimbulkan masalah baru yaitu *data leakage* dan dapat menyebabkan model *machine learning* kita *bias*.

proporsi pembagian data latih dan uji biasanya adalah 80:20. Bahwa proporsi ini hanya kebiasaan umum saja. Tujuan dari data uji adalah untuk untuk mengukur kinerja model pada data baru. Jadi, jika dataset yang kita miliki berukuran sangat kecil, misalnya kurang dari 1.000 sampel, maka pembagian 80:20 ini cukup *ideal*. Namun, jika memiliki dataset berukuran besar, kita perlu memikirkan strategi pembagian dataset lain agar proporsi data uji tidak terlalu banyak.

Sebagai contoh, Terdapat dataset berjumlah 5 juta sampel. Dengan proporsi pembagian 80:20, maka data uji akan berjumlah 1 juta sampel. Tentu ini merupakan jumlah yang terlalu banyak karena untuk proses pengujian tidak dibutuhkan 1 juta sampel. Dalam kasus proses pengujian ini sebenarnya cukup menggunakan 1-2% data atau sebanyak 100.000 hingga 200.000 sampel saja.

<b>Bisa disimpulkan pembagian proporsi untuk *data train* dan *data test* sangat *relative* tergantung ketersedian dataset yang kita punya.</b>

Berikut adalah *code* untuk melakukan pembagian proporsi *data train* dan *data test*:
```
X = df.drop(["Fire Alarm"],axis =1)
y = df["Fire Alarm"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
```
Penjelasan:
- X adalah sebagai variabel *independent*
- y adalah sebagai variabel *dependent*
- *test_size* adalah proporsi untuk *data testnya*
- *random_state* berfungsi untuk mengontrol *random number generator* yang digunakan.

Melihat dimensi hasil pembagian pada *data train* dan *data test*:
```
print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')
```
![25](https://user-images.githubusercontent.com/88027268/203013445-03989581-8cbc-46d1-aca2-4e6790919661.jpg)
Gambar 18. Shape data train dan test

### Standarisasi
Setelah melakukan split data ke dalam *data train* dan *data test*, Selanjutnya adalah melakukan *data scaling*, Karena *value* pada tiap fitur memiliki angka yang signifikan dan hal itu dapat mengakibatkan model dari *machine learning* kita kesulitan dalam mencari polanya oleh karena itu dapat dilakukan penyeragaman value tersebut kedalam rentang -1 to 1 menggunakan *StandardScaler*.

*Code* untuk standarisasi:
```
scaling = StandardScaler()
X_train = scaling.fit_transform(X_train)
```

## Modeling

Selanjutnya akan lakukan *training* pada *data train* dan melakukan *predict* pada *data test* yang telah kita split sebelumnya. Algoritma yang akan kita gunakan adalah *Logistic Regression*.

*Logistic Regression* hampir mirip dengan *Linear Regression*, memiliki kemiripan yaitu sama-sama memiliki garis regresi. Salah satu yang membedakan adalah *Logistic Regression* digunakan untuk menentukan prediksi yang kita buat benar atau salah sedangkan *Linear Regression* digunakan untuk memprediksi nilai yang kontinu.

Kenapa saya menggunakan *Logistic Regression*? ini dikarenakan alogritma model ini sangat cocok ketika kasus *binary classification*.

Kelebihan *Logistic Regression*:
- Ketika terjadi *overfitting* pada algoritma *Logistic Regression* kita dapat menggunakan parameter regularisasi (L1 dan L2) untuk menghindari *overfitting*.
- Tidak memerlukan spesifikasi *device* yang tinggi untuk melakukan *training* pada algoritma *logistic regression*.
- bagus Ketika digunakan pada masalah *binary classification*.

Kekurangan *Logistic Regression*: 
- Pada data dengan *high dimensional* akan memiliki kecenderungan *overfitting*, salah satu cara untuk menghindari hal tersebut adalah dengan melakukan *regularization* akan tetapi hal tersebut dapat menambah kompleksitas dari model yang akan dihasilkan.
- Permasalahan *non-linear* sulit untuk diselesaikan menggunakan *logistic regression* dikarenakan algoritma tersebut memiliki *linear decision surface*.

Selanjutnya *define* sebuah *object* dari *Logistic Regression* dan beberapa parameter yang akan kita lakukan *hyperparameter tuning* untuk mendapatkan parameter terbaik pada *case* kita. 

```
model = LogisticRegression()
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
```
penjelasan dari setiap parameter:

- *solvers* adalah *optimizer* dari algoritma yang akan kita gunakan sebagai *method* untuk mencari *loss* terkecil dari *gradient descent*. terdapat beberapa *solvers* yang dapat digunakan seperti *newton-cg, lbfgs, liblinear, sag* dan *saga*. Secara *default* akan menggunakan *lbfgs*.
- *C_values* adalah nilai dari *regularization* yang kita gunakan, semakin kecil semakin memanandakan *regularization* yang kita lakukan semakin kuat. gunanya adalah untuk menghindari *overfitting*.
- *penalty* adalah bentuk *regularization* dari nilai yang telah kita *specify* sebelumnya pada *c_values*. terdapat beberapa jenis *penalty* yang dapat kita kombinasikan. seperti *l1, l2* dan *elasticnet*.

Selanjutnya adalah melakukan *hyperparameter tuning* menggunakan *GridSearchCV*. *GridSearchCV* adalah metode pemilihan kombinasi model dan *hyperparameter* dengan cara menguji coba satu persatu kombinasi dan melakukan validasi untuk setiap kombinasi. Tujuannya adalah menentukan kombinasi yang menghasilkan performa model terbaik yang dapat dipilih untuk dijadikan model untuk prediksi.

```
grid = dict(solver=solvers,penalty=penalty,C=c_values)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=3, scoring='recall')
grid_result = grid_search.fit(X, y)
```
penjelasan dari parameter yang kita gunakan:

- *estimator* adalah *object* dari algoritma yang telah kita *define* sebelumnya.
- *param_grid* adalah parameter dari *logistic regression* yang akan kita lakukan *hyperparameter tuning* untuk mencari kombinasi parameter yang terbaik
- *n_job*s adalah jumlah dari *processor* yang dipunya untuk gunakan nge *running jobs* tersebut secara *parallel*. *value* -1 berarti kita nge *running jobs* tersebut menggunakan seluruh dari *processor* yang dipunya.
- *cv* adalah *cross-validation generator* yang dimana kita menentukan ingin melakukan berapa kali percobaan secara acak pada dataset kita.

## Evaluation
Menampilkan hasil dari prediksi berupa *score* dari setiap kombinasi parameter yang kita lakukan *tuning* sebelumnya.

![29](https://user-images.githubusercontent.com/88027268/203013702-509ae813-bb35-4022-b4da-11dae76e044d.jpg)
Gambar 19. Score terbaik beserta parameternya

Kita mendapat *score* dari* metric *recall* yang sangat baik yaitu 96% namun *score* ini masih dapat kita improve menggunakan beberapa cara yaitu *feature importance*, melakukan *feature engineering* dan menggunakan parameter lebih banyak lagi untuk di *hyperparameter tuning*.

*metric* yang kita gunakan pada kasus ini adalah: *Recall*, karena hasil yang dinginkan adalah *False Negative(FN)* sekecil mungkin sehingga kita akan menggunakan *recall* sebagai *metric* patokan pada kasus ini. konsep dari *recall* ini sebagai berikut:

```math
Recall = \frac{True positive}{True positive + False Negative}
```

Penjelasan dari formula diatas:
- *True Positve* berarti model *machine learning* berhasil memprediksi bahwa ditempat tersebut terdapat asap dan memang terdapat asap di tempat tersebut.
- *False Negative* berarti model *machine learning* memprediksi bahwa ditempat tersebut tidak ada asap padahal ditempat tersebut terdapat asap. model *machine learning* ini dianggap gagal mendeteksi adanya asap. 

*Recall* adalah salah satu *metric* dari kasus klasifikasi yang lebih fokus untuk memprediksi asap dalam ruangan tersebut padahal tidak ada asap. dibandingkan model *machine learning* memprediksi tidak ada asap dalam ruangan tersebut padahal terdapat asap. tentu akan sangat fatal jika kita memilih model *machine learning* fokus memprediksi tidak ada asap dalam ruangan tersebut padahal terdapat asap, karena jika asap yang keluar adalah potensi dari kebakaran maka hal tersebut dapat berbahaya bagi orang yang ada didalam ruangan tersebut.

Kesimpulan: Menarik bahwa hanya dengan algoritma *machine learning* klasik sudah dapat memberikan *score* yang sangat tinggi. kita dapat mencoba menggunakan *neural network* untuk hasil yang lebih baik lagi karena algoritma *machine learning* klasik seperti *logistic regression* memiliki keterbatasan dan *neural network* dapat menutupi dari keterbatasan itu.

**---END---**
