# Smoke detection

![4](https://user-images.githubusercontent.com/88027268/203010892-b35284b1-4779-4764-923b-6b94c065a446.jpeg)
Gambar 1. Illustrasi asap rokok

## Domain Proyek

Yang melatar belakangi pembuatan project ini adalah karena indonesia menjadi negara dengan mayoritas masyarakatnya perokok aktif dan banyak tempat yang seharusnya dilarang merokok namun seringkali perokok tetap membakar rokoknya ditempat tersebut. 

- banyak dari masyarakat indonesia sering melanggar peraturan dilarang merokok diarea tersebut mengakibatkan orang yang berada di area tersebut dapat terkena dampaknya juga yang dihasilkan dari perokok itu sendiri [(Joaquin Barnoya and Stanton A. Glantz, 2005)](https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.104.492215). orang-orang yang berada di *no smoking area* mengasumsikan kalo diarea tersebut tidak ada perokok sehingga orang-orang yang tidak ingin terkena asap rokok memilih tempat tersebut. selain itu model *machine learning* ini juga dapat digunakan untuk mendeteksi asap kebakaran jika didalam rumah atau ruangan terdapat asap. salah satu solusi yang dapat saya berikan adalah dengan cara membuat sebuah model *machine learning* untuk mendeteksi adanya asap ditempat tersebut, yang nantinya model *machine learning* tersebut dapat dikembangkan ke perangkat *IoT* agar dapat memfasilitasi diberbagai tempat. jika perangkat *IoT* tersebut mendeteksi asap nantinya terdapat sebuah pemberitahuan seperti *alarm*. 
 
## Business Understanding

Pada bagian ini, akan menjelaskan proses klarifikasi masalah.

<img width="960" alt="1" src="https://user-images.githubusercontent.com/88027268/203011051-26a43e4f-4235-467b-9dda-4213f07b131d.png">
Gambar 2. Illustrasi asap rokok

### Problem Statements

- Dari serangkaian fitur yang terdapat pada dataset, fitur apa yang paling berkorelasi terhadap *fire alarm*?
- Apakah model *machine learning* yang telah dibuat nantinya dapat memberikan score yang tinggi?

### Goals

- Mengetahui fitur yang paling berkorelasi dengan *fire alarm*.
- Membuat sebuah model *machine learning* yang dapat mendeteksi asap seakurat mungkin dari fitur-fitur yang ada.

    ### Solution statements
    - dapat menggunakan satu algoritma *machine learning* yaitu *Logistic Regression* dan dari algoritma *machine learning* tersebut akan diimprove recall scorenya menggunakan hyperparameter tuning *GridSearchCV*.

## Data Understanding
Link download dataset: https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset  

### Variabel-variabel pada dataset Smoke detection adalah sebagai berikut:
- *UTC: Time when experiment was performed*
- *Temperature[C]: Temperature of surroundings, measured in celcius*
- *Humidity[%]: Air humidity during the experiment*
- *TVOC[ppb]: Total Volatile Organic Compounds, measured in ppb (parts per billion)*
- *eCO2[ppm]: CO2 equivalent concentration, measured in ppm (parts per million)*
- *Raw H2: The amount of Raw Hydrogen [Raw Molecular Hydrogen; not compensated (Bias, Temperature etc.)] present in surroundings*
- *Raw Ethanol: The amount of Raw Ethanol present in surroundings*
- *Pressure[hPa]: Air pressure, Measured in hPa*
- *PM1.0: Paticulate matter of diameter less than 1.0 micrometer*
- *PM2.5: Paticulate matter of diameter less than 2.5 micrometer*
- *NC0.5: Concentration of particulate matter of diameter less than 0.5 micrometer*
- *NC1.0: Concentration of particulate matter of diameter less than 1.0 micrometer*
- *NC2.5: Concentration of particulate matter of diameter less than 2.5 micrometer*
- *CNT: Sample Count. Fire Alarm(Reality) If fire was present then value is 1 else it is 0*
- *Fire Alarm: 1 means Positive and 0 means Not Positive*

### Data Loading:
Langkah pertama import *library* yang dibutuhkan untuk kasus kali ini:

*Library* yang akan diimport adalah *library* yang berhubungan untuk memanipulasi data, data visualisasi, *preprocessing*, algoritma *machine learning*, evaluasi dan mematikan warning yang didapat setelah menjalankan code. 

Selanjutnya membaca dataset dan menampilkan 5 data teratas:
|   	| UTC        	| Temperature[C] 	| Humidity[%] 	| TVOC[ppb] 	| eCO2[ppm] 	| Raw H2 	| Raw Ethanol 	| Pressure[hPa] 	| PM1.0 	| PM2.5 	| NC0.5 	| NC1.0 	| NC2.5 	| CNT 	|
|---	|------------	|----------------	|-------------	|-----------	|-----------	|--------	|-------------	|---------------	|-------	|-------	|-------	|-------	|-------	|-----	|
| 0 	| 1654733331 	| 20.0           	| 57.36       	| 0         	| 400       	| 12306  	| 18520       	| 939.735       	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 0   	|
| 1 	| 1654733332 	| 20.015         	| 56.67       	| 0         	| 400       	| 12345  	| 18651       	| 939.744       	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 1   	|
| 2 	| 1654733333 	| 20.029         	| 55.96       	| 0         	| 400       	| 12374  	| 18764       	| 939.738       	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 2   	|
| 3 	| 1654733334 	| 20.044         	| 55.28       	| 0         	| 400       	| 12390  	| 18849       	| 939.736       	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 3   	|
| 4 	| 1654733335 	| 20.059         	| 54.69       	| 0         	| 400       	| 12403  	| 18921       	| 939.744       	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 0.0   	| 4   	|

Kita dapat menghapus fitur 'Unnamed: 0' Kemudian melihat dimensi dari dataset ini. lalu melihat dimensi dari dataset kita.

> Dataset ini terdiri dari 62630 data dan 10 kolom.

### Exploratory Data Analysis - Deskripsi Variabel:
*Exploratory data analysis* atau sering disingkat EDA merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. Teknik ini biasanya menggunakan bantuan statistik dan representasi grafis atau visualisasi.

Cakupan proses EDA sangat luas. Namun, secara umum, Anda dapat melakukan proses EDA untuk menjawab beberapa pertanyaan berikut:

1. Apa saja jenis variabel pada dataset?
2. Bagaimana distribusi variabel dalam dataset?
3. Apakah ada *missing value*?
4. Apakah ada fitur yang tidak berguna (*redundant*)?
5. Bagaimana korelasi antara fitur dan target?

| Kolom          	| Type    	|
|----------------	|---------	|
| UTC            	| int64   	|
| Temperature[C] 	| float64 	|
| Humidity[%]    	| float64 	|
| TVOC[ppb]      	| int64   	|
| eCO2[ppm]      	| int64   	|
| Raw H2         	| int64   	|
| Raw Ethanol    	| int64   	|
| Pressure[hPa]  	| float64 	|
| PM1.0          	| float64 	|
| PM2.5          	| float64 	|
| NC0.5          	| float64 	|
| NC1.0          	| float64 	|
| NC2.5          	| float64 	|
| CNT            	| int64   	|
| Fire Alarm     	| int64   	|

> Seluruh fitur pada dataset kita bertipe numeric.

Melakukan pengecekkan deskripsi statistik pada dataset untuk mengetahui apakah terdapat anomali:

|       	| UTC                	| Temperature[C]     	| Humidity[%]        	| TVOC[ppb]          	| eCO2[ppm]          	| Raw H2             	| Raw Ethanol        	| Pressure[hPa]      	| PM1.0              	| PM2.5              	| NC0.5              	| NC1.0              	| NC2.5              	| CNT               	| Fire Alarm         	|
|-------	|--------------------	|--------------------	|--------------------	|--------------------	|--------------------	|--------------------	|--------------------	|--------------------	|--------------------	|--------------------	|--------------------	|--------------------	|--------------------	|-------------------	|--------------------	|
| count 	| 62630.0            	| 62630.0            	| 62630.0            	| 62630.0            	| 62630.0            	| 62630.0            	| 62630.0            	| 62630.0            	| 62630.0            	| 62630.0            	| 62630.0            	| 62630.0            	| 62630.0            	| 62630.0           	| 62630.0            	|
| mean  	| 1654792066.1842728 	| 15.97042358294747  	| 48.539499441162384 	| 1942.0575283410506 	| 670.0210442280057  	| 12942.453935813508 	| 19754.257911543988 	| 938.6276494651127  	| 100.59430911703656 	| 184.46777023790517 	| 491.46360769599227 	| 203.58648749800412 	| 80.04904232795785  	| 10511.38615679387 	| 0.7146255787961041 	|
| std   	| 110002.48807802147 	| 14.359576152610806 	| 8.865367089675287  	| 7811.589055386021  	| 1905.8854393506067 	| 272.4643052353358  	| 609.5131564626391  	| 1.3313435732418013 	| 922.5242445867349  	| 1976.3056148260875 	| 4265.661251435324  	| 2214.738555639472  	| 1083.3831887688002 	| 7597.870997377545 	| 0.4515961881806646 	|
| min   	| 1654712187.0       	| -22.01             	| 10.74              	| 0.0                	| 400.0              	| 10668.0            	| 15317.0            	| 930.852            	| 0.0                	| 0.0                	| 0.0                	| 0.0                	| 0.0                	| 0.0               	| 0.0                	|
| 25%   	| 1654743244.25      	| 10.99425           	| 47.53              	| 130.0              	| 400.0              	| 12830.0            	| 19435.0            	| 938.7              	| 1.28               	| 1.34               	| 8.82               	| 1.384              	| 0.033              	| 3625.25           	| 0.0                	|
| 50%   	| 1654761919.5       	| 20.13              	| 50.15              	| 981.0              	| 400.0              	| 12924.0            	| 19501.0            	| 938.816            	| 1.81               	| 1.88               	| 12.45              	| 1.943              	| 0.044              	| 9336.0            	| 1.0                	|
| 75%   	| 1654777576.75      	| 25.4095            	| 53.24              	| 1189.0             	| 438.0              	| 13109.0            	| 20078.0            	| 939.418            	| 2.09               	| 2.18               	| 14.42              	| 2.249              	| 0.051              	| 17164.75          	| 1.0                	|
| max   	| 1655130051.0       	| 59.93              	| 75.2               	| 60000.0            	| 60000.0            	| 13803.0            	| 21410.0            	| 939.861            	| 14333.69           	| 45432.26           	| 61482.03           	| 51914.68           	| 30026.438          	| 24993.0           	| 1.0                	|

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

|                	| Total 	| Percentage of Missing Values 	|
|----------------	|-------	|------------------------------	|
| UTC            	| 0     	| 0.0                          	|
| Temperature[C] 	| 0     	| 0.0                          	|
| Humidity[%]    	| 0     	| 0.0                          	|
| TVOC[ppb]      	| 0     	| 0.0                          	|
| eCO2[ppm]      	| 0     	| 0.0                          	|
| Raw H2         	| 0     	| 0.0                          	|
| Raw Ethanol    	| 0     	| 0.0                          	|
| Pressure[hPa]  	| 0     	| 0.0                          	|
| PM1.0          	| 0     	| 0.0                          	|
| PM2.5          	| 0     	| 0.0                          	|
| NC0.5          	| 0     	| 0.0                          	|
| NC1.0          	| 0     	| 0.0                          	|
| NC2.5          	| 0     	| 0.0                          	|
| CNT            	| 0     	| 0.0                          	|
| Fire Alarm     	| 0     	| 0.0                          	|

> Tidak terdapat missing value pada dataset.

![8](https://user-images.githubusercontent.com/88027268/203012132-ca1b43c3-1202-4ecb-b1f6-f0d366029902.jpg)
![9](https://user-images.githubusercontent.com/88027268/203012160-f8eb630f-6770-4f48-baf0-7fae887c03b1.jpg)
![10](https://user-images.githubusercontent.com/88027268/203012189-b189c06d-6701-40bd-b466-e4561e06a474.jpg)
![11](https://user-images.githubusercontent.com/88027268/203012218-724f7176-142e-428f-b095-fd845096f658.jpg)
![12](https://user-images.githubusercontent.com/88027268/203012254-f7c28332-b38a-4507-8ca1-3517aa0071b5.jpg)
Gambar 3. Outlier pada dataset

- Outlier yang terjadi pada fitur temperature merupakan nilai yang wajar karena temperature dapat berubah secara signifikan tiap waktu.
- Outlier yang terjadi pada fitur Raw Ethanol, TVOC dan Raw H2 merupakan nilai yang wajar karena dapat berubah secara signifikan tiap saat tergantung pada gas yang dapat dideteksi oleh sistem
- Outlier yang terjadi pada fitur eCO2 merupakan turunan dari nilai TVOC sehingga outlier yang dihasilkan seperti yang terjadi pada fitur TVOC
- Outlier yang terjadi pada fitur Humidity merupakan nilai yang wajar karena dapat berubah secara signifikan di suatu kondisi dan bergantung pada temperature
- Outlier yang terjadi pada fitur Pressure merupakan nilai yang wajar karena bergantung pada temperature saat itu
- Outlier yang terjadi pada fitur PM1.0 dan PM2.5 merupakan nilai yang wajar karena yang dapat menghasilkan PM1.0 dan PM2.5 bukan cuman asap melainkan seperti tempat konstruksi dan jalan tak beraspal.
- Outlier yang terjadi pada fitur NC0.5, NC1.0 dan NC2.5 merupakan nilai yang wajar karena sama halnya pada fitur PM1.0 dan PM2.5

### Exploratory Data Analysis - Univariate Analysis:
Melihat proporsi dari nilai variabel target pada dataset:

![13](https://user-images.githubusercontent.com/88027268/203012313-e90c3936-0252-4286-88e2-24e833ddd769.png)
Gambar 4. Proporsi pada fitur target

> Terjadi data *imbalance* pada variabel target. Permasalahan tersebut dapat diselesaikan dengan cara melakukan pendekatan *undersampling* dan *uppersampling* pada variabel target atau dengan cara lain yaitu memilih *metric* yang tepat seperti *recall*, *precision* dan *F1 Score*. Tidak dapat menggunakan *metric* *accuracy* karena dapat menyebabkan bias pada saat *scoring* model.

### Exploratory Data Analysis - Multivariate Analysis:
Melihat *kernel density estimation* (KDE) plot dari tiap variabel:

Melihat hubungan humidity dengan variabel target menggunakan KDE
![14](https://user-images.githubusercontent.com/88027268/203012378-5e964184-d1d6-45c9-a4b2-6c03ff518b0b.png)

Gambar 5. KDE Humidity vs Fire Alarm
> Selama *experiment* kemungkinan *fire alarm* ditempat yang memiliki kelembapan cukup tinggi. karena puncak *density* pada visualisasi diatas mencakup kelembapan > 40%


Melihat hubungan temperature dengan variabel target menggunakan KDE
![15](https://user-images.githubusercontent.com/88027268/203012429-d73bc0cc-2430-48be-9b13-d85fb0520b05.png)

Gambar 6. KDE Temperature vs Fire Alarm
> Selama *experiment* kemungkinan *fire alarm* ditempat yang memiliki *temperature* sekitar 20 celcius

Melihat hubungan Pressure dengan variabel target menggunakan KDE
![16](https://user-images.githubusercontent.com/88027268/203012552-edf1b456-3842-4f82-99d6-9fc148f003fd.png)

Gambar 7. KDE Pressure vs Fire Alarm
> Berdasarkan puncak *density* dari visualisasi diatas dikita lihat bahwa semakin tinggi *Pressure* maka kemungkinan untuk *fire alarm* berbunyi semakin besar juga

Melihat hubungan Raw H2 dengan variabel target menggunakan KDE
![17](https://user-images.githubusercontent.com/88027268/203012750-ae9d8d1d-246f-4b93-9de5-87a05891b00c.png)

Gambar 8. KDE Raw H2 vs Fire Alarm
> Berdasarkan puncak *density* dari visualisasi diatas dapat dilihat bahwa *Raw H2* pada *yes fire* dan *no fire* memiliki rentang yang serupa yaitu 12500 - 1340

Melihat hubungan Raw Ethanol dengan variabel target menggunakan KDE
![18](https://user-images.githubusercontent.com/88027268/203012805-be948f4d-9b96-4978-92cd-d35da0cdb0ce.png)

Gambar 9. KDE Raw Ethanol vs Fire Alarm
> Berdasarkan puncak *density* dari visualisasi diatas dapat dilihat bahwa *yes fire* memiliki kecenderungan berada di jumlah *Raw Ethanol* sekitar 19500 - 20500 dan *no fire* memiliki kecenderungan berada di jumlah *Raw Ethanol* sekitar 20000 - 21000

Melihat korelasi antara tiap fitur:

![19](https://user-images.githubusercontent.com/88027268/203012863-b8347aed-42c5-486e-8851-e547ef663457.png)

Gambar 10. Korelasi ditiap fitur pada dataset
- Semua kolom *'PM's* dan *'NC's* memiliki korelasi yang tinggi dengan sesama kolom tersebut
- Tidak ada fitur yang berkorelasi tinggi dengan fitur target. *Humidity*, *Pressure* dan *Raw H2* adalah fitur yang memiliki korelasi positif namun tidak tinggi dan sisanya adalah fitur yang berkorelasi rendah dengan fitur targetnya.

## Data Preparation
![22](https://user-images.githubusercontent.com/88027268/203012905-826bd461-2a07-43c8-b63b-129c621f49aa.jpg)

Gambar 11. Illustrasi Data Preparation
Pada bagian ini terdapat empat tahap persiapan data, yaitu:

- *Feature Selection*
- Reduksi dimensi dengan *Principal Component Analysi*s (PCA)
- Pembagian dataset dengan fungsi train_test_split dari library sklearn
- Standarisasi

### Feature Selection
*Feature selection* adalah proses mengurangi jumlah fitur atau variabel input dengan memilih fitur-fitur yang dianggap paling relevan terhadap model.

- Fitur yang sangat didominasi oleh satu nilai saja akan dibuang pada tahap ini. karena fitur yang didominasi satu nilai saja tidak berarti untuk *machine learning*.

> Tidak ada fitur dengan satu nilai saja maka tidak ada fitur yang harus dibuang. 

- menghapus kolom *UTC* karena tidak berpengaruh pada model *machine learning* sehingga hal ini dapat memudahkan *machine learning* dalam pencari pola dari dataset.

### Principal Component Analysis (PCA)
*PCA* bekerja menggunakan metode aljabar linier. Ia mengasumsikan bahwa sekumpulan data pada arah dengan *varians* terbesar merupakan yang paling penting (utama). *PCA* umumnya digunakan ketika variabel dalam data memiliki korelasi yang tinggi. Korelasi positif yang tinggi ini menunjukkan data yang berulang atau *redundant*.

Fitur yang akan dilakukan PCA adalah *PM1.0, PM2.5, NC0.5, NC1.0 dan NC2.5*. Karena fitur tersebut saling berkorelasi positif dan cukup tinggi.

- parameter *n_components* adalah jumlah komponen atau dimensi seperti dikasus ini 5 yaitu *'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0' dan 'NC2.5'*
- parameter *random_state* berfungsi untuk mengontrol *random number generator* yang digunakan. Parameter ini berupa bilangan *integer* dan nilainya bebas. Pada kasus ini, Menerapkan *random_state = 42*. Berapa pun nilai *integer* yang ditentukan. selama itu bilangan *integer*, ia akan memberikan hasil yang sama setiap kali dilakukan pemanggilan fungsi.

> Hasil dari *PCA* adalah, 0.9% informasi pada kelima fitur *'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0' dan 'NC2.5'* terdapat pada *Principal Component* pertama. Sedangkan sisanya, sebesar 0.1%, 0.0%, 0.0% dan 0.0% terdapat pada *Principal Component* kedua, ketiga, keempat dan kelima.

Dari hasil diatas dapat dipertahankan pada *Principal Component* pertama saja untuk menggantikan kelima fitur yang telah direduksi sebelumnya lalu kita beri nama fitur ini dengan *'dimension'*

### Pembagian dataset

Selanjutnya dapat melakukan *split data* ke dalam beberapa bagian yaitu *data train* dan *data test*. kita melakukan beberapa transformasi pada *data train* sedangkan *data test* kita gunakan sebagai data uji yang diasumsikan adalah data baru sama halnya nanti di *production*. karna *data test* adalah data baru sehingga kita tidak diperkenankan untuk melakukan transformasi apapun, jika kita melakukan transformasi pada *data test* itu akan menimbulkan masalah baru yaitu *data leakage* dan dapat menyebabkan model *machine learning* kita *bias*.

proporsi pembagian data latih dan uji biasanya adalah 80:20. Bahwa proporsi ini hanya kebiasaan umum saja. Tujuan dari data uji adalah untuk untuk mengukur kinerja model pada data baru. Jadi, jika dataset yang kita miliki berukuran sangat kecil, misalnya kurang dari 1.000 sampel, maka pembagian 80:20 ini cukup *ideal*. Namun, jika memiliki dataset berukuran besar, kita perlu memikirkan strategi pembagian dataset lain agar proporsi data uji tidak terlalu banyak.

Sebagai contoh, Terdapat dataset berjumlah 5 juta sampel. Dengan proporsi pembagian 80:20, maka data uji akan berjumlah 1 juta sampel. Tentu ini merupakan jumlah yang terlalu banyak karena untuk proses pengujian tidak dibutuhkan 1 juta sampel. Dalam kasus proses pengujian ini sebenarnya cukup menggunakan 1-2% data atau sebanyak 100.000 hingga 200.000 sampel saja.

<b>Bisa disimpulkan pembagian proporsi untuk *data train* dan *data test* sangat *relative* tergantung ketersedian dataset yang kita punya.</b>

Penjelasan:
- X adalah sebagai variabel *independent*
- y adalah sebagai variabel *dependent*
- *test_size* adalah proporsi untuk *data testnya*
- *random_state* berfungsi untuk mengontrol *random number generator* yang digunakan.

Melihat dimensi hasil pembagian pada *data train* dan *data test*:
| Total Data 	| 62630 	|
| Train Data 	| 56367 	|
| Test Data  	| 6263  	|

### Standarisasi
Setelah melakukan split data ke dalam *data train* dan *data test*, Selanjutnya adalah melakukan *data scaling*, Karena *value* pada tiap fitur memiliki angka yang signifikan dan hal itu dapat mengakibatkan model dari *machine learning* kita kesulitan dalam mencari polanya oleh karena itu dapat dilakukan penyeragaman value tersebut kedalam rentang -1 to 1 menggunakan *StandardScaler*.

## Modeling

Selanjutnya akan lakukan *training* pada *data train* dan melakukan *predict* pada *data test* yang telah kita split sebelumnya. Algoritma yang akan kita gunakan adalah *Logistic Regression*.

*Logistic Regression* hampir mirip dengan *Linear Regression*, memiliki kemiripan yaitu sama-sama memiliki garis regresi. Salah satu yang membedakan adalah *Logistic Regression* digunakan untuk menentukan prediksi yang kita buat benar atau salah sedangkan *Linear Regression* digunakan untuk memprediksi nilai yang kontinu.

Kelebihan *Logistic Regression*:
- Ketika terjadi *overfitting* pada algoritma *Logistic Regression* kita dapat menggunakan parameter regularisasi (L1 dan L2) untuk menghindari *overfitting*.
- Tidak memerlukan spesifikasi *device* yang tinggi untuk melakukan *training* pada algoritma *logistic regression*.
- bagus Ketika digunakan pada masalah *binary classification*.

Kekurangan *Logistic Regression*: 
- Pada data dengan *high dimensional* akan memiliki kecenderungan *overfitting*, salah satu cara untuk menghindari hal tersebut adalah dengan melakukan *regularization* akan tetapi hal tersebut dapat menambah kompleksitas dari model yang akan dihasilkan.
- Permasalahan *non-linear* sulit untuk diselesaikan menggunakan *logistic regression* dikarenakan algoritma tersebut memiliki *linear decision surface*.

Selanjutnya *define* sebuah *object* dari *Logistic Regression* dan beberapa parameter yang akan kita lakukan *hyperparameter tuning* untuk mendapatkan parameter terbaik pada *case* kita. 

penjelasan dari setiap parameter:

- *solvers* adalah *optimizer* dari algoritma yang akan kita gunakan sebagai *method* untuk mencari *loss* terkecil dari *gradient descent*. terdapat beberapa *solvers* yang dapat digunakan seperti *newton-cg, lbfgs, liblinear, sag* dan *saga*. Secara *default* akan menggunakan *lbfgs*.
- *C_values* adalah nilai dari *regularization* yang kita gunakan, semakin kecil semakin memanandakan *regularization* yang kita lakukan semakin kuat. gunanya adalah untuk menghindari *overfitting*.
- *penalty* adalah bentuk *regularization* dari nilai yang telah kita *specify* sebelumnya pada *c_values*. terdapat beberapa jenis *penalty* yang dapat kita kombinasikan. seperti *l1, l2* dan *elasticnet*.

Selanjutnya adalah melakukan *hyperparameter tuning* menggunakan *GridSearchCV*. *GridSearchCV* adalah metode pemilihan kombinasi model dan *hyperparameter* dengan cara menguji coba satu persatu kombinasi dan melakukan validasi untuk setiap kombinasi. Tujuannya adalah menentukan kombinasi yang menghasilkan performa model terbaik yang dapat dipilih untuk dijadikan model untuk prediksi.

penjelasan dari parameter yang kita gunakan:

- *estimator* adalah *object* dari algoritma yang telah kita *define* sebelumnya.
- *param_grid* adalah parameter dari *logistic regression* yang akan kita lakukan *hyperparameter tuning* untuk mencari kombinasi parameter yang terbaik
- *n_job*s adalah jumlah dari *processor* yang dipunya untuk gunakan nge *running jobs* tersebut secara *parallel*. *value* -1 berarti kita nge *running jobs* tersebut menggunakan seluruh dari *processor* yang dipunya.
- *cv* adalah *cross-validation generator* yang dimana kita menentukan ingin melakukan berapa kali percobaan secara acak pada dataset kita.

## Evaluation
Menampilkan hasil dari prediksi berupa *score* tertinggi dari kombinasi parameter terbaik yang telah dilakukan *tuning*.

| names   	| values    	|
|---------	|-----------	|
| C       	| 100       	|
| penalty 	| l2        	|
| solver  	| liblinear 	|


|   	| Precision          	| Recall             	| F1-score           	| Support 	|
|---	|--------------------	|--------------------	|--------------------	|---------	|
| 0 	| 0.9385026737967914 	| 0.9892897406989853 	| 0.9632272228320526 	| 1774.0  	|
| 1 	| 0.9956749374004097 	| 0.974381822232123  	| 0.9849133078135555 	| 4489.0  	|

Kita mendapat *score* dari* metric *recall* yang sangat baik yaitu 97% namun *score* ini masih dapat kita improve menggunakan beberapa cara yaitu *feature importance*, melakukan *feature engineering* dan menggunakan parameter lebih banyak lagi untuk di *hyperparameter tuning*.

*metric* yang kita gunakan pada kasus ini adalah: *Recall*, karena hasil yang dinginkan adalah *False Negative(FN)* sekecil mungkin sehingga kita akan menggunakan *recall* sebagai *metric* patokan pada kasus ini. konsep dari *recall* ini sebagai berikut:

```math
Recall = \frac{True positive}{True positive + False Negative}
```

Penjelasan dari formula diatas:
- *True Positve* berarti model *machine learning* berhasil memprediksi bahwa ditempat tersebut terdapat asap dan memang terdapat asap di tempat tersebut.
- *False Negative* berarti model *machine learning* memprediksi bahwa ditempat tersebut tidak ada asap padahal ditempat tersebut terdapat asap. model *machine learning* ini dianggap gagal mendeteksi adanya asap. 

![31](https://user-images.githubusercontent.com/88027268/203662066-12b44ac6-eabe-49cc-9f6d-3dd3dd7cf6f7.png)

Gambar 12. heatmap confusion matrix

*Recall* adalah salah satu *metric* dari kasus klasifikasi yang lebih fokus untuk memprediksi asap dalam ruangan tersebut padahal tidak ada asap. dibandingkan model *machine learning* memprediksi tidak ada asap dalam ruangan tersebut padahal terdapat asap. tentu akan sangat fatal jika kita memilih model *machine learning* fokus memprediksi tidak ada asap dalam ruangan tersebut padahal terdapat asap, karena jika asap yang keluar adalah potensi dari kebakaran maka hal tersebut dapat berbahaya bagi orang yang ada didalam ruangan tersebut.

Kesimpulan: Projek ini berhasil mengetahui bahwa fitur dari *Humadity* dan *CNT* adalah fitur yang paling berkorelasi dengan *Fire Alarm*. Menarik bahwa hanya dengan algoritma *machine learning* klasik sudah dapat memberikan *score* yang sangat tinggi. kita dapat mencoba menggunakan *neural network* untuk hasil yang lebih baik lagi karena algoritma *machine learning* klasik seperti *logistic regression* memiliki keterbatasan dan *neural network* dapat menutupi dari keterbatasan itu.

**---END---**
