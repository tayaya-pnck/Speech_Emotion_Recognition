# Speech-Emotion-Recognition
Proyek ini bertujuan untuk mengenali emosi manusia berdasarkan pola suara, menggunakan algoritma machine learning, khususnya dengan pendekatan deep learning berbasis CNN (Convolutional Neural Network). Proyek ini juga memanfaatkan teknik ekstraksi fitur MFCC (Mel-Frequency Cepstral Coefficients) untuk merepresentasikan karakteristik penting dari data suara.


# Business-Understanding
- Problem Statement : Bagaimana membangun model yang dapat mengenali emosi dari data suara secara akurat menggunakan metode yang efisien?
- Goal : Mengembangkan sistem Speech Emotion Recognition (SER) berbasis Convolutional Neural Network (CNN)
- Solution Statement : Merancang pipeline lengkap untuk pengolahan suara, mulai dari pengumpulan data, noise reduction, preprocessing, hingga data augmentation untuk mengatasi ketidakseimbangan data.


# Data-Understanding
Link Dataset : https://github.com/CheyneyComputerScience/CREMA-D.git

**CREMA-D adalah sebuah dataset yang terdiri dari 7.442 klip asli yang dihasilkan oleh 91 aktor. Klip-klip ini berasal dari 48 aktor pria dan 43 aktor wanita dengan rentang usia antara 20 hingga 74 tahun, yang berasal dari berbagai ras dan etnis (Afrika-Amerika, Asia, Kaukasia, Hispanik, dan Tidak Ditentukan).**

Exploratory Data Analysis

**Audi File**

https://github.com/user-attachments/assets/45e5e12e-9251-4a5e-8fd0-4bd1df0d5830

**Waveplot**

![image](https://github.com/user-attachments/assets/920322f6-e6a9-4b06-bbe9-ba8b50eab35f)

**Noise Reduction**

![image](https://github.com/user-attachments/assets/602ad0d9-c8ca-468a-a19b-58a5d849a33c)

**Count of Emotions**

![image](https://github.com/user-attachments/assets/1f90694c-8af4-4f46-894e-149868d45869)

# Data Preprocessing
mengubah data audio menjadi spectrogram
![image](https://github.com/user-attachments/assets/92b45821-550e-47c8-ae25-ee7faeb69667)
![image](https://github.com/user-attachments/assets/18b15731-2bde-4d54-8d3f-e9a7b621558b)

**Data Augmentation**
```def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

# taking any example and checking for techniques.
path = np.array(crema_df.Path[crema_df.Emotions == emotion])[9]
data, sample_rate = librosa.load(path)
```


# Modelling
CNN Architecture Model Summary

![image](https://github.com/user-attachments/assets/7ea59d83-60cb-4c1e-b2dc-4d393cf47f5d)

# Evaluation
Training accuracy and loss

![image](https://github.com/user-attachments/assets/fea57104-849c-44f0-8dee-a87fce25ba9f)

# Predicted Class
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted Labels</th>
      <th>Actual Labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>happy</td>
      <td>happy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>happy</td>
      <td>angry</td>
    </tr>
    <tr>
      <th>2</th>
      <td>disgust</td>
      <td>angry</td>
    </tr>
    <tr>
      <th>3</th>
      <td>happy</td>
      <td>fear</td>
    </tr>
    <tr>
      <th>4</th>
      <td>disgust</td>
      <td>disgust</td>
    </tr>
    <tr>
      <th>5</th>
      <td>fear</td>
      <td>sad</td>
    </tr>
    <tr>
      <th>6</th>
      <td>fear</td>
      <td>neutral</td>
    </tr>
    <tr>
      <th>7</th>
      <td>fear</td>
      <td>fear</td>
    </tr>
    <tr>
      <th>8</th>
      <td>sad</td>
      <td>happy</td>
    </tr>
    <tr>
      <th>9</th>
      <td>fear</td>
      <td>sad</td>
    </tr>
  </tbody>
</table>
</div>

# Kesimpulan
Solusi Efektif
Model CNN yang dibangun menunjukkan performa yang baik dalam mengenali emosi dari data suara. Teknik preprocessing dan ekstraksi fitur MFCC terbukti efektif dalam meningkatkan akurasi.

**Perluasan Penelitian**

- Mengeksplorasi fitur lain seperti Mel Spectrogram untuk meningkatkan representasi data suara.
- Menggunakan transfer learning untuk meningkatkan generalisasi model.
- Memperluas dataset dengan berbagai bahasa, aksen, dan situasi rekaman.

**Hasil Analisa Simulasi**
Model memiliki akurasi rata-rata sebesar 67%. Penggunaan CNN dan MFCC memberikan kontribusi signifikan dalam performa pengenalan emosi.
![image](https://github.com/user-attachments/assets/5ae64960-f9ea-4153-b370-c2a365cf559c)

# Reference
Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

Jurafsky, D., & Martin, J. H. (2018). Speech and Language Processing. Pearson.
