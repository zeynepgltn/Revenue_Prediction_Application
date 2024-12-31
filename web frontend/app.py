import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Modeli yükle
model = joblib.load('Lasso Regression_model.joblib')

# Başlık ve açıklama
st.title("Gelir Tahmin Uygulaması")
st.write("Bu uygulama, verilen özelliklere göre cafenizin gelirini tahmin eder.")

# Kullanıcıdan veri girişi al
Temperature = st.number_input("Sıcaklık:", min_value=-50, max_value=50, step=1)
Rain = st.number_input("Yağmur var mı? (0: Yok, 1: Var):", min_value=0, max_value=1, step=1)
Weekend = st.number_input("Hafta sonu mu? (0: Hayır, 1: Evet):", min_value=0, max_value=1, step=1)
Promotion = st.number_input("Promosyon var mı? (0: Yok, 1: Var):", min_value=0, max_value=1, step=1)
CafeCategory = st.selectbox("Cafe Kategorisi", ["A", "B", "C"])

# Kategorik veriyi One-Hot Encoding formatında işleme
category_mapping = {"A": 0, "B": 1, "C": 2}
cafe_category_index = category_mapping[CafeCategory]

# CafeCategory'yi One-Hot Encoding'e dönüştür
one_hot_encoding = [0] * 3
one_hot_encoding[cafe_category_index] = 1

# Kullanıcıdan alınan diğer verilerle birleştir
input_data = np.array([*one_hot_encoding, Temperature, Rain, Weekend, Promotion])

# Modelinizin beklediği sütun sayısı
expected_features = 7

# Eksik sütunları sıfırlarla doldurma
if input_data.shape[0] < expected_features:
    missing_features = expected_features - input_data.shape[0]
    input_data = np.append(input_data, [0] * missing_features)

# Giriş verisini modelin beklediği formata dönüştür (1 satır, n sütun)
input_data = input_data.reshape(1, -1)

# Tahmin butonu
if st.button("Tahmin Et"):
    try:
        # Modelle tahmin yap
        prediction = model.predict(input_data)

        # Sonucu göster
        st.success(f"Tahmin edilen gelir: {prediction[0]:.2f} TL")
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")
