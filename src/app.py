import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import generate_data, preprocess_data, prepare_features, create_polynomial_features
from model import build_model, train_model

# Model eğitimi
df = generate_data(n_samples=100)
df_encoded = preprocess_data(df)
X, y = prepare_features(df_encoded)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_poly, poly = create_polynomial_features(X_train, degree=2)
model = build_model()
model = train_model(model, X_train_poly, y_train)

def predict_salary():
    """Kullanıcı girdisiyle maaş tahmini yapar."""
    try:
        deneyim = int(entry_deneyim.get())
        pozisyon = combo_pozisyon.get()
        egitim = combo_egitim.get()
        yas = int(entry_yas.get())
        sehir = combo_sehir.get()

        user_data = pd.DataFrame({
            'deneyim': [deneyim],
            'pozisyon': [pozisyon],
            'egitim_duzeyi': [egitim],
            'yas': [yas],
            'sehir': [sehir]
        })

        user_encoded = pd.get_dummies(user_data, columns=['pozisyon', 'egitim_duzeyi', 'sehir'], drop_first=True)
        user_encoded = user_encoded.reindex(columns=X_train.columns, fill_value=0)
        
        prediction = model.predict(poly.transform(user_encoded))
        label_sonuc.config(text=f"Tahmin Edilen Maaş: {prediction[0]:,.2f} TL")
    except ValueError:
        label_sonuc.config(text="Lütfen geçerli değerler girin!")

# Tkinter GUI
root = tk.Tk()
root.title("Salary Prediction AI")
root.geometry("350x400")

# Başlık
ttk.Label(root, text="Maaş Tahmin Aracı", font=('Arial', 16, 'bold')).pack(pady=10)

# Deneyim
ttk.Label(root, text="Deneyim (Yıl):").pack()
entry_deneyim = ttk.Entry(root)
entry_deneyim.pack(pady=5)

# Pozisyon
ttk.Label(root, text="Pozisyon:").pack()
combo_pozisyon = ttk.Combobox(root, values=['Junior Developer', 'Mid Developer', 'Senior Developer'])
combo_pozisyon.pack(pady=5)

# Eğitim
ttk.Label(root, text="Eğitim Düzeyi:").pack()
combo_egitim = ttk.Combobox(root, values=['Lise', 'Ön Lisans', 'Lisans'])
combo_egitim.pack(pady=5)

# Yaş
ttk.Label(root, text="Yaş:").pack()
entry_yas = ttk.Entry(root)
entry_yas.pack(pady=5)

# Şehir
ttk.Label(root, text="Şehir:").pack()
combo_sehir = ttk.Combobox(root, values=['İstanbul', 'Ankara', 'İzmir'])
combo_sehir.pack(pady=5)

# Buton
ttk.Button(root, text="Maaş Tahmini Yap", command=predict_salary).pack(pady=15)

# Sonuç
label_sonuc = ttk.Label(root, text="", font=('Arial', 12, 'bold'))
label_sonuc.pack(pady=10)

root.mainloop()
