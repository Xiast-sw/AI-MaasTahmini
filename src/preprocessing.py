import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def generate_data(n_samples=100, random_state=42):
    """Rastgele maaş verisi oluşturur."""
    np.random.seed(random_state)
    
    data = {
        'deneyim': np.random.randint(0, 21, size=n_samples),
        'pozisyon': np.random.choice(['Junior Developer', 'Mid Developer', 'Senior Developer'], size=n_samples),
        'egitim_duzeyi': np.random.choice(['Lise', 'Ön Lisans', 'Lisans'], size=n_samples),
        'yas': np.random.randint(20, 60, size=n_samples),
        'sehir': np.random.choice(['İstanbul', 'Ankara', 'İzmir'], size=n_samples),
        'maas': np.random.randint(3000, 15000, size=n_samples)
    }
    
    return pd.DataFrame(data)

def preprocess_data(df):
    """Kategorik değişkenleri sayısal hale getirir."""
    df_encoded = pd.get_dummies(df, columns=['pozisyon', 'egitim_duzeyi', 'sehir'], drop_first=True)
    return df_encoded

def prepare_features(df):
    """X ve y olarak ayırır."""
    X = df.drop('maas', axis=1)
    y = df['maas']
    return X, y

def create_polynomial_features(X, degree=2):
    """Polinom özellikler oluşturur."""
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    return X_poly, poly
