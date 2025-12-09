from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def build_model():
    """Linear Regression modeli oluşturur."""
    return LinearRegression()

def train_model(model, X_train, y_train):
    """Modeli eğitir."""
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Model performansını değerlendirir."""
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return {
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'predictions': y_pred
    }

def predict_salary(model, poly, user_data, feature_columns):
    """Kullanıcı verisiyle maaş tahmini yapar."""
    user_df = user_data.reindex(columns=feature_columns, fill_value=0)
    X_poly = poly.transform(user_df)
    prediction = model.predict(X_poly)
    return prediction[0]
