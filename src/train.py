import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocessing import generate_data, preprocess_data, prepare_features, create_polynomial_features
from model import build_model, train_model, evaluate_model

# 1) Veri oluştur
print("Generating data...")
df = generate_data(n_samples=100)
print(f"Dataset shape: {df.shape}")

# 2) Veriyi işle
df_encoded = preprocess_data(df)
X, y = prepare_features(df_encoded)

# 3) Train/Test ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# 4) Polinom özellikler
X_train_poly, poly = create_polynomial_features(X_train, degree=2)
X_test_poly = poly.transform(X_test)

# 5) Model oluştur ve eğit
print("\nTraining model...")
model = build_model()
model = train_model(model, X_train_poly, y_train)

# 6) Değerlendir
results = evaluate_model(model, X_test_poly, y_test)

print("\n📊 Model Performance:")
print(f"R² Score: {results['R2']:.4f}")
print(f"RMSE: {results['RMSE']:.2f} TL")
print(f"MAE: {results['MAE']:.2f} TL")

# 7) Grafik - Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, results['predictions'], alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Salary (TL)')
plt.ylabel('Predicted Salary (TL)')
plt.title('Actual vs Predicted Salary')
plt.savefig('results/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# 8) Grafik - Feature Importance (Deneyim vs Maaş)
plt.figure(figsize=(10, 6))
plt.scatter(df['deneyim'], df['maas'], alpha=0.7, color='green')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary (TL)')
plt.title('Experience vs Salary')
plt.savefig('results/experience_vs_salary.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ Training completed!")
print("📈 Graphs saved to results/ folder")
