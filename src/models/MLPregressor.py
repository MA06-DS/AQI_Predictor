import os
import joblib
import hopsworks
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. Connect to Hopsworks
# ==========================================

project = hopsworks.login(
    project="anaskaaqi",
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

print("Connected to Hopsworks")


# ==========================================
# 2. Get Feature View
# ==========================================

fs = project.get_feature_store()

feature_view = fs.get_feature_view(
    name="aqi_72_hour_forecast",
    version=2
)

print("Feature View loaded")


# ==========================================
# 3. Get Training Data
# ==========================================

X, y = feature_view.training_data(
    description="72-hour AQI forecasting training dataset"
)

print("\nDataset:")
print("X:", X.shape)
print("y:", y.shape)


# ==========================================
# 4. Convert to NumPy
# ==========================================

X = X.values.astype(np.float32)
y = y.values.astype(np.float32)


# ==========================================
# 5. Chronological Split
# ==========================================

n = len(X)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:validation_end]
y_val = y[train_end:validation_end]

X_test = X[validation_end:]
y_test = y[validation_end:]

print("\nSplit:")
print("Training   :", X_train.shape, y_train.shape)
print("Validation :", X_val.shape, y_val.shape)
print("Testing    :", X_test.shape, y_test.shape)


# ==========================================
# 6. Scale Input Features
# ==========================================

print("\nScaling features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

print("Feature scaling completed!")


# ==========================================
# 7. Create Neural Network
# ==========================================

model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=100,
    batch_size=64,
    early_stopping=True,
    validation_fraction=0.10,
    n_iter_no_change=10,
    random_state=42,
    verbose=True
)


# ==========================================
# 8. Train
# ==========================================

print("\nTraining MLP Neural Network...")
print("Architecture: 26 → 64 → 32 → 72")
print("Batch size: 64")
print("Maximum iterations: 100")

model.fit(
    X_train,
    y_train
)

print("\nMLP training completed!")


# ==========================================
# 9. Validation Prediction
# ==========================================

y_val_pred = model.predict(X_val)

val_mae = mean_absolute_error(
    y_val,
    y_val_pred
)

val_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        y_val_pred
    )
)

val_r2 = r2_score(
    y_val,
    y_val_pred,
    multioutput="uniform_average"
)

print("\n===================================")
print("VALIDATION RESULTS")
print("===================================")

print(f"MAE  : {val_mae:.4f}")
print(f"RMSE : {val_rmse:.4f}")
print(f"R²   : {val_r2:.4f}")


# ==========================================
# 10. Test Prediction
# ==========================================

y_test_pred = model.predict(X_test)

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_test_pred
    )
)

test_r2 = r2_score(
    y_test,
    y_test_pred,
    multioutput="uniform_average"
)

print("\n===================================")
print("TEST RESULTS")
print("===================================")

print(f"MAE  : {test_mae:.4f}")
print(f"RMSE : {test_rmse:.4f}")
print(f"R²   : {test_r2:.4f}")


# ==========================================
# 11. 72-Hour Horizon Performance
# ==========================================

print("\n===================================")
print("72-HOUR HORIZON PERFORMANCE")
print("===================================")

for i in range(72):

    horizon_mae = mean_absolute_error(
        y_test[:, i],
        y_test_pred[:, i]
    )

    horizon_rmse = np.sqrt(
        mean_squared_error(
            y_test[:, i],
            y_test_pred[:, i]
        )
    )

    horizon_r2 = r2_score(
        y_test[:, i],
        y_test_pred[:, i]
    )

    if i in [0, 5, 11, 23, 35, 47, 59, 71]:

        print(
            f"+{i+1:02d}h | "
            f"MAE: {horizon_mae:.2f} | "
            f"RMSE: {horizon_rmse:.2f} | "
            f"R²: {horizon_r2:.4f}"
        )


# ==========================================
# 12. Save Model + Scaler
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

model_path = "models/aqi_72_hour_mlp.pkl"
scaler_path = "models/aqi_72_hour_mlp_scaler.pkl"

joblib.dump(
    model,
    model_path
)

joblib.dump(
    scaler,
    scaler_path
)

print("\n===================================")
print("MODEL SAVED")
print("===================================")

print("Model :", model_path)
print("Scaler:", scaler_path)