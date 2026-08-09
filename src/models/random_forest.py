import os
import joblib
import hopsworks
import numpy as np

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
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
print("training data made")
print("\nDataset:")
print("X:", X.shape)
print("y:", y.shape)


# ==========================================
# 4. Convert to NumPy
# ==========================================

X = X.values
y = y.values


# ==========================================
# 5. Chronological Train/Validation/Test Split
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
# 6. Create Model
# ==========================================

base_model = ExtraTreesRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model = MultiOutputRegressor(
    base_model,
    n_jobs=-1
)


# ==========================================
# 7. Train
# ==========================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed!")


# ==========================================
# 8. Validation Prediction
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
# 9. Test Prediction
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
# 10. Evaluate Each Forecast Horizon
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
# 11. Save Model
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

model_path = "models/aqi_72_hour_model.pkl"

joblib.dump(
    model,
    model_path
)

print("\n===================================")
print("MODEL SAVED")
print("===================================")

print(model_path)