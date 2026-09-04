import os
import gc
import warnings
import joblib
import shutil
import tempfile

import numpy as np
import pandas as pd
import hopsworks
import xgboost as xgb

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from pathlib import Path

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_NAME = "anaskaaqi"

FEATURE_VIEW_NAME = "aqi_72_hour_forecast"

FEATURE_VIEW_VERSION = 2

API_KEY = os.getenv("HOPSWORKS_API_KEY")

RANDOM_STATE = 42



# ============================================================
# ALL HORIZONS
# ============================================================

SEARCH_HORIZONS = [
    1,
    24,
    48,
    72
]


# ============================================================
# MAXIMUM ESTIMATORS
# ============================================================

# +1h  -> maximum 1500
# Others -> maximum 2500

MAX_ESTIMATORS = {

    1: 1500,

    24: 2500,

    48: 2500,

    72: 2500

}


EARLY_STOPPING_ROUNDS = 100


# ============================================================
# 2. XGBOOST INFORMATION
# ============================================================

print("===================================")
print("XGBOOST INFORMATION")
print("===================================")

print(
    "XGBoost version:",
    xgb.__version__
)


# ============================================================
# 3. CHECK GPU
# ============================================================

try:

    build_info = xgb.build_info()

    cuda_available = (

        "USE_CUDA" in build_info

        and

        str(
            build_info["USE_CUDA"]
        ).upper() == "ON"

    )

except Exception:

    cuda_available = False


if cuda_available:

    DEVICE = "cuda"

    print()
    print("CUDA detected.")
    print("Training will use GPU.")

else:

    DEVICE = "cpu"

    print()
    print("CUDA not detected.")
    print("Training will use CPU.")


# ============================================================
# 4. CONNECT TO HOPSWORKS
# ============================================================

print()
print("===================================")
print("CONNECTING TO HOPSWORKS")
print("===================================")

project = hopsworks.login(

    project=PROJECT_NAME,

    api_key_value=API_KEY

)

print(
    "Hopsworks connected successfully!"
)


# ============================================================
# 5. LOAD FEATURE VIEW
# ============================================================

fs = project.get_feature_store()


feature_view = fs.get_feature_view(

    name=FEATURE_VIEW_NAME,

    version=FEATURE_VIEW_VERSION

)


print(
    "Feature View loaded successfully!"
)


# ============================================================
# 6. MODEL REGISTRY
# ============================================================

print()
print("===================================")
print("CONNECTING TO MODEL REGISTRY")
print("===================================")

mr = project.get_model_registry()

MODEL_REGISTRY_NAMES = {
    1: "aqi_1_hour_xgboost",
    24: "aqi_24_hour_xgboost",
    48: "aqi_48_hour_xgboost",
    72: "aqi_72_hour_xgboost"
}


def get_latest_registered_model(model_name):
    """
    Return the latest registered version for a model name.

    If no version exists, return None.
    """
    try:
        registered_models = mr.get_models(model_name)
    except Exception as e:
        print(
            f"Could not retrieve registered model "
            f"'{model_name}': {e}"
        )
        return None

    if not registered_models:
        return None

    registered_models = sorted(
        registered_models,
        key=lambda model: int(model.version)
    )

    return registered_models[-1]


def get_registered_r2(model):
    """
    Read the R2 metric stored in the Hopsworks Model Registry.

    Current Hopsworks exposes registered metrics through
    the model.training_metrics property.
    """
    if model is None:
        return None

    metrics = getattr(
        model,
        "training_metrics",
        None
    )

    if metrics is None:
        # Compatibility fallback for older client versions.
        metrics = getattr(
            model,
            "metrics",
            None
        )

    if metrics is None:
        return None

    try:
        r2 = metrics.get("r2")
    except AttributeError:
        return None

    if r2 is None:
        return None

    try:
        return float(r2)
    except (TypeError, ValueError):
        return None


def register_model_if_better(
    horizon,
    model,
    test_r2,
    test_mae,
    test_rmse,
    selected_features,
    best_iteration,
    final_estimators,
    params,
    feature_set_name
):
    """
    Register the final model only when its TEST R2 is greater
    than the latest registered version.

    No persistent model files are kept locally.

    A temporary directory is used only as Hopsworks upload
    staging and is deleted immediately after registration.
    """

    model_name = MODEL_REGISTRY_NAMES[horizon]

    print()
    print("===================================")
    print(
        f"MODEL REGISTRY CHECK +{horizon}h"
    )
    print("===================================")
    print(
        "Registry model:",
        model_name
    )
    print(
        "New TEST R2:",
        f"{test_r2:.6f}"
    )

    latest_model = get_latest_registered_model(
        model_name
    )

    if latest_model is None:

        print(
            "No previous registered version found."
        )
        print(
            "The new model will be registered."
        )

        old_r2 = None
        old_version = None

    else:

        old_version = int(
            latest_model.version
        )

        old_r2 = get_registered_r2(
            latest_model
        )

        print(
            "Latest registered version:",
            old_version
        )

        if old_r2 is None:
            raise RuntimeError(
                f"Latest registered model "
                f"'{model_name}' version "
                f"{old_version} does not contain "
                f"a valid 'r2' metric. "
                f"Registration stopped to avoid "
                f"making an unsafe comparison."
            )

        print(
            "Old TEST R2:",
            f"{old_r2:.6f}"
        )

        if test_r2 <= old_r2:

            print()
            print(
                "NEW MODEL NOT REGISTERED."
            )
            print(
                f"New R2 ({test_r2:.6f}) "
                f"<= old R2 ({old_r2:.6f})"
            )

            return {
                "registered": False,
                "version": old_version,
                "old_r2": old_r2
            }

        print()
        print(
            f"New R2 ({test_r2:.6f}) "
            f"> old R2 ({old_r2:.6f})"
        )
        print(
            "New model will be registered."
        )

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"aqi_{horizon}h_registry_"
        )
    )

    try:

        # --------------------------------------------------------
        # TEMPORARY MODEL ARTIFACT
        # --------------------------------------------------------

        model_path = (
            temp_dir / "model.json"
        )

        model.save_model(
            str(model_path)
        )

        # --------------------------------------------------------
        # TEMPORARY FEATURE LIST
        # --------------------------------------------------------

        feature_path = (
            temp_dir / "features.pkl"
        )

        joblib.dump(
            selected_features,
            feature_path
        )

        # --------------------------------------------------------
        # MODEL REGISTRY METRICS
        # --------------------------------------------------------

        metrics = {
            "r2": float(test_r2),
            "mae": float(test_mae),
            "rmse": float(test_rmse)
        }

        description = (
            f"Optimized XGBoost AQI forecasting model "
            f"for +{horizon} hour horizon. "
            f"Feature set: {feature_set_name}. "
            f"Best iteration: {best_iteration}. "
            f"Final estimators: {final_estimators}. "
            f"Test R2: {test_r2:.6f}."
        )

        # --------------------------------------------------------
        # REGISTER DIRECTLY TO HOPSWORKS
        # --------------------------------------------------------

        registered_model = (
            mr.python.create_model(
                name=model_name,
                metrics=metrics,
                description=description
            )
        )

        saved_model = registered_model.save(
            str(temp_dir)
        )

        registered_version = int(
            saved_model.version
        )

        print()
        print(
            "MODEL REGISTERED SUCCESSFULLY"
        )
        print(
            "Model:",
            model_name
        )
        print(
            "Version:",
            registered_version
        )
        print(
            "R2:",
            f"{test_r2:.6f}"
        )

        return {
            "registered": True,
            "version": registered_version,
            "old_r2": old_r2
        }

    finally:

        # --------------------------------------------------------
        # DELETE LOCAL STAGING DIRECTORY
        # --------------------------------------------------------

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        print()
        print(
            "Temporary model files deleted."
        )




# ============================================================
# 6. LOAD DATA
# ============================================================

print()
print("===================================")
print("LOADING TRAINING DATA")
print("===================================")


X, y = feature_view.training_data(

    description=
    "72-hour AQI forecasting training dataset"

)


print(
    "Original X shape:",
    X.shape
)

print(
    "Original y shape:",
    y.shape
)


# ============================================================
# 7. COPY DATA
# ============================================================

X = X.copy()

y = y.copy()


# ============================================================
# 8. SORT CHRONOLOGICALLY
# ============================================================

if "timestamp" in X.columns:

    X["timestamp"] = pd.to_datetime(
        X["timestamp"]
    )

    order = X[
        "timestamp"
    ].argsort()

    X = (
        X
        .iloc[order]
        .reset_index(drop=True)
    )

    y = (
        y
        .iloc[order]
        .reset_index(drop=True)
    )


elif "datetime" in X.columns:

    X["datetime"] = pd.to_datetime(
        X["datetime"]
    )

    order = X[
        "datetime"
    ].argsort()

    X = (
        X
        .iloc[order]
        .reset_index(drop=True)
    )

    y = (
        y
        .iloc[order]
        .reset_index(drop=True)
    )


else:

    print(
        "No timestamp/datetime column found."
    )

    print(
        "Assuming Hopsworks data is already chronological."
    )


# ============================================================
# 9. COMBINE X + Y
# ============================================================

data = X.copy()


for column in y.columns:

    data[column] = y[
        column
    ].values


print()
print(
    "Combined shape:",
    data.shape
)


# ============================================================
# 10. FIND AQI COLUMN
# ============================================================

if "aqi" in data.columns:

    AQI_COLUMN = "aqi"

elif "AQI" in data.columns:

    AQI_COLUMN = "AQI"

else:

    raise ValueError(
        "AQI column not found."
    )


print(
    "AQI column:",
    AQI_COLUMN
)


# ============================================================
# 11. CREATE EXTRA LAG FEATURES
# ============================================================

print()
print("===================================")
print("CREATING EXTRA LAG FEATURES")
print("===================================")


LAGS = [

    1,
    2,
    3,
    6,
    12,
    18,
    24,
    36,
    48,
    72

]


for lag in LAGS:

    data[
        f"aqi_extra_lag_{lag}"
    ] = (

        data[
            AQI_COLUMN
        ]

        .shift(lag)

    )


# ============================================================
# 12. ROLLING FEATURES
# ============================================================

ROLLING_WINDOWS = [

    3,
    6,
    12,
    24,
    36,
    48,
    72

]


for window in ROLLING_WINDOWS:

    data[
        f"aqi_extra_mean_{window}"
    ] = (

        data[
            AQI_COLUMN
        ]

        .rolling(window)
        .mean()

    )


    data[
        f"aqi_extra_std_{window}"
    ] = (

        data[
            AQI_COLUMN
        ]

        .rolling(window)
        .std()

    )


    data[
        f"aqi_extra_min_{window}"
    ] = (

        data[
            AQI_COLUMN
        ]

        .rolling(window)
        .min()

    )


    data[
        f"aqi_extra_max_{window}"
    ] = (

        data[
            AQI_COLUMN
        ]

        .rolling(window)
        .max()

    )


# ============================================================
# 13. AQI CHANGE FEATURES
# ============================================================

data["aqi_change_1"] = (

    data[
        AQI_COLUMN
    ]

    .diff(1)

)


data["aqi_change_3"] = (

    data[
        AQI_COLUMN
    ]

    .diff(3)

)


data["aqi_change_6"] = (

    data[
        AQI_COLUMN
    ]

    .diff(6)

)


data["aqi_change_12"] = (

    data[
        AQI_COLUMN
    ]

    .diff(12)

)


data["aqi_change_24"] = (

    data[
        AQI_COLUMN
    ]

    .diff(24)

)


data["aqi_change_48"] = (

    data[
        AQI_COLUMN
    ]

    .diff(48)

)


data["aqi_pct_change_6"] = (

    data[
        AQI_COLUMN
    ]

    .pct_change(6)

)


data["aqi_pct_change_24"] = (

    data[
        AQI_COLUMN
    ]

    .pct_change(24)

)


# ============================================================
# 14. CYCLICAL TIME FEATURES
# ============================================================

if "hour" in data.columns:

    data["hour_sin"] = np.sin(

        2
        * np.pi
        * data["hour"]
        / 24

    )


    data["hour_cos"] = np.cos(

        2
        * np.pi
        * data["hour"]
        / 24

    )


if "day_of_week" in data.columns:

    data["dow_sin"] = np.sin(

        2
        * np.pi
        * data["day_of_week"]
        / 7

    )


    data["dow_cos"] = np.cos(

        2
        * np.pi
        * data["day_of_week"]
        / 7

    )


# ============================================================
# 15. CLEAN DATA
# ============================================================

data.replace(

    [
        np.inf,
        -np.inf
    ],

    np.nan,

    inplace=True

)


# ============================================================
# 16. TARGET COLUMNS
# ============================================================

TARGET_COLUMNS = [

    f"target_aqi_{i}"

    for i in range(
        1,
        73
    )

]


missing_targets = [

    col

    for col in TARGET_COLUMNS

    if col not in data.columns

]


if missing_targets:

    raise ValueError(

        "Missing target columns:\n"

        + str(
            missing_targets
        )

    )


# ============================================================
# 17. DROP ROWS WITH MISSING TARGETS
# ============================================================

data = (

    data

    .dropna(
        subset=TARGET_COLUMNS
    )

    .reset_index(
        drop=True
    )

)


print()
print(
    "After feature engineering:",
    data.shape
)


# ============================================================
# 18. FEATURE COLUMNS
# ============================================================

EXCLUDE_COLUMNS = set(
    TARGET_COLUMNS
)


if "timestamp" in data.columns:

    EXCLUDE_COLUMNS.add(
        "timestamp"
    )


if "datetime" in data.columns:

    EXCLUDE_COLUMNS.add(
        "datetime"
    )


FEATURE_COLUMNS = []


for column in data.columns:

    if column in EXCLUDE_COLUMNS:

        continue


    if pd.api.types.is_numeric_dtype(

        data[column]

    ):

        FEATURE_COLUMNS.append(
            column
        )


print()
print("===================================")
print("FEATURE INFORMATION")
print("===================================")


print(
    "Total features:",
    len(FEATURE_COLUMNS)
)


# ============================================================
# 19. X DATA
# ============================================================

X_ALL = (

    data[
        FEATURE_COLUMNS
    ]

    .astype(
        np.float32
    )

)


# ============================================================
# 20. CHRONOLOGICAL SPLIT
# ============================================================

N = len(data)


TRAIN_END = int(
    N * 0.70
)


VALIDATION_END = int(
    N * 0.85
)


X_TRAIN = X_ALL.iloc[
    :TRAIN_END
]


X_VAL = X_ALL.iloc[
    TRAIN_END:VALIDATION_END
]


X_TEST = X_ALL.iloc[
    VALIDATION_END:
]


print()
print("===================================")
print("DATA SPLIT")
print("===================================")


print(
    "Train:",
    X_TRAIN.shape
)


print(
    "Validation:",
    X_VAL.shape
)


print(
    "Test:",
    X_TEST.shape
)


# ============================================================
# 21. FEATURE SETS
# ============================================================

SHORT_TERM_FEATURES = {

    "aqi_change_1",

    "aqi_extra_lag_1",

    "aqi_extra_lag_2"

}


FEATURE_SETS = {

    "full":
        FEATURE_COLUMNS,

    "long_horizon":

        [

            f

            for f in FEATURE_COLUMNS

            if f not in SHORT_TERM_FEATURES

        ]

}


# ============================================================
# 22. TARGETED SEARCH SPACE
# ============================================================

SEARCH_SPACE = {


    # ========================================================
    # +1 HOUR
    # ========================================================

    1: [

        {

            "max_depth": 5,

            "learning_rate": 0.015,

            "min_child_weight": 5,

            "subsample": 0.90,

            "colsample_bytree": 0.90,

            "reg_alpha": 0.05,

            "reg_lambda": 2.0,

            "gamma": 0.05,

            "n_estimators": 1500

        }

    ],


    # ========================================================
    # +24 HOURS
    # ========================================================

    24: [

        {

            "max_depth": 4,

            "learning_rate": 0.015,

            "min_child_weight": 3,

            "subsample": 0.90,

            "colsample_bytree": 0.90,

            "reg_alpha": 0.05,

            "reg_lambda": 2.0,

            "gamma": 0.00,

            "n_estimators": 2500

        },


        {

            "max_depth": 5,

            "learning_rate": 0.012,

            "min_child_weight": 5,

            "subsample": 0.90,

            "colsample_bytree": 0.90,

            "reg_alpha": 0.10,

            "reg_lambda": 3.0,

            "gamma": 0.05,

            "n_estimators": 2500

        },


        {

            "max_depth": 6,

            "learning_rate": 0.010,

            "min_child_weight": 5,

            "subsample": 0.95,

            "colsample_bytree": 0.95,

            "reg_alpha": 0.05,

            "reg_lambda": 2.0,

            "gamma": 0.00,

            "n_estimators": 2500

        },


        {

            "max_depth": 5,

            "learning_rate": 0.010,

            "min_child_weight": 7,

            "subsample": 0.85,

            "colsample_bytree": 0.85,

            "reg_alpha": 0.15,

            "reg_lambda": 4.0,

            "gamma": 0.05,

            "n_estimators": 2500

        },


        {

            "max_depth": 6,

            "learning_rate": 0.008,

            "min_child_weight": 7,

            "subsample": 0.90,

            "colsample_bytree": 0.85,

            "reg_alpha": 0.10,

            "reg_lambda": 5.0,

            "gamma": 0.10,

            "n_estimators": 2500

        }

    ],


    # ========================================================
    # +48 HOURS
    # ========================================================

    48: [

        {

            "max_depth": 3,

            "learning_rate": 0.015,

            "min_child_weight": 5,

            "subsample": 0.90,

            "colsample_bytree": 0.90,

            "reg_alpha": 0.10,

            "reg_lambda": 3.0,

            "gamma": 0.05,

            "n_estimators": 2500

        },


        {

            "max_depth": 4,

            "learning_rate": 0.012,

            "min_child_weight": 5,

            "subsample": 0.90,

            "colsample_bytree": 0.85,

            "reg_alpha": 0.15,

            "reg_lambda": 4.0,

            "gamma": 0.10,

            "n_estimators": 2500

        },


        {

            "max_depth": 4,

            "learning_rate": 0.010,

            "min_child_weight": 7,

            "subsample": 0.90,

            "colsample_bytree": 0.90,

            "reg_alpha": 0.10,

            "reg_lambda": 5.0,

            "gamma": 0.05,

            "n_estimators": 2500

        },


        {

            "max_depth": 5,

            "learning_rate": 0.010,

            "min_child_weight": 7,

            "subsample": 0.85,

            "colsample_bytree": 0.85,

            "reg_alpha": 0.20,

            "reg_lambda": 5.0,

            "gamma": 0.10,

            "n_estimators": 2500

        },


        {

            "max_depth": 4,

            "learning_rate": 0.008,

            "min_child_weight": 9,

            "subsample": 0.90,

            "colsample_bytree": 0.80,

            "reg_alpha": 0.20,

            "reg_lambda": 6.0,

            "gamma": 0.10,

            "n_estimators": 2500

        }

    ],


    # ========================================================
    # +72 HOURS
    # ========================================================

    72: [

        {

            "max_depth": 3,

            "learning_rate": 0.015,

            "min_child_weight": 5,

            "subsample": 0.90,

            "colsample_bytree": 0.90,

            "reg_alpha": 0.10,

            "reg_lambda": 3.0,

            "gamma": 0.05,

            "n_estimators": 2500

        },


        {

            "max_depth": 4,

            "learning_rate": 0.012,

            "min_child_weight": 7,

            "subsample": 0.90,

            "colsample_bytree": 0.85,

            "reg_alpha": 0.15,

            "reg_lambda": 4.0,

            "gamma": 0.10,

            "n_estimators": 2500

        },


        {

            "max_depth": 4,

            "learning_rate": 0.010,

            "min_child_weight": 7,

            "subsample": 0.95,

            "colsample_bytree": 0.90,

            "reg_alpha": 0.10,

            "reg_lambda": 5.0,

            "gamma": 0.05,

            "n_estimators": 2500

        },


        {

            "max_depth": 5,

            "learning_rate": 0.010,

            "min_child_weight": 9,

            "subsample": 0.85,

            "colsample_bytree": 0.85,

            "reg_alpha": 0.20,

            "reg_lambda": 5.0,

            "gamma": 0.10,

            "n_estimators": 2500

        },


        {

            "max_depth": 4,

            "learning_rate": 0.008,

            "min_child_weight": 9,

            "subsample": 0.90,

            "colsample_bytree": 0.80,

            "reg_alpha": 0.20,

            "reg_lambda": 6.0,

            "gamma": 0.15,

            "n_estimators": 2500

        }

    ]

}


# ============================================================
# 23. TRAINING FUNCTION
# ============================================================

def train_candidate(

    horizon,

    params,

    feature_set_name

):


    print()
    print(
        f"Training +{horizon}h | "
        f"Feature set: {feature_set_name}"
    )


    selected_features = FEATURE_SETS[
        feature_set_name
    ]


    Xtr = X_TRAIN[
        selected_features
    ]


    Xv = X_VAL[
        selected_features
    ]


    target_column = (
        f"target_aqi_{horizon}"
    )


    y_all = data[
        target_column
    ].astype(
        np.float32
    )


    ytr = y_all.iloc[
        :TRAIN_END
    ]


    yv = y_all.iloc[
        TRAIN_END:VALIDATION_END
    ]


    # --------------------------------------------------------
    # MAXIMUM NUMBER OF TREES
    # --------------------------------------------------------

    max_estimators = params[
        "n_estimators"
    ]


    model = xgb.XGBRegressor(

        objective=
        "reg:squarederror",


        n_estimators=
        max_estimators,


        max_depth=
        params[
            "max_depth"
        ],


        learning_rate=
        params[
            "learning_rate"
        ],


        min_child_weight=
        params[
            "min_child_weight"
        ],


        subsample=
        params[
            "subsample"
        ],


        colsample_bytree=
        params[
            "colsample_bytree"
        ],


        reg_alpha=
        params[
            "reg_alpha"
        ],


        reg_lambda=
        params[
            "reg_lambda"
        ],


        gamma=
        params[
            "gamma"
        ],


        random_state=
        RANDOM_STATE,


        n_jobs=-1,


        tree_method="hist",


        device=
        DEVICE,


        eval_metric="rmse",


        early_stopping_rounds=
        EARLY_STOPPING_ROUNDS

    )


    model.fit(

        Xtr,

        ytr,

        eval_set=[

            (
                Xv,
                yv
            )

        ],

        verbose=False

    )


    # --------------------------------------------------------
    # VALIDATION PREDICTION
    # --------------------------------------------------------

    y_pred = model.predict(
        Xv
    )


    mae = mean_absolute_error(

        yv,

        y_pred

    )


    rmse = np.sqrt(

        mean_squared_error(

            yv,

            y_pred

        )

    )


    r2 = r2_score(

        yv,

        y_pred

    )


    # --------------------------------------------------------
    # BEST ITERATION
    # --------------------------------------------------------

    try:

        best_iteration = (
            model.best_iteration
        )

    except Exception:

        best_iteration = (
            max_estimators - 1
        )


    print()
    print(
        f"Validation R²: {r2:.4f}"
    )


    print(
        f"Validation MAE: {mae:.4f}"
    )


    print(
        f"Validation RMSE: {rmse:.4f}"
    )


    print(
        f"Maximum estimators: "
        f"{max_estimators}"
    )


    print(
        f"Best iteration: "
        f"{best_iteration}"
    )


    print(
        f"Best number of trees: "
        f"{best_iteration + 1}"
    )


    del Xtr
    del Xv
    del ytr
    del yv
    del y_pred

    gc.collect()


    return {

        "model":
        model,


        "horizon":
        horizon,


        "feature_set":
        feature_set_name,


        "params":
        params.copy(),


        "r2":
        r2,


        "mae":
        mae,


        "rmse":
        rmse,


        "best_iteration":
        best_iteration,


        "feature_count":
        len(
            selected_features
        )

    }


# ============================================================
# 24. RUN TARGETED SEARCH
# ============================================================

all_results = []


print()
print()
print("===================================")
print("TARGETED HYPERPARAMETER SEARCH")
print("===================================")


for horizon in SEARCH_HORIZONS:


    print()
    print()
    print("###################################")


    print(
        f"HORIZON +{horizon}h"
    )


    print("###################################")


    configurations = (
        SEARCH_SPACE[
            horizon
        ]
    )


    config_number = 0


    for params in configurations:


        config_number += 1


        print()
        print("-----------------------------------")


        print(
            f"Configuration "
            f"{config_number}/"
            f"{len(configurations)}"
        )


        print("-----------------------------------")


        print(params)


        # ----------------------------------------------------
        # +1h uses only the specified configuration.
        #
        # +24/+48/+72 use both feature sets.
        # ----------------------------------------------------

        if horizon == 1:

            feature_sets_to_test = [
                "full"
            ]

        else:

            feature_sets_to_test = [

                "full",

                "long_horizon"

            ]


        for feature_set_name in feature_sets_to_test:


            result = train_candidate(

                horizon,

                params,

                feature_set_name

            )


            all_results.append(
                result
            )


# ============================================================
# 25. SEARCH RESULTS
# ============================================================

results_df = pd.DataFrame([

    {

        "horizon":
        r["horizon"],


        "feature_set":
        r["feature_set"],


        "r2":
        r["r2"],


        "mae":
        r["mae"],


        "rmse":
        r["rmse"],


        "best_iteration":
        r["best_iteration"],


        "best_trees":
        r["best_iteration"] + 1,


        "max_estimators":
        r["params"][
            "n_estimators"
        ],


        "feature_count":
        r["feature_count"]

    }


    for r in all_results

])


print()
print()
print("===================================")
print("ALL SEARCH RESULTS")
print("===================================")


print(

    results_df

    .sort_values(

        [
            "horizon",
            "r2"

        ],

        ascending=[

            True,
            False

        ]

    )

    .to_string(
        index=False
    )

)


# ============================================================
# 26. FIND BEST MODEL PER HORIZON
# ============================================================

best_results = {}


for horizon in SEARCH_HORIZONS:


    horizon_results = [

        r

        for r in all_results

        if r["horizon"] == horizon

    ]


    best = max(

        horizon_results,

        key=lambda x: x["r2"]

    )


    best_results[
        horizon
    ] = best


# ============================================================
# 27. PRINT BEST CONFIGURATIONS
# ============================================================

print()
print()
print("===================================")
print("BEST CONFIGURATION PER HORIZON")
print("===================================")


for horizon in SEARCH_HORIZONS:


    best = best_results[
        horizon
    ]


    print()
    print(
        f"+{horizon}h"
    )


    print(
        "Validation R²:",
        f"{best['r2']:.4f}"
    )


    print(
        "Feature set:",
        best["feature_set"]
    )


    print(
        "Features:",
        best["feature_count"]
    )


    print(
        "Maximum estimators:",
        best["params"][
            "n_estimators"
        ]
    )


    print(
        "Best iteration:",
        best["best_iteration"]
    )


    print(
        "Best number of trees:",
        best["best_iteration"] + 1
    )


    print(
        "Parameters:"
    )


    print(
        best["params"]
    )


# ============================================================
# 28. SAVE SEARCH RESULTS
# ============================================================

results_df.to_csv(

    "models/"
    "aqi_targeted_xgboost_search_results.csv",

    index=False

)


# ============================================================
# 29. FINAL TRAINING
# ============================================================

print()
print()
print("===================================")
print("TRAINING FINAL MODELS")
print("===================================")


final_models = {}


for horizon in SEARCH_HORIZONS:


    best = best_results[
        horizon
    ]


    params = best[
        "params"
    ]


    feature_set_name = (
        best[
            "feature_set"
        ]
    )


    selected_features = (
        FEATURE_SETS[
            feature_set_name
        ]
    )


    # --------------------------------------------------------
    # TRAIN + VALIDATION
    # --------------------------------------------------------

    X_train_final = X_ALL[
        selected_features
    ].iloc[
        :VALIDATION_END
    ]


    X_test_final = X_ALL[
        selected_features
    ].iloc[
        VALIDATION_END:
    ]


    target_column = (
        f"target_aqi_{horizon}"
    )


    y_all = data[
        target_column
    ].astype(
        np.float32
    )


    y_train_final = y_all.iloc[
        :VALIDATION_END
    ]


    y_test_final = y_all.iloc[
        VALIDATION_END:
    ]


    print()
    print("-----------------------------------")


    print(
        f"Final +{horizon}h model"
    )


    print("-----------------------------------")


    # --------------------------------------------------------
    # USE BEST ITERATION
    # --------------------------------------------------------

    best_iteration = (
        best[
            "best_iteration"
        ]
    )


    if best_iteration < 0:

        final_estimators = 1

    else:

        final_estimators = (
            best_iteration + 1
        )


    print(
        "Best iteration:",
        best_iteration
    )


    print(
        "Final estimators:",
        final_estimators
    )


    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    final_model = xgb.XGBRegressor(

        objective=
        "reg:squarederror",


        n_estimators=
        final_estimators,


        max_depth=
        params[
            "max_depth"
        ],


        learning_rate=
        params[
            "learning_rate"
        ],


        min_child_weight=
        params[
            "min_child_weight"
        ],


        subsample=
        params[
            "subsample"
        ],


        colsample_bytree=
        params[
            "colsample_bytree"
        ],


        reg_alpha=
        params[
            "reg_alpha"
        ],


        reg_lambda=
        params[
            "reg_lambda"
        ],


        gamma=
        params[
            "gamma"
        ],


        random_state=
        RANDOM_STATE,


        n_jobs=-1,


        tree_method="hist",


        device=
        DEVICE,


        eval_metric="rmse"

    )


    # --------------------------------------------------------
    # RETRAIN USING TRAIN + VALIDATION
    # --------------------------------------------------------

    final_model.fit(

        X_train_final,

        y_train_final,

        verbose=False

    )


    # ========================================================
    # TEST
    # ========================================================

    y_test_pred = final_model.predict(

        X_test_final

    )


    test_mae = mean_absolute_error(

        y_test_final,

        y_test_pred

    )


    test_rmse = np.sqrt(

        mean_squared_error(

            y_test_final,

            y_test_pred

        )

    )


    test_r2 = r2_score(

        y_test_final,

        y_test_pred

    )


    print()
    print(
        f"+{horizon:02d}h TEST"
    )


    print(
        f"MAE  : {test_mae:.4f}"
    )


    print(
        f"RMSE : {test_rmse:.4f}"
    )


    print(
        f"R²   : {test_r2:.4f}"
    )


    # ========================================================
    # MODEL REGISTRY
    # ========================================================

    registry_result = register_model_if_better(
        horizon=horizon,
        model=final_model,
        test_r2=test_r2,
        test_mae=test_mae,
        test_rmse=test_rmse,
        selected_features=selected_features,
        best_iteration=best_iteration,
        final_estimators=final_estimators,
        params=params,
        feature_set_name=feature_set_name
    )

# ========================================================
    # STORE RESULT
    # ========================================================

    final_models[
        horizon
    ] = {

        "model":
        final_model,


        "test_r2":
        test_r2,


        "test_mae":
        test_mae,


        "test_rmse":
        test_rmse,


        "best_iteration":
        best_iteration,


        "final_estimators":
        final_estimators,


        "features":
        selected_features,

        "registry_registered":
        registry_result["registered"],

        "registry_version":
        registry_result["version"],

        "old_registry_r2":
        registry_result["old_r2"]

    }


    del X_train_final
    del X_test_final
    del y_train_final
    del y_test_final
    del y_test_pred

    gc.collect()


# ============================================================
# 30. FINAL RESULTS
# ============================================================

print()
print()
print("===================================")
print("FINAL OPTIMIZED RESULTS")
print("===================================")


final_rows = []


for horizon in SEARCH_HORIZONS:


    result = final_models[
        horizon
    ]


    final_rows.append({

        "Horizon":
        horizon,


        "MAE":
        result[
            "test_mae"
        ],


        "RMSE":
        result[
            "test_rmse"
        ],


        "R2":
        result[
            "test_r2"
        ],


        "Best_Iteration":
        result[
            "best_iteration"
        ],


        "Final_Trees":
        result[
            "final_estimators"
        ]

    })


final_df = pd.DataFrame(
    final_rows
)


print(

    final_df.to_string(

        index=False,

        float_format=
        lambda x:
        f"{x:.4f}"

    )

)


# ============================================================
# 31. FINAL 4-HORIZON PERFORMANCE
# ============================================================

r2_1 = final_models[
    1
][
    "test_r2"
]


r2_24 = final_models[
    24
][
    "test_r2"
]


r2_48 = final_models[
    48
][
    "test_r2"
]


r2_72 = final_models[
    72
][
    "test_r2"
]


average_r2 = (

    r2_1

    + r2_24

    + r2_48

    + r2_72

) / 4


print()
print("===================================")
print("FINAL 4-HORIZON PERFORMANCE")
print("===================================")


print(
    f"+01h | R²: "
    f"{r2_1:.4f}"
)


print(
    f"+24h | R²: "
    f"{r2_24:.4f}"
)


print(
    f"+48h | R²: "
    f"{r2_48:.4f}"
)


print(
    f"+72h | R²: "
    f"{r2_72:.4f}"
)


print()
print("===================================")
print("FINAL AVERAGE TEST R²")
print("===================================")


print(
    f"{average_r2:.4f}"
)


# ============================================================
# 32. TARGET CHECK
# ============================================================

print()
print("===================================")
print("TARGET CHECK")
print("===================================")


print(
    f"+1h >= 0.80 : "
    f"{r2_1 >= 0.80}"
)


print(
    f"+24h >= 0.70 : "
    f"{r2_24 >= 0.70}"
)


print(
    f"+48h >= 0.50 : "
    f"{r2_48 >= 0.50}"
)


print(
    f"+72h >= 0.50 : "
    f"{r2_72 >= 0.50}"
)


print(
    f"Average > 0.70: "
    f"{average_r2 > 0.70}"
)


# ============================================================
# 33. SAVE FINAL RESULTS
# ============================================================

final_df.to_csv(

    "models/"
    "aqi_optimized_final_results.csv",

    index=False

)


# ============================================================
# 34. SAVE SUMMARY
# ============================================================

summary = {

    "1h_r2":
    r2_1,


    "24h_r2":
    r2_24,


    "48h_r2":
    r2_48,


    "72h_r2":
    r2_72,


    "average_r2":
    average_r2,


    "1h_best_iteration":
    final_models[
        1
    ][
        "best_iteration"
    ],


    "1h_final_trees":
    final_models[
        1
    ][
        "final_estimators"
    ],


    "24h_best_iteration":
    final_models[
        24
    ][
        "best_iteration"
    ],


    "24h_final_trees":
    final_models[
        24
    ][
        "final_estimators"
    ],


    "48h_best_iteration":
    final_models[
        48
    ][
        "best_iteration"
    ],


    "48h_final_trees":
    final_models[
        48
    ][
        "final_estimators"
    ],


    "72h_best_iteration":
    final_models[
        72
    ][
        "best_iteration"
    ],


    "72h_final_trees":
    final_models[
        72
    ][
        "final_estimators"
    ]

}


joblib.dump(

    summary,

    "models/"
    "aqi_final_r2_summary.pkl"

)


# ============================================================
# 35. FINAL FILE LIST
# ============================================================

print()
print("===================================")
print("OUTPUT FILES")
print("===================================")

print()
print(
    "Search results:"
)
print(
    "  models/aqi_targeted_xgboost_search_results.csv"
)

print()
print(
    "Final results:"
)
print(
    "  models/aqi_optimized_final_results.csv"
)

print()
print(
    "Summary:"
)
print(
    "  models/aqi_final_r2_summary.pkl"
)

print()
print(
    "MODEL ARTIFACTS:"
)
print(
    "  Final XGBoost model files are NOT "
    "kept locally."
)
print(
    "  Models are registered in the "
    "Hopsworks Model Registry only when "
    "TEST R2 is better than the latest "
    "registered version."
)

# ============================================================
# 36. DONE
# ============================================================

print()
print("===================================")
print("SEARCH + FINAL TRAINING COMPLETE")
print("===================================")

print()
print(
    "All four optimized models were evaluated "
    "against the Hopsworks Model Registry."
)

print()
print(
    "Only models with TEST R2 greater than "
    "the latest registered version were "
    "registered as new versions."
)

print()
print(
    "IMPORTANT:"
)

print(
    "+1h was ACTUALLY TRAINED."
)

print(
    "+1h maximum estimators = 1500."
)

print(
    "+1h final estimators = best_iteration + 1."
)

print(
    "+24/+48/+72 maximum estimators = 2500."
)

print(
    "Final models use TRAIN + VALIDATION."
)

print(
    "TEST was used only for final evaluation."
)

print()
print(
    "==================================="
)