# ============================================================
# AQI MODEL REGISTRY
# Register +1h, +24h, +48h and +72h XGBoost models
# ============================================================

import os
import shutil
import tempfile
from pathlib import Path

import hopsworks


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_NAME = "anaskaaqi"
API_KEY=os.getenv("HOPSWORKS_API_KEY")
MODELS_DIR = Path("models")


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

MODEL_CONFIGS = {

    1: {
        "model_name": "aqi_1_hour_xgboost",
        "model_file": "aqi_1_hour_xgboost_optimized.json",
        "feature_file": "aqi_1_hour_features_optimized.pkl",

        # Your existing final +1h test R²
        "r2": 0.8906,

        "description":
            "Optimized XGBoost model for +1 hour AQI prediction."
    },

    24: {
        "model_name": "aqi_24_hour_xgboost",
        "model_file": "aqi_24_hour_xgboost_optimized.json",
        "feature_file": "aqi_24_hour_features_optimized.pkl",

        # CHANGE THIS after your final training
        "r2": 0.5681,

        "description":
            "Optimized XGBoost model for +24 hour AQI prediction."
    },

    48: {
        "model_name": "aqi_48_hour_xgboost",
        "model_file": "aqi_48_hour_xgboost_optimized.json",
        "feature_file": "aqi_48_hour_features_optimized.pkl",

        # CHANGE THIS after your final training
        "r2": 0.5262,

        "description":
            "Optimized XGBoost model for +48 hour AQI prediction."
    },

    72: {
        "model_name": "aqi_72_hour_xgboost",
        "model_file": "aqi_72_hour_xgboost_optimized.json",
        "feature_file": "aqi_72_hour_features_optimized.pkl",

        # CHANGE THIS after your final training
        "r2": 0.5125,

        "description":
            "Optimized XGBoost model for +72 hour AQI prediction."
    }

}


# ============================================================
# 3. CHECK API KEY
# ============================================================

if not API_KEY:

    raise EnvironmentError(
        "HOPSWORKS_API_KEY environment variable is not set."
    )


# ============================================================
# 4. CHECK MODEL DIRECTORY
# ============================================================

if not MODELS_DIR.exists():

    raise FileNotFoundError(
        "models/ directory was not found."
    )


print()
print("===================================")
print("AQI MODEL REGISTRY")
print("===================================")

print()
print("Models directory:")
print(MODELS_DIR.resolve())


# ============================================================
# 5. CONNECT TO HOPSWORKS
# ============================================================

print()
print("===================================")
print("CONNECTING TO HOPSWORKS")
print("===================================")

project = hopsworks.login(
    project=PROJECT_NAME,
    api_key_value=API_KEY
)

print()
print("Hopsworks connected successfully!")


# ============================================================
# 6. GET MODEL REGISTRY
# ============================================================

mr = project.get_model_registry()

print("Model Registry connected successfully!")


# ============================================================
# 7. REGISTER EACH MODEL
# ============================================================

registered_models = []


for horizon, config in MODEL_CONFIGS.items():

    print()
    print()
    print("###################################")
    print(f"REGISTERING +{horizon}H MODEL")
    print("###################################")


    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    model_file = (
        MODELS_DIR /
        config["model_file"]
    )

    feature_file = (
        MODELS_DIR /
        config["feature_file"]
    )


    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not model_file.exists():

        print()
        print(
            f"WARNING: +{horizon}h model not found:"
        )

        print(model_file)

        print(
            f"Skipping +{horizon}h."
        )

        continue


    # --------------------------------------------------------
    # Check feature list
    # --------------------------------------------------------

    if not feature_file.exists():

        print()
        print(
            f"WARNING: Feature file not found:"
        )

        print(feature_file)

        print(
            "The model will still be uploaded."
        )


    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print()
    print("Model file:")
    print(model_file)

    print()
    print("Feature file:")
    print(feature_file)

    print()
    print("Registry name:")
    print(config["model_name"])


    # --------------------------------------------------------
    # Create temporary directory
    # --------------------------------------------------------

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"aqi_{horizon}h_"
        )
    )


    print()
    print("Temporary directory:")
    print(temp_dir)


    try:

        # ----------------------------------------------------
        # Copy XGBoost model
        # ----------------------------------------------------

        shutil.copy2(
            model_file,
            temp_dir / model_file.name
        )


        # ----------------------------------------------------
        # Copy feature list
        # ----------------------------------------------------

        if feature_file.exists():

            shutil.copy2(
                feature_file,
                temp_dir / feature_file.name
            )


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        # Hopsworks expects numeric metric values.
        #
        # Do NOT put:
        #
        # "framework": "XGBoost"
        #
        # inside metrics.

        metrics = {}

        if config["r2"] is not None:

            metrics["r2"] = float(
                config["r2"]
            )


        # ----------------------------------------------------
        # CREATE MODEL
        # ----------------------------------------------------

        print()
        print(
            f"Creating registry entry "
            f"for +{horizon}h..."
        )


        if metrics:

            registered_model = (
                mr.python.create_model(

                    name=config[
                        "model_name"
                    ],

                    metrics=metrics,

                    description=config[
                        "description"
                    ]

                )
            )

        else:

            registered_model = (
                mr.python.create_model(

                    name=config[
                        "model_name"
                    ],

                    description=config[
                        "description"
                    ]

                )
            )


        # ----------------------------------------------------
        # UPLOAD MODEL
        # ----------------------------------------------------

        print()
        print(
            f"Uploading +{horizon}h model..."
        )


        saved_model = registered_model.save(
            str(temp_dir)
        )


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        print()
        print("-----------------------------------")
        print(
            f"+{horizon}H MODEL REGISTERED"
        )
        print("-----------------------------------")

        print(
            "Name:",
            saved_model.name
        )

        print(
            "Version:",
            saved_model.version
        )

        print(
            "Model path:",
            saved_model.model_path
        )

        print(
            "Version path:",
            saved_model.version_path
        )


        registered_models.append({

            "horizon": horizon,

            "name":
                saved_model.name,

            "version":
                saved_model.version,

            "r2":
                config["r2"]

        })


    finally:

        # ----------------------------------------------------
        # Remove temporary files
        # ----------------------------------------------------

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# ============================================================
# 8. FINAL SUMMARY
# ============================================================

print()
print()
print("===================================")
print("REGISTRATION COMPLETE")
print("===================================")

print()

if not registered_models:

    print(
        "No models were registered."
    )

else:

    for model in registered_models:

        print(
            f"+{model['horizon']:02d}h | "
            f"{model['name']} | "
            f"Version {model['version']} | "
            f"R²: {model['r2']}"
        )


print()
print("===================================")
print("HOPSWORKS MODEL REGISTRY UPDATED")
print("===================================")