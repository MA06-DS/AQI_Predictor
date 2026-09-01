import os
import shutil
import tempfile
from pathlib import Path

import hopsworks
from dotenv import load_dotenv


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. CONFIGURATION
# ============================================================

PROJECT_NAME = "anaskaaqi"

API_KEY = os.getenv(
    "HOPSWORKS_API_KEY"
)

MODELS_DIR = Path("models")


# ============================================================
# 3. MODEL CONFIGURATION
# ============================================================

MODEL_CONFIGS = {

    # --------------------------------------------------------
    # +1 HOUR
    # --------------------------------------------------------

    1: {

        "model_name":
            "aqi_1_hour_xgboost",

        "model_file":
            "aqi_1_hour_xgboost_optimized.json",

        "feature_file":
            "aqi_1_hour_features_optimized.pkl",

        "r2":
            0.8919,

        "description":
            "Optimized XGBoost model for +1 hour AQI prediction."

    },


    # --------------------------------------------------------
    # +24 HOURS
    # --------------------------------------------------------

    24: {

        "model_name":
            "aqi_24_hour_xgboost",

        "model_file":
            "aqi_24_hour_xgboost_optimized.json",

        "feature_file":
            "aqi_24_hour_features_optimized.pkl",

        "r2":
            0.5605,

        "description":
            "Optimized XGBoost model for +24 hour AQI prediction."

    },


    # --------------------------------------------------------
    # +48 HOURS
    # --------------------------------------------------------

    48: {

        "model_name":
            "aqi_48_hour_xgboost",

        "model_file":
            "aqi_48_hour_xgboost_optimized.json",

        "feature_file":
            "aqi_48_hour_features_optimized.pkl",

        "r2":
            0.5347,

        "description":
            "Optimized XGBoost model for +48 hour AQI prediction."

    },


    # --------------------------------------------------------
    # +72 HOURS
    # --------------------------------------------------------

    72: {

        "model_name":
            "aqi_72_hour_xgboost",

        "model_file":
            "aqi_72_hour_xgboost_optimized.json",

        "feature_file":
            "aqi_72_hour_features_optimized.pkl",

        "r2":
            0.5283,

        "description":
            "Optimized XGBoost model for +72 hour AQI prediction."

    }

}


# ============================================================
# 4. CHECK API KEY
# ============================================================

if not API_KEY:

    raise EnvironmentError(
        "HOPSWORKS_API_KEY environment variable is not set."
    )


# ============================================================
# 5. CHECK MODEL DIRECTORY
# ============================================================

if not MODELS_DIR.exists():

    raise FileNotFoundError(
        "models/ directory was not found."
    )


print()
print("=" * 70)
print("AQI MODEL REGISTRY")
print("=" * 70)

print()
print(
    "Models directory:"
)

print(
    MODELS_DIR.resolve()
)


# ============================================================
# 6. CONNECT TO HOPSWORKS
# ============================================================

print()
print("=" * 70)
print("CONNECTING TO HOPSWORKS")
print("=" * 70)


project = hopsworks.login(

    project=PROJECT_NAME,

    api_key_value=API_KEY

)


print()
print(
    "Hopsworks connected successfully!"
)


# ============================================================
# 7. GET MODEL REGISTRY
# ============================================================

mr = project.get_model_registry()


print(
    "Model Registry connected successfully!"
)


# ============================================================
# 8. REGISTER MODELS
# ============================================================

registered_models = []


for horizon, config in MODEL_CONFIGS.items():

    print()
    print()
    print("#" * 70)

    print(
        f"REGISTERING +{horizon}H MODEL"
    )

    print("#" * 70)


    # --------------------------------------------------------
    # MODEL PATH
    # --------------------------------------------------------

    model_file = (

        MODELS_DIR
        /
        config["model_file"]

    )


    # --------------------------------------------------------
    # FEATURE FILE PATH
    # --------------------------------------------------------

    feature_file = (

        MODELS_DIR
        /
        config["feature_file"]

    )


    # --------------------------------------------------------
    # CHECK MODEL FILE
    # --------------------------------------------------------

    if not model_file.exists():

        print()
        print(
            f"WARNING: +{horizon}h model not found:"
        )

        print(
            model_file
        )

        print()
        print(
            f"Skipping +{horizon}h model."
        )

        continue


    # --------------------------------------------------------
    # FEATURE FILE WARNING
    # --------------------------------------------------------

    if not feature_file.exists():

        print()
        print(
            "WARNING: Feature file not found:"
        )

        print(
            feature_file
        )

        print()
        print(
            "The model will still be uploaded."
        )


    # --------------------------------------------------------
    # PRINT MODEL INFORMATION
    # --------------------------------------------------------

    print()
    print(
        "Model file:"
    )

    print(
        model_file
    )


    print()
    print(
        "Feature file:"
    )

    print(
        feature_file
    )


    print()
    print(
        "Registry name:"
    )

    print(
        config["model_name"]
    )


    print()
    print(
        "R² score:"
    )

    print(
        config["r2"]
    )


    # --------------------------------------------------------
    # CREATE TEMPORARY DIRECTORY
    # --------------------------------------------------------

    temp_dir = Path(

        tempfile.mkdtemp(

            prefix=f"aqi_{horizon}h_"

        )

    )


    print()
    print(
        "Temporary directory:"
    )

    print(
        temp_dir
    )


    try:

        # ----------------------------------------------------
        # COPY XGBOOST MODEL
        # ----------------------------------------------------

        shutil.copy2(

            model_file,

            temp_dir
            /
            model_file.name

        )


        # ----------------------------------------------------
        # COPY FEATURE FILE
        # ----------------------------------------------------

        if feature_file.exists():

            shutil.copy2(

                feature_file,

                temp_dir
                /
                feature_file.name

            )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metrics = {

            "r2":
                float(
                    config["r2"]
                )

        }


        # ----------------------------------------------------
        # CREATE MODEL REGISTRY ENTRY
        # ----------------------------------------------------

        print()
        print(
            f"Creating registry entry "
            f"for +{horizon}h..."
        )


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


        # ----------------------------------------------------
        # UPLOAD MODEL
        # ----------------------------------------------------

        print()
        print(
            f"Uploading +{horizon}h model..."
        )


        saved_model = (

            registered_model.save(

                str(
                    temp_dir
                )

            )

        )


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        print()
        print("-" * 70)

        print(
            f"+{horizon}H MODEL REGISTERED"
        )

        print("-" * 70)


        print(
            "Name:",
            saved_model.name
        )


        print(
            "Version:",
            saved_model.version
        )


        print(
            "R²:",
            config["r2"]
        )


        print(
            "Model path:",
            saved_model.model_path
        )


        print(
            "Version path:",
            saved_model.version_path
        )


        # ----------------------------------------------------
        # SAVE SUMMARY
        # ----------------------------------------------------

        registered_models.append({

            "horizon":
                horizon,

            "name":
                saved_model.name,

            "version":
                saved_model.version,

            "r2":
                config["r2"]

        })


    finally:

        # ----------------------------------------------------
        # REMOVE TEMPORARY DIRECTORY
        # ----------------------------------------------------

        shutil.rmtree(

            temp_dir,

            ignore_errors=True

        )


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 70)
print("REGISTRATION COMPLETE")
print("=" * 70)


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
print("=" * 70)
print("HOPSWORKS MODEL REGISTRY UPDATED")
print("=" * 70)

