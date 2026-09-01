# ============================================================
# AQI HOURLY CI/CD FEATURE UPDATE
# ============================================================
#
# Sources:
#   1. Open-Meteo -> hourly weather
#   2. OpenAQ     -> hourly PM2.5
#
# OpenAQ:
#   Sensor: 6135426
#   Location: Aga Khan University Main Campus
#
# Hopsworks:
#   Project: anaskaaqi
#   Feature View: aqi_72_hour_forecast
#   Version: 2
#
# Purpose:
#   Run this file every hour from CI/CD.
#
# Pipeline:
#
#   OpenAQ + Open-Meteo
#          ↓
#      hourly data
#          ↓
#   merge using datetime_local
#          ↓
#   calendar features
#          ↓
#   lag features
#          ↓
#   rolling features
#          ↓
#   target_aqi_1 ... target_aqi_72
#          ↓
#   Hopsworks Feature Group
#
# IMPORTANT:
#   We recompute a rolling window instead of only creating
#   one row because newly available AQI values change:
#
#       aqi_lag_*
#       aqi_mean_*
#       aqi_std_*
#       target_aqi_1 ... target_aqi_72
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import sys
import time
import requests
import numpy as np
import pandas as pd
import hopsworks

from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv


# ============================================================
# 2. ENVIRONMENT
# ============================================================

load_dotenv()


HOPSWORKS_API_KEY = os.getenv(
    "HOPSWORKS_API_KEY"
)

OPENAQ_API_KEY = os.getenv(
    "OPENAQ_API_KEY"
)


if not HOPSWORKS_API_KEY:
    raise RuntimeError(
        "HOPSWORKS_API_KEY is not set."
    )


if not OPENAQ_API_KEY:
    raise RuntimeError(
        "OPENAQ_API_KEY is not set."
    )


# ============================================================
# 3. CONFIGURATION
# ============================================================

PROJECT_NAME = "anaskaaqi"

FEATURE_VIEW_NAME = "aqi_72_hour_forecast"

FEATURE_VIEW_VERSION = 2


# ------------------------------------------------------------
# Karachi
# ------------------------------------------------------------

CITY = "Karachi"

LATITUDE = 24.8607

LONGITUDE = 67.0011

TIMEZONE = "Asia/Karachi"


# ------------------------------------------------------------
# OpenAQ
# ------------------------------------------------------------

OPENAQ_SENSOR_ID = 6135426

OPENAQ_LOCATION_NAME = (
    "Aga Khan University Main Campus"
)

PM25_PARAMETER_ID = 2


OPENAQ_BASE_URL = (
    "https://api.openaq.org/v3"
)


# ------------------------------------------------------------
# Open-Meteo
# ------------------------------------------------------------

OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
)


# ------------------------------------------------------------
# Number of historical hours to retrieve
#
# We need more than 72 because:
#
#   lag_24
#   rolling_24
#   target_72
#
# all depend on surrounding observations.
#
# 96 gives us a safe buffer.
# ------------------------------------------------------------

LOOKBACK_HOURS = 96


# ------------------------------------------------------------
# Target horizon
# ------------------------------------------------------------

MAX_TARGET_HORIZON = 72


# ------------------------------------------------------------
# Feature windows
# ------------------------------------------------------------

LAG_HOURS = [
    1,
    3,
    6,
    12,
    24
]


ROLLING_WINDOWS = [
    6,
    12,
    24
]


# ============================================================
# 4. FINAL SCHEMA
# ============================================================

BASE_COLUMNS = [

    "datetime_local",

    "temperature",

    "humidity",

    "pressure",

    "wind_speed",

    "wind_direction",

    "precipitation",

    "cloud_cover",

    "datetime_utc",

    "pm25",

    "aqi",

    "hour",

    "day",

    "month",

    "year",

    "day_of_week",

    "is_weekend",

    "aqi_lag_1",

    "aqi_lag_3",

    "aqi_lag_6",

    "aqi_lag_12",

    "aqi_lag_24",

    "aqi_mean_6",

    "aqi_std_6",

    "aqi_mean_12",

    "aqi_std_12",

    "aqi_mean_24",

    "aqi_std_24",

]


TARGET_COLUMNS = [

    f"target_aqi_{i}"

    for i in range(
        1,
        MAX_TARGET_HORIZON + 1
    )

]


FINAL_SCHEMA = (
    BASE_COLUMNS
    +
    TARGET_COLUMNS
)


# ============================================================
# 5. GET CURRENT KARACHI HOUR
# ============================================================

def get_current_hour():

    now = pd.Timestamp.now(
        tz=TIMEZONE
    )

    return now.floor("h")


# ============================================================
# 6. GET UPDATE WINDOW
# ============================================================

def get_update_window():

    current_hour = (
        get_current_hour()
    )

    start = (
        current_hour
        - pd.Timedelta(
            hours=LOOKBACK_HOURS
        )
    )

    # We only process completed hours.
    end = current_hour

    print()
    print("=" * 60)
    print("UPDATE WINDOW")
    print("=" * 60)

    print(
        "Current Karachi hour:",
        current_hour
    )

    print(
        "Fetch from:",
        start
    )

    print(
        "Fetch until:",
        end
    )

    return start, end


# ============================================================
# 7. OPEN-METEO
# ============================================================

def fetch_openmeteo(
    start,
    end
):

    print()
    print("=" * 60)
    print("FETCHING OPEN-METEO")
    print("=" * 60)

    params = {

        "latitude":
            LATITUDE,

        "longitude":
            LONGITUDE,

        "start_date":
            start.strftime(
                "%Y-%m-%d"
            ),

        "end_date":
            end.strftime(
                "%Y-%m-%d"
            ),

        "hourly": ",".join([

            "temperature_2m",

            "relative_humidity_2m",

            "pressure_msl",

            "wind_speed_10m",

            "wind_direction_10m",

            "precipitation",

            "cloud_cover"

        ]),

        "timezone":
            TIMEZONE,

        "temperature_unit":
            "celsius",

        "wind_speed_unit":
            "kmh",

        "precipitation_unit":
            "mm"

    }


    response = requests.get(

        OPEN_METEO_URL,

        params=params,

        timeout=120

    )


    response.raise_for_status()


    payload = response.json()


    if "hourly" not in payload:

        raise RuntimeError(
            "Open-Meteo did not return hourly data."
        )


    hourly = payload["hourly"]


    weather = pd.DataFrame({

        "datetime_local":
            hourly["time"],

        "temperature":
            hourly["temperature_2m"],

        "humidity":
            hourly[
                "relative_humidity_2m"
            ],

        "pressure":
            hourly[
                "pressure_msl"
            ],

        "wind_speed":
            hourly[
                "wind_speed_10m"
            ],

        "wind_direction":
            hourly[
                "wind_direction_10m"
            ],

        "precipitation":
            hourly[
                "precipitation"
            ],

        "cloud_cover":
            hourly[
                "cloud_cover"
            ]

    })


    # --------------------------------------------------------
    # Local datetime
    # --------------------------------------------------------

    weather[
        "datetime_local"
    ] = pd.to_datetime(
        weather[
            "datetime_local"
        ]
    )


    # --------------------------------------------------------
    # Filter exact requested window
    # --------------------------------------------------------

    weather = weather[
        (
            weather[
                "datetime_local"
            ]
            >= start.tz_localize(None)
        )
        &
        (
            weather[
                "datetime_local"
            ]
            < end.tz_localize(None)
        )
    ]


    # --------------------------------------------------------
    # UTC datetime
    # --------------------------------------------------------

    local_aware = (
        weather[
            "datetime_local"
        ]
        .dt.tz_localize(
            TIMEZONE
        )
    )


    weather[
        "datetime_utc"
    ] = (
        local_aware
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )


    print(
        "Open-Meteo rows:",
        len(weather)
    )


    return weather


# ============================================================
# 8. OPENAQ
# ============================================================

def fetch_openaq(
    start,
    end
):

    print()
    print("=" * 60)
    print("FETCHING OPENAQ")
    print("=" * 60)

    print(
        "Sensor:",
        OPENAQ_SENSOR_ID
    )

    print(
        "Location:",
        OPENAQ_LOCATION_NAME
    )


    url = (

        OPENAQ_BASE_URL

        +

        f"/sensors/"
        f"{OPENAQ_SENSOR_ID}"
        f"/hours"

    )


    headers = {

        "X-API-Key":
            OPENAQ_API_KEY

    }


    # --------------------------------------------------------
    # OpenAQ expects datetime filters.
    # --------------------------------------------------------

    start_utc = (
        start
        .tz_convert("UTC")
    )


    end_utc = (
        end
        .tz_convert("UTC")
    )


    all_rows = []


    page = 1


    while True:

        params = {

            "datetime_from":
                start_utc.isoformat(),

            "datetime_to":
                end_utc.isoformat(),

            "limit":
                100,

            "page":
                page

        }


        response = requests.get(

            url,

            params=params,

            headers=headers,

            timeout=120

        )


        if response.status_code == 429:

            print(
                "OpenAQ rate limit reached."
            )

            print(
                "Waiting 10 seconds..."
            )

            time.sleep(10)

            continue


        response.raise_for_status()


        payload = response.json()


        results = payload.get(
            "results",
            []
        )


        if not results:

            break


        all_rows.extend(
            results
        )


        if len(results) < 100:

            break


        page += 1


        time.sleep(
            0.5
        )


    if not all_rows:

        raise RuntimeError(
            "OpenAQ returned no hourly "
            "PM2.5 data for sensor "
            f"{OPENAQ_SENSOR_ID}."
        )


    rows = []


    for row in all_rows:

        value = row.get(
            "value"
        )


        parameter = row.get(
            "parameter",
            {}
        )


        parameter_id = parameter.get(
            "id"
        )


        if (
            parameter_id is not None
            and
            parameter_id != PM25_PARAMETER_ID
        ):

            continue


        period = row.get(
            "period",
            {}
        )


        datetime_from = period.get(
            "datetimeFrom",
            {}
        )


        local_time = (
            datetime_from.get(
                "local"
            )
        )


        utc_time = (
            datetime_from.get(
                "utc"
            )
        )


        if local_time is None:

            continue


        rows.append({

            "datetime_local":
                local_time,

            "datetime_utc":
                utc_time,

            "pm25":
                value

        })


    if not rows:

        raise RuntimeError(
            "OpenAQ response contained no "
            "usable PM2.5 records."
        )


    aqi = pd.DataFrame(
        rows
    )


    # --------------------------------------------------------
    # Parse datetime
    # --------------------------------------------------------

    aqi[
        "datetime_local"
    ] = pd.to_datetime(
        aqi[
            "datetime_local"
        ]
    )


    # Remove timezone if supplied
    if (
        aqi[
            "datetime_local"
        ].dt.tz is not None
    ):

        aqi[
            "datetime_local"
        ] = (
            aqi[
                "datetime_local"
            ]
            .dt.tz_localize(None)
        )


    # --------------------------------------------------------
    # Numeric PM2.5
    # --------------------------------------------------------

    aqi[
        "pm25"
    ] = pd.to_numeric(
        aqi[
            "pm25"
        ],
        errors="coerce"
    )


    aqi = aqi.dropna(
        subset=[
            "datetime_local",
            "pm25"
        ]
    )


    # --------------------------------------------------------
    # Remove duplicate hours
    # --------------------------------------------------------

    aqi = (

        aqi

        .groupby(
            "datetime_local",
            as_index=False
        )

        ["pm25"]

        .mean()

    )


    # --------------------------------------------------------
    # Convert PM2.5 -> AQI
    # --------------------------------------------------------

    aqi[
        "aqi"
    ] = aqi_from_pm25(
        aqi[
            "pm25"
        ]
    )


    # --------------------------------------------------------
    # Recreate UTC timestamp
    # --------------------------------------------------------

    local_aware = (
        aqi[
            "datetime_local"
        ]
        .dt.tz_localize(
            TIMEZONE
        )
    )


    aqi[
        "datetime_utc"
    ] = (
        local_aware
        .dt.tz_convert(
            "UTC"
        )
        .dt.tz_localize(
            None
        )
    )


    print(
        "OpenAQ rows:",
        len(aqi)
    )


    print(
        "PM2.5 range:",
        aqi["pm25"].min(),
        "to",
        aqi["pm25"].max()
    )


    return aqi


# ============================================================
# 9. AQI FUNCTION
# ============================================================

def aqi_from_pm25(
    pm25_series
):

    def convert(
        concentration
    ):

        if pd.isna(
            concentration
        ):

            return np.nan


        try:

            concentration = float(
                concentration
            )

        except (
            TypeError,
            ValueError
        ):

            return np.nan


        if concentration < 0:

            return np.nan


        # ----------------------------------------------------
        # EPA PM2.5 concentration truncation
        # ----------------------------------------------------

        c = (
            np.floor(
                concentration * 10
            )
            / 10
        )


        # ----------------------------------------------------
        # PM2.5 AQI breakpoints
        # ----------------------------------------------------

        breakpoints = [

            (
                0.0,
                9.0,
                0,
                50
            ),

            (
                9.1,
                35.4,
                51,
                100
            ),

            (
                35.5,
                55.4,
                101,
                150
            ),

            (
                55.5,
                125.4,
                151,
                200
            ),

            (
                125.5,
                225.4,
                201,
                300
            ),

            (
                225.5,
                325.4,
                301,
                500
            )

        ]


        for (

            c_low,
            c_high,
            i_low,
            i_high

        ) in breakpoints:

            if (

                c >= c_low
                and
                c <= c_high

            ):

                aqi = (

                    (
                        i_high - i_low
                    )
                    /
                    (
                        c_high - c_low
                    )
                ) * (

                    c - c_low

                ) + i_low


                return round(
                    aqi
                )


        # ----------------------------------------------------
        # Above maximum breakpoint
        # ----------------------------------------------------

        if c > 325.4:

            return 500


        return np.nan


    return pm25_series.apply(
        convert
    )


# ============================================================
# 10. MERGE WEATHER + AQI
# ============================================================

def merge_sources(
    weather,
    aqi
):

    print()
    print("=" * 60)
    print("MERGING WEATHER + OPENAQ")
    print("=" * 60)


    data = pd.merge(

        weather,

        aqi[
            [
                "datetime_local",
                "pm25",
                "aqi"
            ]
        ],

        on="datetime_local",

        how="left"

    )


    data = (
        data
        .sort_values(
            "datetime_local"
        )
        .reset_index(
            drop=True
        )
    )


    print(
        "Merged rows:",
        len(data)
    )


    print(
        "Missing PM2.5:",
        data[
            "pm25"
        ].isna().sum()
    )


    print(
        "Missing AQI:",
        data[
            "aqi"
        ].isna().sum()
    )


    return data


# ============================================================
# 11. CALENDAR FEATURES
# ============================================================

def create_calendar_features(
    data
):

    data = data.copy()


    dt = pd.to_datetime(
        data[
            "datetime_local"
        ]
    )


    data[
        "hour"
    ] = dt.dt.hour


    data[
        "day"
    ] = dt.dt.day


    data[
        "month"
    ] = dt.dt.month


    data[
        "year"
    ] = dt.dt.year


    data[
        "day_of_week"
    ] = dt.dt.dayofweek


    data[
        "is_weekend"
    ] = (

        data[
            "day_of_week"
        ]
        >= 5

    ).astype(
        int
    )


    return data


# ============================================================
# 12. LOAD HISTORICAL DATA FROM HOPSWORKS
# ============================================================

def load_hopsworks_history(
    feature_view,
    start,
    end
):

    print()
    print("=" * 60)
    print("LOADING HISTORY FROM HOPSWORKS")
    print("=" * 60)


    # --------------------------------------------------------
    # We need the previous 3 days / 72 hours to correctly
    # construct lag and target features.
    # --------------------------------------------------------

    history_start = (
        start
        - pd.Timedelta(
            hours=72
        )
    )


    try:

        history = (
            feature_view
            .get_batch_data(
                start_time=history_start,
                end_time=end,
                dataframe_type="pandas",
                transformed=False
            )
        )

    except TypeError:

        # Compatibility with older Hopsworks versions.

        history = (
            feature_view
            .get_batch_data(
                start_time=history_start,
                end_time=end,
                dataframe_type="pandas"
            )
        )


    if history is None:

        return pd.DataFrame()


    history = pd.DataFrame(
        history
    )


    if history.empty:

        print(
            "No historical data returned."
        )

        return history


    if "datetime_local" not in history.columns:

        raise RuntimeError(
            "Hopsworks data does not contain "
            "'datetime_local'."
        )


    history[
        "datetime_local"
    ] = pd.to_datetime(
        history[
            "datetime_local"
        ]
    )


    # Remove timezone if any

    if (
        history[
            "datetime_local"
        ].dt.tz is not None
    ):

        history[
            "datetime_local"
        ] = (
            history[
                "datetime_local"
            ]
            .dt.tz_localize(None)
        )


    history = (
        history
        .sort_values(
            "datetime_local"
        )
        .drop_duplicates(
            "datetime_local",
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )


    print(
        "Historical rows:",
        len(history)
    )


    if not history.empty:

        print(
            "History from:",
            history[
                "datetime_local"
            ].min()
        )

        print(
            "History to:",
            history[
                "datetime_local"
            ].max()
        )


    return history


# ============================================================
# 13. MERGE HISTORICAL + NEW API DATA
# ============================================================

def combine_history_and_new(
    history,
    new_data
):

    print()
    print("=" * 60)
    print("COMBINING HISTORY + NEW DATA")
    print("=" * 60)


    # --------------------------------------------------------
    # Keep only raw/source columns from historical data.
    #
    # We deliberately recompute all derived features.
    # --------------------------------------------------------

    required_raw = [

        "datetime_local",

        "temperature",

        "humidity",

        "pressure",

        "wind_speed",

        "wind_direction",

        "precipitation",

        "cloud_cover",

        "datetime_utc",

        "pm25",

        "aqi"

    ]


    if history.empty:

        combined = new_data.copy()

    else:

        available = [

            c
            for c in required_raw
            if c in history.columns
        ]


        history_raw = history[
            available
        ].copy()


        combined = pd.concat(

            [
                history_raw,
                new_data
            ],

            ignore_index=True

        )


    # --------------------------------------------------------
    # New API data takes priority over historical values.
    # --------------------------------------------------------

    combined = (

        combined

        .sort_values(
            "datetime_local"
        )

        .drop_duplicates(
            "datetime_local",
            keep="last"
        )

        .reset_index(
            drop=True
        )

    )


    return combined


# ============================================================
# 14. CREATE LAG + ROLLING FEATURES
# ============================================================

def create_derived_features(
    data
):

    print()
    print("=" * 60)
    print("CREATING DERIVED FEATURES")
    print("=" * 60)


    data = (

        data

        .sort_values(
            "datetime_local"
        )

        .reset_index(
            drop=True
        )

        .copy()

    )


    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    for lag in LAG_HOURS:

        data[
            f"aqi_lag_{lag}"
        ] = (

            data[
                "aqi"
            ]
            .shift(lag)

        )


    # --------------------------------------------------------
    # Rolling features
    #
    # shift(1) prevents current AQI leakage.
    # --------------------------------------------------------

    previous_aqi = (
        data[
            "aqi"
        ]
        .shift(1)
    )


    for window in ROLLING_WINDOWS:

        data[
            f"aqi_mean_{window}"
        ] = (

            previous_aqi

            .rolling(
                window=window,
                min_periods=window
            )

            .mean()

        )


        data[
            f"aqi_std_{window}"
        ] = (

            previous_aqi

            .rolling(
                window=window,
                min_periods=window
            )

            .std()

        )


    return data


# ============================================================
# 15. CREATE 72 TARGETS
# ============================================================

def create_targets(
    data
):

    print()
    print("=" * 60)
    print("CREATING 72 TARGETS")
    print("=" * 60)


    data = (
        data
        .sort_values(
            "datetime_local"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )


    for horizon in range(
        1,
        MAX_TARGET_HORIZON + 1
    ):

        data[
            f"target_aqi_{horizon}"
        ] = (

            data[
                "aqi"
            ]
            .shift(
                -horizon
            )

        )


    return data


# ============================================================
# 16. APPLY THE 3-DAY RULE
# ============================================================

def apply_three_day_rule(
    data
):

    """
    Your stated logic:

    FIRST 3 DAYS
    -------------
    Use the historical AQI sequence with shift-based
    calculation.

    AFTER 3 DAYS
    ------------
    Use datetime_local as the authoritative hourly
    alignment.

    The important part here is that the dataframe is
    explicitly sorted and reindexed by datetime_local
    before shift/target creation.

    This avoids relying on API row order.
    """

    print()
    print("=" * 60)
    print("APPLYING 3-DAY AQI ALIGNMENT RULE")
    print("=" * 60)


    data = (
        data
        .sort_values(
            "datetime_local"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )


    data[
        "datetime_local"
    ] = pd.to_datetime(
        data[
            "datetime_local"
        ]
    )


    # --------------------------------------------------------
    # Ensure one hourly row per local timestamp.
    # --------------------------------------------------------

    data = (
        data
        .drop_duplicates(
            "datetime_local",
            keep="last"
        )
    )


    # --------------------------------------------------------
    # Reindex to hourly Karachi timeline.
    #
    # This is important once the pipeline is operating live.
    # If OpenAQ misses one hour, we don't accidentally make
    # shift(1) mean "previous available API row".
    # --------------------------------------------------------

    data = (
        data
        .set_index(
            "datetime_local"
        )
        .sort_index()
    )


    full_index = pd.date_range(

        start=data.index.min(),

        end=data.index.max(),

        freq="h"

    )


    data = data.reindex(
        full_index
    )


    data.index.name = (
        "datetime_local"
    )


    data = data.reset_index()


    # --------------------------------------------------------
    # Restore UTC timestamps for missing/reindexed rows.
    # --------------------------------------------------------

    local_aware = (
        data[
            "datetime_local"
        ]
        .dt.tz_localize(
            TIMEZONE
        )
    )


    data[
        "datetime_utc"
    ] = (

        local_aware

        .dt.tz_convert(
            "UTC"
        )

        .dt.tz_localize(
            None
        )

    )


    # --------------------------------------------------------
    # Recalculate calendar fields.
    # --------------------------------------------------------

    data = create_calendar_features(
        data
    )


    # --------------------------------------------------------
    # Derived features AFTER timeline alignment.
    # --------------------------------------------------------

    data = create_derived_features(
        data
    )


    # --------------------------------------------------------
    # Targets AFTER timeline alignment.
    # --------------------------------------------------------

    data = create_targets(
        data
    )


    print(
        "Aligned rows:",
        len(data)
    )


    return data


# ============================================================
# 17. GET ONLY ROWS THAT SHOULD BE UPDATED
# ============================================================

def select_update_rows(
    data,
    update_start,
    update_end
):

    data = data.copy()


    data[
        "datetime_local"
    ] = pd.to_datetime(
        data[
            "datetime_local"
        ]
    )


    # --------------------------------------------------------
    # We update:
    #
    #   previous 72 hours
    #
    # because new AQI values affect their target columns.
    #
    # Plus all newly available rows.
    # --------------------------------------------------------

    target_update_start = (

        update_start

        - pd.Timedelta(
            hours=MAX_TARGET_HORIZON
        )

    )


    mask = (

        data[
            "datetime_local"
        ]
        >= target_update_start

    ) & (

        data[
            "datetime_local"
        ]
        < update_end

    )


    result = data[
        mask
    ].copy()


    print()
    print("=" * 60)
    print("ROWS TO UPSERT")
    print("=" * 60)


    print(
        "Update from:",
        target_update_start
    )


    print(
        "Update until:",
        update_end
    )


    print(
        "Rows:",
        len(result)
    )


    return result


# ============================================================
# 18. PREPARE FINAL HOPSWORKS DATA
# ============================================================

def prepare_final_data(
    data
):

    data = data.copy()


    # --------------------------------------------------------
    # Replace infinities
    # --------------------------------------------------------

    data.replace(

        [
            np.inf,
            -np.inf
        ],

        np.nan,

        inplace=True

    )


    # --------------------------------------------------------
    # Ensure schema
    # --------------------------------------------------------

    for column in FINAL_SCHEMA:

        if column not in data.columns:

            data[
                column
            ] = np.nan


    # --------------------------------------------------------
    # Exact column order
    # --------------------------------------------------------

    data = data[
        FINAL_SCHEMA
    ]


    # --------------------------------------------------------
    # Convert datetime to string.
    #
    # If your existing Feature Group stores datetime_local
    # as string, this preserves the same representation.
    # --------------------------------------------------------

    data[
        "datetime_local"
    ] = pd.to_datetime(
        data[
            "datetime_local"
        ]
    ).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    data[
        "datetime_utc"
    ] = pd.to_datetime(
        data[
            "datetime_utc"
        ]
    ).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


    # --------------------------------------------------------
    # Remove completely duplicated timestamps
    # --------------------------------------------------------

    data = (

        data

        .drop_duplicates(
            "datetime_local",
            keep="last"
        )

        .sort_values(
            "datetime_local"
        )

        .reset_index(
            drop=True
        )

    )


    return data


# ============================================================
# 19. FIND UNDERLYING FEATURE GROUP
# ============================================================

def get_feature_group_from_view(
    feature_view
):

    print()
    print("=" * 60)
    print("FINDING HOPSWORKS FEATURE GROUP")
    print("=" * 60)


    # --------------------------------------------------------
    # Use datetime_local because it definitely belongs to
    # the feature schema you provided.
    # --------------------------------------------------------

    feature = (
        feature_view
        .get_feature(
            "datetime_local"
        )
    )


    feature_group = (
        feature.feature_group
    )


    if feature_group is None:

        raise RuntimeError(
            "Could not determine the underlying "
            "Feature Group from the Feature View."
        )


    print(
        "Feature Group:",
        feature_group.name
    )


    print(
        "Feature Group version:",
        feature_group.version
    )


    return feature_group


# ============================================================
# 20. UPSERT INTO HOPSWORKS
# ============================================================

def upload_to_hopsworks(
    feature_group,
    data
):

    print()
    print("=" * 60)
    print("UPLOADING TO HOPSWORKS")
    print("=" * 60)


    if data.empty:

        print(
            "Nothing to upload."
        )

        return


    print(
        "Rows:",
        len(data)
    )


    print(
        "Columns:",
        len(data.columns)
    )


    # --------------------------------------------------------
    # Important:
    #
    # operation="upsert"
    #
    # This allows us to update the previous 72 rows whose
    # target values may have changed after a new AQI value
    # became available.
    #
    # Hopsworks supports Pandas DataFrames in FeatureGroup
    # insert().
    # --------------------------------------------------------

    try:

        result = feature_group.insert(

            data,

            operation="upsert",

            wait=True

        )

    except TypeError:

        # Compatibility fallback for Hopsworks versions
        # where wait is not accepted.

        result = feature_group.insert(

            data,

            operation="upsert"

        )


    print()
    print(
        "Hopsworks upload completed."
    )


    return result


# ============================================================
# 21. VALIDATE FINAL DATA
# ============================================================

def validate_data(
    data
):

    print()
    print("=" * 60)
    print("FINAL VALIDATION")
    print("=" * 60)


    missing_columns = [

        column

        for column in FINAL_SCHEMA

        if column not in data.columns

    ]


    extra_columns = [

        column

        for column in data.columns

        if column not in FINAL_SCHEMA

    ]


    if missing_columns:

        raise RuntimeError(
            "Missing columns: "
            + str(missing_columns)
        )


    if extra_columns:

        raise RuntimeError(
            "Unexpected columns: "
            + str(extra_columns)
        )


    if list(data.columns) != FINAL_SCHEMA:

        raise RuntimeError(
            "Column order does not match FINAL_SCHEMA."
        )


    print(
        "Schema: PASS"
    )


    print(
        "Rows:",
        len(data)
    )


    print(
        "AQI available:",
        data["aqi"].notna().sum()
    )


    print(
        "PM2.5 available:",
        data["pm25"].notna().sum()
    )


    for horizon in [

        1,
        24,
        48,
        72

    ]:

        column = (
            f"target_aqi_{horizon}"
        )


        print(

            f"{column}:",

            data[
                column
            ]
            .notna()
            .sum()

        )


# ============================================================
# 22. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("KARACHI AQI HOURLY CI/CD PIPELINE")
    print("=" * 70)


    # --------------------------------------------------------
    # UPDATE WINDOW
    # --------------------------------------------------------

    update_start, update_end = (
        get_update_window()
    )


    # --------------------------------------------------------
    # HOPSWORKS LOGIN
    # --------------------------------------------------------

    print()
    print(
        "Connecting to Hopsworks..."
    )


    project = hopsworks.login(

        project=PROJECT_NAME,

        api_key_value=HOPSWORKS_API_KEY

    )


    print(
        "Connected to Hopsworks"
    )


    # --------------------------------------------------------
    # FEATURE STORE
    # --------------------------------------------------------

    fs = project.get_feature_store()


    # --------------------------------------------------------
    # FEATURE VIEW
    # --------------------------------------------------------

    feature_view = fs.get_feature_view(

        name=FEATURE_VIEW_NAME,

        version=FEATURE_VIEW_VERSION

    )


    print(
        "Feature View:",
        FEATURE_VIEW_NAME
    )


    print(
        "Feature View version:",
        FEATURE_VIEW_VERSION
    )


    # --------------------------------------------------------
    # UNDERLYING FEATURE GROUP
    # --------------------------------------------------------

    feature_group = (
        get_feature_group_from_view(
            feature_view
        )
    )


    # --------------------------------------------------------
    # FETCH WEATHER
    # --------------------------------------------------------

    weather = fetch_openmeteo(

        update_start,

        update_end

    )


    # --------------------------------------------------------
    # FETCH OPENAQ
    # --------------------------------------------------------

    aqi = fetch_openaq(

        update_start,

        update_end

    )


    # --------------------------------------------------------
    # MERGE API SOURCES
    # --------------------------------------------------------

    new_data = merge_sources(

        weather,

        aqi

    )


    # --------------------------------------------------------
    # LOAD LAST 72 HOURS FROM HOPSWORKS
    #
    # This is the important part for your stated logic.
    # --------------------------------------------------------

    history = load_hopsworks_history(

        feature_view,

        update_start,

        update_end

    )


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    combined = combine_history_and_new(

        history,

        new_data

    )


    # --------------------------------------------------------
    # APPLY 3-DAY / LOCAL TIME RULE
    # --------------------------------------------------------

    combined = apply_three_day_rule(
        combined
    )


    # --------------------------------------------------------
    # SELECT ONLY ROWS THAT NEED TO BE UPSERTED
    # --------------------------------------------------------

    update_rows = select_update_rows(

        combined,

        update_start,

        update_end

    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Don't upload rows where there is no AQI at all.
    #
    # Weather alone cannot produce your AQI target.
    # --------------------------------------------------------

    update_rows = update_rows[
        update_rows[
            "aqi"
        ].notna()
    ].copy()


    # --------------------------------------------------------
    # FINAL SCHEMA
    # --------------------------------------------------------

    final_data = prepare_final_data(

        update_rows

    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validate_data(

        final_data

    )


    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    upload_to_hopsworks(

        feature_group,

        final_data

    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


    print(
        "OpenAQ sensor:",
        OPENAQ_SENSOR_ID
    )


    print(
        "Location:",
        OPENAQ_LOCATION_NAME
    )


    print(
        "Rows uploaded:",
        len(final_data)
    )


    if not final_data.empty:

        print(
            "First timestamp:",
            final_data[
                "datetime_local"
            ].iloc[0]
        )


        print(
            "Last timestamp:",
            final_data[
                "datetime_local"
            ].iloc[-1]
        )


    print(
        "Feature View:",
        f"{FEATURE_VIEW_NAME} "
        f"v{FEATURE_VIEW_VERSION}"
    )


    print("=" * 70)


# ============================================================
# 23. RUN
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print()
        print("=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)

        print(
            type(error).__name__,
            ":",
            error
        )

        raise