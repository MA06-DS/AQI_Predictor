# ============================================================
# OPENAQ - KARACHI AQI 2026+ DATA FETCHER
# ============================================================
#
# Source:
#   OpenAQ API v3
#
# Location:
#   6135426
#   Aga Khan University Main Campus
#
# Parameter:
#   PM2.5 (parameter ID = 2)
#
# Output:
#   data/2026/karachi_2026_aqi.csv
#
# IMPORTANT:
#   - Only Location 6135426 is used.
#   - No automatic Karachi station discovery.
#   - No other OpenAQ stations are merged.
#   - Missing PM2.5 values are NOT artificially filled.
#   - Data is requested in smaller date chunks.
#   - OpenAQ timestamps are handled as UTC internally.
#   - Final local timestamps are Asia/Karachi.
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import time
import requests
import numpy as np
import pandas as pd

from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# 2. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 3. CONFIGURATION
# ============================================================

CITY = "Karachi"

TIMEZONE = "Asia/Karachi"


# ------------------------------------------------------------
# OpenAQ location
# ------------------------------------------------------------

OPENAQ_LOCATION_ID = 6135426

OPENAQ_LOCATION_NAME = (
    "Aga Khan University Main Campus"
)


# ------------------------------------------------------------
# PM2.5 parameter
# ------------------------------------------------------------

PM25_PARAMETER_ID = 2


# ------------------------------------------------------------
# Date range
# ------------------------------------------------------------

START_DATE = "2026-01-01"

# None = automatically use yesterday
END_DATE = None


# ------------------------------------------------------------
# Historical API chunk size
# ------------------------------------------------------------
#
# Instead of requesting 8 months in one request, the script
# requests smaller chunks.
#
# This makes the historical download more reliable and also
# helps identify exactly which periods contain data.
#
# ------------------------------------------------------------

CHUNK_DAYS = 30


# ============================================================
# 4. OPENAQ API
# ============================================================

OPENAQ_BASE_URL = (
    "https://api.openaq.org/v3"
)


OPENAQ_API_KEY = os.getenv(
    "OPENAQ_API_KEY"
)


if not OPENAQ_API_KEY:

    raise RuntimeError(
        "\n"
        "OPENAQ_API_KEY is missing.\n\n"
        "Set it using:\n\n"
        "export OPENAQ_API_KEY='YOUR_KEY'\n"
    )


HEADERS = {
    "X-API-Key": OPENAQ_API_KEY,
    "Accept": "application/json",
}


# ============================================================
# 5. OUTPUT
# ============================================================

OUTPUT_DIR = Path(
    "data/2026"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


AQI_FILE = (
    OUTPUT_DIR /
    "karachi_2026_aqi.csv"
)


MISSING_FILE = (
    OUTPUT_DIR /
    "karachi_2026_missing_aqi_hours.csv"
)


# ============================================================
# 6. DATE RANGE
# ============================================================

def get_date_range():

    # --------------------------------------------------------
    # Always create timezone-aware timestamps.
    # --------------------------------------------------------

    start = pd.Timestamp(
        START_DATE,
        tz=TIMEZONE
    )

    if END_DATE is None:

        # Current date in Karachi
        today_local = (
            pd.Timestamp.now(
                tz=TIMEZONE
            )
            .normalize()
        )

        # Use yesterday so today's incomplete data
        # is not included.
        end = (
            today_local
            - pd.Timedelta(days=1)
        )

    else:

        end = pd.Timestamp(
            END_DATE,
            tz=TIMEZONE
        )

    if end < start:

        raise ValueError(
            "END_DATE cannot be earlier than START_DATE."
        )

    return start, end


START_TS, END_TS = get_date_range()


# ============================================================
# 7. PRINT CONFIGURATION
# ============================================================

print()
print("=" * 70)
print("OPENAQ KARACHI AQI FETCHER")
print("=" * 70)

print(
    f"Location ID : {OPENAQ_LOCATION_ID}"
)

print(
    f"Location    : {OPENAQ_LOCATION_NAME}"
)

print(
    f"Parameter   : PM2.5 ({PM25_PARAMETER_ID})"
)

print(
    f"Start       : {START_TS.date()}"
)

print(
    f"End         : {END_TS.date()}"
)

print(
    f"Timezone    : {TIMEZONE}"
)

print(
    f"Chunk size  : {CHUNK_DAYS} days"
)

print("=" * 70)


# ============================================================
# 8. GET SENSORS FOR LOCATION
# ============================================================

def get_location_sensors():

    print()
    print("=" * 70)
    print("GETTING SENSORS")
    print("=" * 70)

    url = (
        f"{OPENAQ_BASE_URL}"
        f"/locations/{OPENAQ_LOCATION_ID}/sensors"
    )

    params = {
        "limit": 100,
        "page": 1,
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=120,
    )

    response.raise_for_status()

    payload = response.json()

    sensors = payload.get(
        "results",
        []
    )

    print(
        f"Sensors returned: {len(sensors)}"
    )

    if not sensors:

        raise RuntimeError(
            f"No sensors found for "
            f"OpenAQ location "
            f"{OPENAQ_LOCATION_ID}."
        )

    for sensor in sensors:

        sensor_id = sensor.get(
            "id"
        )

        sensor_name = sensor.get(
            "name"
        )

        parameter = sensor.get(
            "parameter",
            {}
        )

        parameter_id = parameter.get(
            "id"
        )

        parameter_name = parameter.get(
            "name"
        )

        units = parameter.get(
            "units"
        )

        print(
            f"Sensor {sensor_id}: "
            f"{sensor_name} | "
            f"parameter={parameter_name} "
            f"(id={parameter_id}) | "
            f"units={units}"
        )

    return sensors


# ============================================================
# 9. FIND PM2.5 SENSOR
# ============================================================

def find_pm25_sensor():

    sensors = get_location_sensors()

    print()
    print("=" * 70)
    print("SEARCHING FOR PM2.5 SENSOR")
    print("=" * 70)

    selected = []

    for sensor in sensors:

        parameter = sensor.get(
            "parameter",
            {}
        )

        parameter_id = parameter.get(
            "id"
        )

        parameter_name = str(
            parameter.get(
                "name",
                ""
            )
        ).lower()

        if (

            parameter_id ==
            PM25_PARAMETER_ID

            or

            parameter_name
            in {
                "pm25",
                "pm2.5",
                "pm 2.5",
            }

        ):

            selected.append(
                sensor
            )

    if not selected:

        raise RuntimeError(
            "\n"
            f"No PM2.5 sensor exists at "
            f"OpenAQ location "
            f"{OPENAQ_LOCATION_ID}.\n\n"
            "The location exists, but OpenAQ "
            "did not return a PM2.5 sensor "
            "for this location."
        )

    print(
        f"PM2.5 sensors found: "
        f"{len(selected)}"
    )

    for sensor in selected:

        print(
            f"Selected sensor: "
            f"{sensor.get('id')} | "
            f"{sensor.get('name')}"
        )

    return selected


# ============================================================
# 10. EXTRACT OPENAQ TIMESTAMP
# ============================================================

def extract_utc_timestamp(row):

    """
    Extract the UTC timestamp from an OpenAQ v3 measurement.

    OpenAQ v3 can return datetimeFrom as an object:

        {
            "utc": "...",
            "local": "..."
        }

    We explicitly extract the UTC value.
    """

    period = row.get(
        "period",
        {}
    )

    datetime_from = period.get(
        "datetimeFrom"
    )

    # --------------------------------------------------------
    # Normal OpenAQ v3 structure
    # --------------------------------------------------------

    if isinstance(
        datetime_from,
        dict
    ):

        utc_value = (
            datetime_from.get(
                "utc"
            )
        )

        if utc_value is not None:

            return utc_value

    # --------------------------------------------------------
    # Fallback if datetimeFrom is already a string
    # --------------------------------------------------------

    if isinstance(
        datetime_from,
        str
    ):

        return datetime_from

    # --------------------------------------------------------
    # Fallback fields
    # --------------------------------------------------------

    value = row.get(
        "datetimeFrom"
    )

    if isinstance(
        value,
        str
    ):

        return value

    value = row.get(
        "datetime"
    )

    if isinstance(
        value,
        str
    ):

        return value

    return None


# ============================================================
# 11. FETCH ONE DATE CHUNK
# ============================================================

def fetch_sensor_hourly_chunk(
    sensor_id,
    chunk_start,
    chunk_end
):

    url = (
        f"{OPENAQ_BASE_URL}"
        f"/sensors/{sensor_id}"
        f"/measurements/hourly"
    )

    # --------------------------------------------------------
    # chunk_start and chunk_end are already timezone-aware.
    # DO NOT call tz_localize() here.
    # --------------------------------------------------------

    params = {

        "datetime_from":
            chunk_start.isoformat(),

        "datetime_to":
            chunk_end.isoformat(),

        "limit":
            100,

        "page":
            1,
    }

    all_rows = []

    while True:

        print(
            f"    Requesting page "
            f"{params['page']}..."
        )

        response = requests.get(

            url,

            headers=HEADERS,

            params=params,

            timeout=120,

        )

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

        print(
            f"    Received "
            f"{len(results)} rows"
        )

        meta = payload.get(
            "meta",
            {}
        )

        found = meta.get(
            "found"
        )

        if found is not None:

            try:

                if len(all_rows) >= int(
                    found
                ):

                    break

            except (
                ValueError,
                TypeError
            ):

                pass

        if len(results) < 100:

            break

        params["page"] += 1

        time.sleep(
            0.25
        )

    return all_rows


# ============================================================
# 12. FETCH HOURLY SENSOR DATA
# ============================================================

def fetch_sensor_hourly(
    sensor_id
):

    print()
    print("=" * 70)
    print(
        f"FETCHING HOURLY DATA "
        f"FOR SENSOR {sensor_id}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Create chunks.
    # --------------------------------------------------------

    chunks = []

    current_start = START_TS

    while current_start < (
        END_TS
        + pd.Timedelta(days=1)
    ):

        current_end = min(

            current_start
            + pd.Timedelta(
                days=CHUNK_DAYS
            ),

            END_TS
            + pd.Timedelta(days=1)

        )

        chunks.append(
            (
                current_start,
                current_end
            )
        )

        current_start = current_end

    print(
        f"Total date chunks: "
        f"{len(chunks)}"
    )

    all_rows = []

    # --------------------------------------------------------
    # Fetch each chunk.
    # --------------------------------------------------------

    for index, (
        chunk_start,
        chunk_end
    ) in enumerate(
        chunks,
        start=1
    ):

        print()
        print(
            f"Chunk [{index}/{len(chunks)}]"
        )

        print(
            f"    From: "
            f"{chunk_start}"
        )

        print(
            f"    To:   "
            f"{chunk_end}"
        )

        try:

            chunk_rows = (
                fetch_sensor_hourly_chunk(
                    sensor_id,
                    chunk_start,
                    chunk_end
                )
            )

            print(
                f"    Chunk rows: "
                f"{len(chunk_rows)}"
            )

            all_rows.extend(
                chunk_rows
            )

        except requests.HTTPError as e:

            print(
                f"    Chunk failed: "
                f"{e}"
            )

            if e.response is not None:

                print(
                    f"    HTTP status: "
                    f"{e.response.status_code}"
                )

                print(
                    f"    Response: "
                    f"{e.response.text[:1000]}"
                )

            raise

        time.sleep(
            0.25
        )

    print()
    print(
        f"Total raw hourly rows: "
        f"{len(all_rows)}"
    )

    if not all_rows:

        return pd.DataFrame()

    # ========================================================
    # Extract timestamp and PM2.5
    # ========================================================

    rows = []

    for row in all_rows:

        datetime_from = (
            extract_utc_timestamp(
                row
            )
        )

        value = row.get(
            "value"
        )

        if datetime_from is None:

            continue

        rows.append({

            "datetime":
                datetime_from,

            "pm25":
                value,

            "sensor_id":
                sensor_id,

        })

    if not rows:

        print(
            "No usable timestamp/value rows found."
        )

        return pd.DataFrame()

    df = pd.DataFrame(
        rows
    )

    # ========================================================
    # PM2.5 numeric conversion
    # ========================================================

    df["pm25"] = pd.to_numeric(
        df["pm25"],
        errors="coerce"
    )

    # ========================================================
    # Timestamp conversion
    # ========================================================

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
        errors="coerce"
    )

    # ========================================================
    # Remove invalid values
    # ========================================================

    df = df[
        df["pm25"].notna()
        &
        df["datetime"].notna()
    ].copy()

    # ========================================================
    # Remove duplicate timestamps
    # ========================================================

    df = (
        df
        .drop_duplicates(
            subset=[
                "datetime",
                "sensor_id"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        f"Valid PM2.5 rows: "
        f"{len(df)}"
    )

    # ========================================================
    # Diagnostic timestamps
    # ========================================================

    if not df.empty:

        print(
            "First UTC timestamp:",
            df["datetime"].min()
        )

        print(
            "Last UTC timestamp:",
            df["datetime"].max()
        )

    return df


# ============================================================
# 13. COMBINE PM2.5 SENSORS
# ============================================================

def fetch_openaq():

    sensors = find_pm25_sensor()

    all_sensor_data = []

    print()
    print("=" * 70)
    print("FETCHING PM2.5 DATA")
    print("=" * 70)

    for index, sensor in enumerate(
        sensors,
        start=1
    ):

        sensor_id = sensor.get(
            "id"
        )

        print()
        print(
            f"[{index}/{len(sensors)}] "
            f"Sensor ID: {sensor_id}"
        )

        try:

            df = fetch_sensor_hourly(
                sensor_id
            )

            if not df.empty:

                all_sensor_data.append(
                    df
                )

            else:

                print(
                    "No data returned."
                )

        except Exception as e:

            print(
                f"Sensor {sensor_id} failed:"
            )

            print(
                repr(e)
            )

            raise

        time.sleep(
            0.25
        )

    if not all_sensor_data:

        raise RuntimeError(
            "\n"
            "OpenAQ returned no PM2.5 "
            "measurements."
        )

    data = pd.concat(
        all_sensor_data,
        ignore_index=True
    )

    print()
    print(
        "Raw combined rows:",
        len(data)
    )

    # ========================================================
    # Diagnostic timestamp information
    # ========================================================

    print()
    print(
        "Raw timestamp sample:"
    )

    print(
        data["datetime"].head()
    )

    print()
    print(
        "Raw first timestamp:",
        data["datetime"].min()
    )

    print(
        "Raw last timestamp:",
        data["datetime"].max()
    )

    # ========================================================
    # Convert UTC -> Karachi
    # ========================================================

    data["datetime_local"] = (

        data["datetime"]

        .dt.tz_convert(
            TIMEZONE
        )

    )

    # ========================================================
    # Keep requested date range
    # ========================================================

    # Both values are timezone-aware here.
    #
    # This avoids mixing timezone-aware and timezone-naive
    # timestamps.
    # ========================================================

    local_start = START_TS

    local_end = (
        END_TS
        + pd.Timedelta(days=1)
    )

    data = data[
        (
            data["datetime_local"]
            >= local_start
        )
        &
        (
            data["datetime_local"]
            < local_end
        )
    ].copy()

    print()
    print(
        "Rows after date filtering:",
        len(data)
    )

    if data.empty:

        print()
        print(
            "WARNING:"
        )

        print(
            "OpenAQ returned records, but none "
            "fall inside the requested date range."
        )

        print(
            "Requested:",
            local_start,
            "->",
            local_end
        )

        print(
            "Available UTC:",
            data["datetime"].min()
            if not data.empty
            else "unknown"
        )

        raise RuntimeError(
            "OpenAQ returned no records "
            "inside the requested date range."
        )

    # ========================================================
    # Multiple sensors
    # ========================================================
    #
    # If more than one PM2.5 sensor exists at this location,
    # average values for the same hour.
    # ========================================================

    data = (

        data

        .groupby(
            "datetime_local",
            as_index=False
        )

        ["pm25"]

        .mean()

    )

    # ========================================================
    # Convert PM2.5 -> AQI
    # ========================================================

    data["aqi"] = (
        aqi_from_pm25(
            data["pm25"]
        )
    )

    # ========================================================
    # Create UTC timestamp for output
    # ========================================================

    # datetime_local is still timezone-aware.
    #
    # Convert directly to UTC.
    # ========================================================

    data["datetime_utc"] = (

        data["datetime_local"]

        .dt.tz_convert(
            "UTC"
        )

    )

    # ========================================================
    # Remove timezone for CSV
    # ========================================================

    data["datetime_local"] = (

        data["datetime_local"]

        .dt.tz_localize(
            None
        )

    )

    data["datetime_utc"] = (

        data["datetime_utc"]

        .dt.tz_localize(
            None
        )

    )

    # ========================================================
    # Format timestamps
    # ========================================================

    data["datetime_local"] = (
        data["datetime_local"]
        .dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    data["datetime_utc"] = (
        data["datetime_utc"]
        .dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    # ========================================================
    # Final column order
    # ========================================================

    data = data[
        [
            "datetime_local",
            "datetime_utc",
            "pm25",
            "aqi",
        ]
    ]

    data = (
        data
        .sort_values(
            "datetime_local"
        )
        .reset_index(
            drop=True
        )
    )

    return data


# ============================================================
# 14. PM2.5 -> US EPA AQI
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
            ValueError,
            TypeError
        ):

            return np.nan

        if concentration < 0:

            return np.nan

        # ----------------------------------------------------
        # EPA PM2.5 AQI calculation
        # ----------------------------------------------------
        #
        # PM2.5 is truncated to one decimal place.
        # ----------------------------------------------------

        c = (
            np.floor(
                concentration * 10
            )
            / 10
        )

        breakpoints = [

            # PM2.5     AQI

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
            ),

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
                        i_high
                        - i_low
                    )

                    /

                    (
                        c_high
                        - c_low
                    )

                    *

                    (
                        c
                        - c_low
                    )

                    +

                    i_low

                )

                return round(
                    aqi
                )

        # ----------------------------------------------------
        # Above highest EPA breakpoint
        # ----------------------------------------------------

        if c > 325.4:

            return 500

        return np.nan

    return pm25_series.apply(
        convert
    )


# ============================================================
# 15. DATA QUALITY REPORT
# ============================================================

def print_quality_report(
    data
):

    print()
    print("=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)

    print(
        "Rows:",
        len(data)
    )

    print()

    print(
        "Missing PM2.5:",
        data["pm25"].isna().sum()
    )

    print(
        "Missing AQI:",
        data["aqi"].isna().sum()
    )

    print()

    if not data.empty:

        print(
            "First timestamp:",
            data["datetime_local"].min()
        )

        print(
            "Last timestamp:",
            data["datetime_local"].max()
        )

        print(
            "Minimum PM2.5:",
            data["pm25"].min()
        )

        print(
            "Maximum PM2.5:",
            data["pm25"].max()
        )

        print(
            "Average PM2.5:",
            data["pm25"].mean()
        )

    print()

    # ========================================================
    # Expected hourly timestamps
    # ========================================================

    expected = pd.date_range(

        start=(
            START_TS
            .tz_localize(None)
        ),

        end=(
            END_TS
            + pd.Timedelta(hours=23)
        ).tz_localize(None),

        freq="h"

    )

    actual = pd.to_datetime(
        data["datetime_local"]
    )

    # Remove duplicate timestamps from actual
    # just in case.

    actual = (
        actual
        .drop_duplicates()
        .sort_values()
    )

    missing_hours = (
        expected
        .difference(actual)
    )

    print(
        "Expected hourly rows:",
        len(expected)
    )

    print(
        "Actual hourly rows:",
        len(actual)
    )

    print(
        "Missing hourly rows:",
        len(missing_hours)
    )

    # ========================================================
    # Save missing hours
    # ========================================================

    if len(missing_hours) > 0:

        pd.DataFrame({

            "datetime_local":
                missing_hours

        }).to_csv(

            MISSING_FILE,

            index=False

        )

        print()
        print(
            "Missing hours saved:"
        )

        print(
            MISSING_FILE
        )

    else:

        print()
        print(
            "No missing hourly timestamps."
        )

    print()
    print("=" * 70)


# ============================================================
# 16. SAVE
# ============================================================

def save_data(
    data
):

    print()
    print("=" * 70)
    print("SAVING OPENAQ DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # The existing historical CSV is only overwritten after
    # the complete fetch and quality checks have succeeded.
    # --------------------------------------------------------

    data.to_csv(

        AQI_FILE,

        index=False

    )

    print()
    print(
        "Saved:",
        AQI_FILE
    )

    print(
        "Rows:",
        len(data)
    )

    print(
        "Columns:",
        len(data.columns)
    )


# ============================================================
# 17. MAIN
# ============================================================

def main():

    try:

        data = fetch_openaq()

        if data.empty:

            raise RuntimeError(
                "No OpenAQ data was returned."
            )

        print_quality_report(
            data
        )

        save_data(
            data
        )

        print()
        print("=" * 70)
        print("OPENAQ UPDATE COMPLETE")
        print("=" * 70)

        print()
        print(
            f"Location: "
            f"{OPENAQ_LOCATION_ID} - "
            f"{OPENAQ_LOCATION_NAME}"
        )

        print(
            f"Date range: "
            f"{START_TS.date()} -> "
            f"{END_TS.date()}"
        )

        print(
            f"Output: {AQI_FILE}"
        )

        print()
        print(
            "SUCCESS"
        )

        print(
            "=" * 70
        )

    except requests.HTTPError as e:

        print()
        print("=" * 70)
        print("OPENAQ HTTP ERROR")
        print("=" * 70)

        print(
            str(e)
        )

        print()

        if e.response is not None:

            print(
                "Status:",
                e.response.status_code
            )

            try:

                print(
                    e.response.text[:2000]
                )

            except Exception:

                pass

        raise

    except Exception as e:

        print()
        print("=" * 70)
        print("OPENAQ PIPELINE FAILED")
        print("=" * 70)

        print(
            repr(e)
        )

        raise


# ============================================================
# 18. RUN
# ============================================================

if __name__ == "__main__":

    main()

