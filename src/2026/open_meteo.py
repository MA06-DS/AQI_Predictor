# ============================================================
# OPEN-METEO - KARACHI WEATHER DATA FETCHER
# ============================================================
#
# Output:
#   data/2026/karachi_2026_weather.csv
#
# Features:
#
#   datetime_local
#   temperature
#   humidity
#   pressure
#   wind_speed
#   wind_direction
#   precipitation
#   cloud_cover
#   datetime_utc
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import time
import requests
import pandas as pd

from pathlib import Path


# ============================================================
# 2. CONFIGURATION
# ============================================================

CITY = "Karachi"

LATITUDE = 24.8607
LONGITUDE = 67.0011

TIMEZONE = "Asia/Karachi"


# ============================================================
# 3. DATE RANGE
# ============================================================

START_DATE = "2026-01-01"

# None = automatically use yesterday
END_DATE = None


# ============================================================
# 4. OPEN-METEO API
# ============================================================

OPEN_METEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)


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


WEATHER_FILE = (
    OUTPUT_DIR /
    "karachi_2026_weather.csv"
)


MISSING_FILE = (
    OUTPUT_DIR /
    "karachi_2026_missing_weather_hours.csv"
)


# ============================================================
# 6. OPEN-METEO VARIABLES
# ============================================================
#
# Open-Meteo API variable
#       ↓
# Our CSV column
#
# temperature_2m
#       ↓
# temperature
#
# relative_humidity_2m
#       ↓
# humidity
#
# surface_pressure
#       ↓
# pressure
#
# wind_speed_10m
#       ↓
# wind_speed
#
# wind_direction_10m
#       ↓
# wind_direction
#
# precipitation
#       ↓
# precipitation
#
# cloud_cover
#       ↓
# cloud_cover
#
# ============================================================

OPEN_METEO_VARIABLES = [

    "temperature_2m",

    "relative_humidity_2m",

    "surface_pressure",

    "wind_speed_10m",

    "wind_direction_10m",

    "precipitation",

    "cloud_cover",

]


# ============================================================
# 7. GET DATE RANGE
# ============================================================

def get_date_range():

    start = pd.Timestamp(
        START_DATE
    )

    if END_DATE is None:

        # Current date in Karachi
        today_local = (
            pd.Timestamp.now(
                tz=TIMEZONE
            )
            .normalize()
        )

        # Use yesterday so today's incomplete
        # data is not included.
        end = (
            today_local
            - pd.Timedelta(
                days=1
            )
        )

        # Remove timezone
        end = pd.Timestamp(
            end.date()
        )

    else:

        end = pd.Timestamp(
            END_DATE
        )

    if end < start:

        raise ValueError(
            "END_DATE cannot be earlier "
            "than START_DATE."
        )

    return start, end


START_TS, END_TS = get_date_range()


# ============================================================
# 8. PRINT CONFIGURATION
# ============================================================

print()
print("=" * 70)
print("OPEN-METEO KARACHI WEATHER FETCHER")
print("=" * 70)

print(
    f"City        : {CITY}"
)

print(
    f"Latitude    : {LATITUDE}"
)

print(
    f"Longitude   : {LONGITUDE}"
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

print()
print(
    "Features:"
)

print(
    "  - temperature"
)

print(
    "  - humidity"
)

print(
    "  - pressure"
)

print(
    "  - wind_speed"
)

print(
    "  - wind_direction"
)

print(
    "  - precipitation"
)

print(
    "  - cloud_cover"
)

print(
    "=" * 70
)


# ============================================================
# 9. CREATE MONTHLY CHUNKS
# ============================================================

def create_monthly_chunks(
    start_date,
    end_date
):

    chunks = []

    current = start_date

    while current <= end_date:

        # First day of current month
        month_start = current

        # First day of next month
        next_month = (
            current
            + pd.offsets.MonthBegin(1)
        )

        # Last day of current month
        month_end = min(

            next_month
            - pd.Timedelta(
                days=1
            ),

            end_date

        )

        chunks.append(
            (
                month_start,
                month_end
            )
        )

        current = (
            month_end
            + pd.Timedelta(
                days=1
            )
        )

    return chunks


# ============================================================
# 10. FETCH ONE MONTH
# ============================================================

def fetch_weather_chunk(
    chunk_start,
    chunk_end
):

    print()
    print("-" * 70)

    print(
        f"Fetching:"
    )

    print(
        f"  {chunk_start.date()} "
        f"-> "
        f"{chunk_end.date()}"
    )

    print(
        "-" * 70
    )


    # ========================================================
    # API PARAMETERS
    # ========================================================

    params = {

        "latitude":
            LATITUDE,

        "longitude":
            LONGITUDE,

        "start_date":
            chunk_start.strftime(
                "%Y-%m-%d"
            ),

        "end_date":
            chunk_end.strftime(
                "%Y-%m-%d"
            ),

        "hourly":
            ",".join(
                OPEN_METEO_VARIABLES
            ),

        "timezone":
            TIMEZONE,

    }


    print(
        "Requesting Open-Meteo..."
    )


    # ========================================================
    # REQUEST
    # ========================================================

    response = requests.get(

        OPEN_METEO_URL,

        params=params,

        timeout=120

    )


    response.raise_for_status()


    payload = response.json()


    # ========================================================
    # CHECK RESPONSE
    # ========================================================

    if "hourly" not in payload:

        print()
        print(
            "Open-Meteo response:"
        )

        print(
            payload
        )

        raise RuntimeError(
            "Open-Meteo response does not "
            "contain hourly data."
        )


    hourly = payload[
        "hourly"
    ]


    # ========================================================
    # CONVERT TO DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        hourly
    )


    if df.empty:

        print(
            "No data returned."
        )

        return pd.DataFrame()


    print(
        f"Rows received: "
        f"{len(df)}"
    )


    return df


# ============================================================
# 11. FETCH ALL WEATHER DATA
# ============================================================

def fetch_open_meteo():

    # --------------------------------------------------------
    # Create monthly chunks
    # --------------------------------------------------------

    chunks = create_monthly_chunks(

        START_TS,

        END_TS

    )


    print()
    print(
        f"Total monthly chunks: "
        f"{len(chunks)}"
    )


    all_data = []


    # ========================================================
    # FETCH EACH MONTH
    # ========================================================

    for index, (
        chunk_start,
        chunk_end
    ) in enumerate(
        chunks,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(chunks)}]"
        )


        try:

            df = fetch_weather_chunk(

                chunk_start,

                chunk_end

            )


            if not df.empty:

                all_data.append(
                    df
                )


        except requests.HTTPError as e:

            print()
            print(
                "OPEN-METEO HTTP ERROR"
            )

            print(
                str(e)
            )

            if e.response is not None:

                print(
                    "Status:",
                    e.response.status_code
                )

                print(
                    e.response.text[:2000]
                )

            raise


        # Small delay
        time.sleep(
            0.5
        )


    # ========================================================
    # CHECK DATA
    # ========================================================

    if not all_data:

        raise RuntimeError(
            "Open-Meteo returned no weather data."
        )


    # ========================================================
    # COMBINE MONTHS
    # ========================================================

    data = pd.concat(

        all_data,

        ignore_index=True

    )


    print()
    print("=" * 70)
    print("COMBINED WEATHER DATA")
    print("=" * 70)


    print(
        "Raw rows:",
        len(data)
    )


    # ========================================================
    # RENAME COLUMNS
    # ========================================================

    data = data.rename(

        columns={

            "time":
                "datetime_local",

            "temperature_2m":
                "temperature",

            "relative_humidity_2m":
                "humidity",

            "surface_pressure":
                "pressure",

            "wind_speed_10m":
                "wind_speed",

            "wind_direction_10m":
                "wind_direction",

        }

    )


    # ========================================================
    # CONVERT LOCAL DATETIME
    # ========================================================
    #
    # Open-Meteo returns the requested local timezone
    # because timezone=Asia/Karachi was supplied.
    #
    # The timestamp is therefore local Karachi time.
    #
    # ========================================================

    data["datetime_local"] = pd.to_datetime(

        data["datetime_local"],

        errors="coerce"

    )


    # ========================================================
    # REMOVE INVALID TIMESTAMPS
    # ========================================================

    data = data[
        data["datetime_local"].notna()
    ].copy()


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    data = (

        data

        .drop_duplicates(
            subset=[
                "datetime_local"
            ]
        )

        .sort_values(
            "datetime_local"
        )

        .reset_index(
            drop=True
        )

    )


    # ========================================================
    # KEEP ONLY REQUESTED DATE RANGE
    # ========================================================

    start_naive = pd.Timestamp(
        START_DATE
    )


    end_naive = (

        END_TS

    )


    data = data[

        (
            data["datetime_local"]
            >= start_naive
        )

        &

        (
            data["datetime_local"]
            <
            (
                end_naive
                + pd.Timedelta(
                    days=1
                )
            )
        )

    ].copy()


    # ========================================================
    # CHECK AFTER FILTER
    # ========================================================

    print(
        "Rows after date filtering:",
        len(data)
    )


    if data.empty:

        raise RuntimeError(
            "No weather data remains after "
            "date filtering."
        )


    # ========================================================
    # CREATE UTC TIMESTAMP
    # ========================================================
    #
    # datetime_local is Karachi local time.
    #
    # First localize it to Asia/Karachi.
    # Then convert it to UTC.
    #
    # IMPORTANT:
    #
    # We use tz_localize here because datetime_local
    # is timezone-naive.
    #
    # ========================================================

    local_aware = (

        data["datetime_local"]

        .dt.tz_localize(
            TIMEZONE
        )

    )


    data["datetime_utc"] = (

        local_aware

        .dt.tz_convert(
            "UTC"
        )

    )


    # ========================================================
    # REMOVE TIMEZONE INFORMATION FOR CSV
    # ========================================================

    data["datetime_local"] = (

        data["datetime_local"]

        .dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    )


    data["datetime_utc"] = (

        data["datetime_utc"]

        .dt.tz_localize(
            None
        )

        .dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    )


    # ========================================================
    # FINAL COLUMNS
    # ========================================================

    final_columns = [

        "datetime_local",

        "temperature",

        "humidity",

        "pressure",

        "wind_speed",

        "wind_direction",

        "precipitation",

        "cloud_cover",

        "datetime_utc",

    ]


    data = data[
        final_columns
    ]


    # ========================================================
    # FINAL SORT
    # ========================================================

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
# 12. DATA QUALITY REPORT
# ============================================================

def print_quality_report(
    data
):

    print()
    print("=" * 70)
    print("OPEN-METEO DATA QUALITY REPORT")
    print("=" * 70)


    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    print(
        "Rows:",
        len(data)
    )


    print(
        "Columns:",
        len(data.columns)
    )


    print()


    print(
        "First timestamp:",
        data[
            "datetime_local"
        ].min()
    )


    print(
        "Last timestamp:",
        data[
            "datetime_local"
        ].max()
    )


    # ========================================================
    # MISSING VALUES
    # ========================================================

    print()
    print(
        "MISSING VALUES"
    )


    print(
        "-" * 70
    )


    missing = data.isna().sum()


    for column, count in missing.items():

        print(
            f"{column:25s}: "
            f"{count}"
        )


    # ========================================================
    # EXPECTED HOURLY DATA
    # ========================================================

    expected = pd.date_range(

        start=pd.Timestamp(
            START_DATE
        ),

        end=(
            END_TS
            + pd.Timedelta(
                hours=23
            )
        ),

        freq="h"

    )


    actual = (

        pd.to_datetime(
            data[
                "datetime_local"
            ]
        )

        .drop_duplicates()

    )


    missing_hours = (

        expected

        .difference(
            actual
        )

    )


    print()
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
    # SAVE MISSING HOURS
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


    # ========================================================
    # SAMPLE
    # ========================================================

    print()
    print(
        "DATA SAMPLE"
    )


    print(
        "-" * 70
    )


    print(
        data.head()
    )


    print()
    print(
        "=" * 70
    )


# ============================================================
# 13. SAVE DATA
# ============================================================

def save_data(
    data
):

    print()
    print("=" * 70)
    print("SAVING OPEN-METEO DATA")
    print("=" * 70)


    data.to_csv(

        WEATHER_FILE,

        index=False

    )


    print()
    print(
        "Saved:",
        WEATHER_FILE
    )


    print(
        "Rows:",
        len(data)
    )


    print(
        "Columns:",
        len(data.columns)
    )


    print()
    print(
        "Final columns:"
    )


    for column in data.columns:

        print(
            f"  - {column}"
        )


# ============================================================
# 14. MAIN
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # Fetch
        # ----------------------------------------------------

        data = fetch_open_meteo()


        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        if data.empty:

            raise RuntimeError(
                "No Open-Meteo data was returned."
            )


        # ----------------------------------------------------
        # Quality report
        # ----------------------------------------------------

        print_quality_report(
            data
        )


        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_data(
            data
        )


        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("OPEN-METEO UPDATE COMPLETE")
        print("=" * 70)


        print()
        print(
            f"Location: "
            f"{CITY}"
        )


        print(
            f"Coordinates: "
            f"{LATITUDE}, "
            f"{LONGITUDE}"
        )


        print(
            f"Date range: "
            f"{START_TS.date()} "
            f"-> "
            f"{END_TS.date()}"
        )


        print(
            f"Output: "
            f"{WEATHER_FILE}"
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
        print("OPEN-METEO HTTP ERROR")
        print("=" * 70)


        print(
            str(e)
        )


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
        print("OPEN-METEO PIPELINE FAILED")
        print("=" * 70)


        print(
            repr(e)
        )


        raise


# ============================================================
# 15. RUN
# ============================================================

if __name__ == "__main__":

    main()

