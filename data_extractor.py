import ast
import datetime
import json
import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional
import os
import math


########## LOAD DATA ##########
def depack_df(path: Path, key: Optional[str] = None, tag: Optional[str] = None):
    """
    Collects information from what is intended to be a csv file and expands out most values into new columns for use elsewhere.
    :param path: Path to a csv file.
    :param key: Optional key column to filter data if Key column is present.
    :param tag: Optional tag column to filter data if Tag column is present.
    :return: Dataframe of expanded data.
    """

    # load df
    df_temp = pd.read_csv(path)

    # Filter if necessary
    # variable settings are not optimal here but we should be fine given the scope
    if key and tag:
        df_set = df_temp[(df_temp["Tag"] == tag) & (df_temp["Key"] == key)]
    elif key and not tag:
        df_set = df_temp[(df_temp["Key"]) == key]
    elif tag and not key:
        df_set = df_temp[(df_temp["Tag"]) == tag]
    else:
        df_set = df_temp

    # because some files are uniquely strange, capitalize True/False to ensure reading succeeds
    df_set.loc[:, 'Value'] = df_set['Value'].apply(
        lambda x: x.replace(':false', ':False').replace(':true', ':True') if isinstance(x, str) else x)

    # set value column to dictionary
    df_set.loc[:, "Value"] = df_set["Value"].apply(ast.literal_eval)

    # Convert dictionaries to new df
    df_outer = pd.json_normalize(df_set["Value"])

    # re-set index to make merging possible
    df_outer.index = df_set.index

    # merge back together. Keep value json column because why not...
    df_outer = df_set.join(df_outer)

    # standardize datetime if possible
    for cols in df_outer.columns:
        if "time" in cols.lower():
            try:
                df_outer.loc[:, cols] = df_outer[cols].apply(pd.to_datetime, unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                print("NOTE: Datetime not applied to:", cols)

    return (df_outer)


## Create empty dataframes to concat to
df_heartrate = pd.DataFrame()
df_sleep = pd.DataFrame()
df_sport_info = pd.DataFrame()
df_hr_fine = pd.DataFrame()
df_sleep_fine = pd.DataFrame()

# Get filenames in data directory
for filename in os.listdir(os.getcwd() + "\data"):

    if "hlth_center_aggregated_fitness_data.csv" in filename:
        hlthc_agg = Path("data\\" + filename)

        df_heartrate = pd.concat([df_heartrate, depack_df(hlthc_agg, key="heart_rate", tag="daily_report")],
                                 ignore_index=True)
        df_sleep = pd.concat([df_sleep, depack_df(hlthc_agg, key="sleep", tag="daily_report")], ignore_index=True)

    elif "hlth_center_sport_record.csv" in filename:
        hlthc_sport = Path("data\\" + filename)
        df_sport_info = pd.concat([df_sport_info, depack_df(hlthc_sport)], ignore_index=True)

    elif "hlth_center_fitness_data.csv" in filename:
        hlthc_fit = Path("data\\" + filename)
        df_hr_fine = pd.concat([df_hr_fine, depack_df(hlthc_fit, key="heart_rate")], ignore_index=True)
        df_sleep_fine = pd.concat([df_sleep_fine, depack_df(hlthc_fit, key="sleep")], ignore_index=True)

    elif "Weather" in filename:
        df_weather = pd.read_csv(Path("data\\" + filename))

for filename in os.listdir(os.getcwd() + "\data\sgar"):

    if "heart_rate" in filename:
        with open(Path("data\\sgar\\" + filename)) as f:
            data = json.load(f)

            # When including heart rate of multiple days, top keys become dates. So I fixed that...
            all_heart_rate_values = []
            for date_key in data:
                heart_rate_values = data[date_key].get("heartRateValues", [])
                all_heart_rate_values.extend(heart_rate_values)
                uid_sgar = data[date_key].get("userProfilePK", [])

            # Convert to DataFrame
            df_hr_fine_sgar = pd.DataFrame(heart_rate_values, columns=["timestamp", "heartRate"])

            # Convert timestamp from milliseconds to seconds
            df_hr_fine_sgar["timestamp"] = df_hr_fine_sgar["timestamp"] // 1000  # Integer division

            # Convert timestamp to datetime
            df_hr_fine_sgar["timestamp"] = pd.to_datetime(df_hr_fine_sgar["timestamp"], unit="s")

            # TODO: determine if necessary to add heart rate zones here, or do later

    elif "sleep" in filename:
        df_sleep_sgar = pd.read_json(Path("data\\sgar\\" + filename))

        # TODO: adjust the table so that dats are NOT columns... what is this...??????
        # print(df_sleep_sgar.head())

    elif "Activities" in filename:
        df_sport_info_sgar = pd.read_csv(Path("data\\sgar\\" + filename))

#### Merge datasets

# Merge heartrate
df_hr_fine = df_hr_fine.drop(columns=["Sid", "Key", "Time", "Value", "UpdateTime"])
df_hr_fine_sgar = df_hr_fine_sgar.rename(columns={"timestamp": "time", "heartRate": "bpm"})
df_hr_fine_sgar["Uid"] = 123645046
df_hr_fine = pd.concat([df_hr_fine, df_hr_fine_sgar], ignore_index=True)

# Merge sleep
# TODO: MErge sleep but it sucks because it's a lot of columns and very little matches
print(df_sleep.head())
print(df_sleep.columns)
print(df_sleep_sgar.head())
print(df_sleep_sgar.columns)

# Merge activities and remove an activity that was not measured on a watch
# Remove not-watch measured activity (probs not gonna be included anyways due to date but i've yet to filter that)
df_sport_info = df_sport_info[df_sport_info['Sid'].apply(lambda x: isinstance(x, int))]

df_sport_info = df_sport_info.drop(columns=["Sid",
                                            "Key",
                                            "Time",
                                            "Value",
                                            "UpdateTime",
                                            "vitality",
                                            "cloud_course_source",
                                            "did",
                                            "designated_course",
                                            "switch_just_dance_id",
                                            "avg_touchdown_air_ratio",
                                            "half_marathon_grade_prediction_duration",
                                            "fall_height",
                                            "five_kilometre_grade_prediction_duration",
                                            "min_touchdown_air_ratio",
                                            "full_marathon_grade_prediction_duration",
                                            "ten_kilometre_grade_prediction_duration",
                                            "max_cadence",
                                            "max_height",
                                            "max_pace",
                                            "max_speed",
                                            "min_height",
                                            "min_pace",
                                            "proto_type",
                                            "sport_type",
                                            "entityEndTime",
                                            "entityOffsetTime",
                                            "entityStartTime",
                                            "avg_speed",
                                            "running_ability_index",
                                            "rise_height",
                                            "training_status",
                                            "avg_stride",
                                            "avg_height",
                                            "end_time",
                                            "start_time",
                                            "timezone",
                                            "version",
                                            "total_cal",
                                            "avg_cadence",
                                            "avg_pace",
                                            "running_ability_level",
                                            "training_experience",
                                            "valid_duration",
                                            "distance",
                                            "vo2_max_level",
                                            "steps",
                                            "anaerobic_train_effect_level",
                                            "aerobic_train_effect_level",
                                            "train_load_level"
                                            ])

df_sport_info_sgar = df_sport_info_sgar.drop(columns=["Favorite",
                                                      "Best Lap Time",
                                                      "Number of Laps",
                                                      "Title",
                                                      "Distance",
                                                      "Max Cadence",
                                                      "Avg Pace",
                                                      "Best Pace",
                                                      "Total Ascent",
                                                      "Total Descent",
                                                      "Avg Stride Length",
                                                      "Steps",
                                                      "Decompression",
                                                      "Moving Time",
                                                      "Elapsed Time",
                                                      "Min Elevation",
                                                      "Max Elevation",
                                                      "Avg Cadence"
                                                      ])



df_sport_info_sgar = df_sport_info_sgar.rename(columns={"Activity Type": "Category",
                                                        "Date": "time",
                                                        "Time": "duration",
                                                        "Training Stress Score®": "train_load",
                                                        "Calories": "calories",
                                                        "Max HR": "max_hrm",
                                                        "Avg HR": "avg_hrm",
                                                        })
# Add UID
df_sport_info_sgar["Uid"] = uid_sgar

# Note that there will be some NaNs due to sgar data not having the columns but I felt the cols were important to keep
df_sport_info = pd.concat([df_sport_info, df_sport_info_sgar], ignore_index=True)


########## PROCESS ##########

#### RELEVANT DFs:
# df_heartrate: contains aggregated hr data from daily report
# df_sleep: contains aggregated sleep data from daily report
# df_sport_info: contains information about exercise records
# df_hr_fine: contains raw measurements of hr throughout the day
# df_sleep_fine: contains raw measurements of sleep
# df_weather: contains weather data
# df_hr_fine_sgar: heartrate data for sgar
# df_sleep_sgar: sleep data of sgar, try to merge back
# df_sport_info_sgar: activity information for sgar, try to merge back


# Collect overlap times in data
unique_ids = df_heartrate["Uid"].unique()
min_times = []
max_times = []

# Collect min and max times for each student
for ids in unique_ids:
    min_times.append(min(df_hr_fine[df_hr_fine["Uid"] == ids]["time"]))
    max_times.append(max(df_hr_fine[df_hr_fine["Uid"] == ids]["time"]))

# Floor and ceil them so that we get only overlap times as much as possible
min_time = pd.to_datetime(max(min_times)).ceil("D")  # floor to start of day
max_time = pd.to_datetime(min(max_times)).floor("D")  # ceil to start of next day

## If i have time...
# TODO: it is possible to extract heart rate zone per hour vs. just through the day?
# TODO: note how the app calculates hr zones and include it in report

## PRIORITY
# TODO: get+merge sam dataset
# TODO: visualizations


# Zoltan
# TODO: linear regressions for RQ
## Temperature, humidity, UV index, precipitation... Eindhoven, Tilburg, Eersal
