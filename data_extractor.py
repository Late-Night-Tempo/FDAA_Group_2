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

    # set value column to dictionary
    try:
        df_set.loc[:, "Value"] = df_set["Value"].apply(ast.literal_eval)
    except Exception as e:
        # WHAT DO YOU MEAN MALFORMED STRING
        # IT'S ALL AUTOMATIC
        # THERE IS NO MALFORMED STRING IN BA SING SE
        print(path)
        print(e)

        # Does cool prints but guess waht that takes time so deal with it later
        # for idx, value in enumerate(df_set['Value']):
        #     try:
        #         ast.literal_eval(value)
        #     except (ValueError, SyntaxError) as e:
        #         print(f"Error at row {idx}: {repr(value)} - {e}")
        #         print(df_set['Value'].apply(type).value_counts())
        #         for i, char in enumerate(value):
        #             try:
        #                 ast.literal_eval(value[:i + 1])  # Check incrementally
        #             except Exception as sub_e:
        #                 print(f"Fails at position {i}: {repr(char)} - {sub_e}")
        #                 break  # Stop at the first broken character


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

    return(df_outer)



## Create empty dataframes to concat to
df_heartrate = pd.DataFrame()
df_sleep = pd.DataFrame()
df_sport_info = pd.DataFrame()
df_hr_fine = pd.DataFrame()
df_sleep_fine = pd.DataFrame()

# Get filenames in data directory
for filename in os.listdir(os.getcwd()+"\data"):

    if "hlth_center_aggregated_fitness_data.csv" in filename:
        hlthc_agg = Path("data\\"+filename)

        df_heartrate = pd.concat([df_heartrate, depack_df(hlthc_agg, key="heart_rate", tag="daily_report")], ignore_index=True)
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

for filename in os.listdir(os.getcwd()+"\data\sgar"):

    if "heart_rate" in filename:
        with open(Path("data\\sgar\\" + filename)) as f:
            data = json.load(f)

        # Extract only heartRateValues
        heart_rate_values = data.get("heartRateValues", [])

        # Convert to DataFrame
        df_hr_fine_sgar = pd.DataFrame(heart_rate_values, columns=["timestamp", "heartRate"])

        # Convert timestamp from milliseconds to seconds
        df_hr_fine_sgar["timestamp"] = df_hr_fine_sgar["timestamp"] // 1000  # Integer division

        # Convert timestamp to datetime
        df_hr_fine_sgar["timestamp"] = pd.to_datetime(df_hr_fine_sgar["timestamp"], unit="s")

        # Display first few rows
        print(df_hr_fine_sgar.head())

    elif "Sleep" in filename:
        print("sleep")

    elif "walking" in filename:
        df_sport_info_sgar = pd.read_csv(Path("data\\sgar\\" + filename))
        print(df_sport_info_sgar.head())



########## PROCESS ##########

#### RELEVANT DFs:
# df_heartrate: contains aggregated hr data from daily report
# df_sleep: contains aggregated sleep data from daily report
# df_sport_info: contains information about exercise records
# df_hr_fine: contains raw measurements of hr throughout the day
# df_sleep_fine: contains raw measurements of sleep

# Collect overlap times in data
unique_ids = df_heartrate["Uid"].unique()
min_times = []
max_times = []

# Collect min and max times for each student
for ids in unique_ids:
    min_times.append(min(df_hr_fine[df_hr_fine["Uid"] == ids]["Time"]))
    max_times.append(max(df_hr_fine[df_hr_fine["Uid"] == ids]["Time"]))

# Print min max days
print("Overlap", max(min_times), min(max_times))

# Collect time difference
setup = [max(min_times), min(max_times)]
datetime_objects = pd.to_datetime(setup)
time_difference = max(datetime_objects) - min(datetime_objects)
print(f"Time difference: {time_difference}")


print(len(df_sport_info))

print(df_sleep.iloc[0])




## If i have time...
# TODO: it is possible to extract heart rate zone per hour vs. just through the day?
# TODO: note how the app calculates hr zones and include it in report

## PRIORITY
# TODO: begin making linregs
# TODO: get+merge sam dataset
# TODO: visualizations


# Zoltan
# TODO: linear regressions for RQ
# TODO: See how to merge weather dataset w/ mi fit data (probably just visualize via date) (tip to zoltan: look for previous 2 weeks, not predictions)
## Temperature, humidity, UV index, precipitation... Eindhoven, Tilburg, Eersal