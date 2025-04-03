import ast
import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional
import os


## Collect heart rate zones, sleep data, and physical activity (determine if can use hr zones)

## make a function to make this easier

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

    return(df_outer)


# set paths for the csvs to be loaded
# TODO: automate loading paths and associated dataframe filtering settings
# note that agg and fit get used twice to get sleep + hr data...

# open folder
# for items in folder, if item ends in """""", do process
for filename in os.listdir(os.getcwd()+"\data"):
   print(filename)

hlthc_agg = Path("hlth_center_aggregated_fitness_data.csv")
hlthc_sport = Path("hlth_center_sport_record.csv")
hlthc_fit = Path("hlth_center_fitness_data.csv")

#### RELEVANT DFs:
# df_heartrate: contains aggregated hr data from daily report
# df_sleep: contains aggregated sleep data from daily report
# df_sport_info: contains information about exercise records
# df_hr_fine: contains raw measurements of hr throughout the day
# df_sleep_fine: contains raw measurements of sleep

# set dataframes
df_heartrate = depack_df(hlthc_agg, key="heart_rate", tag="daily_report")
df_sleep = depack_df(hlthc_agg, key="sleep", tag="daily_report")
df_sport_info = depack_df(hlthc_sport)
df_hr_fine = depack_df(hlthc_fit, key="heart_rate")
df_sleep_fine = depack_df(hlthc_fit, key="sleep")
df_weather = pd.read_csv(Path("data/Daily Weather Data.csv"))

print(df_weather.head)

# print(df_heartrate.iloc[0])
# print(df_sleep.iloc[0])
# print(df_sport_info.iloc[1])
# print(df_hr_fine.iloc[0])
# print(df_sleep_fine.iloc[0])



## If i have time...
# TODO: make visualizations of different attributes for the data in regards to RQ
# TODO: determine if possible to extract heart rate zone per hour vs. just through the day? (ask tutor which is easier)

## MAIN PRIORITIES
# TODO: get data start and end points for datetime (so we have our 2wk+ measurements overlap) determine mathematically via datetime
# TODO: Check sam watch data (get single day data)


# Zoltan
# TODO: linear regressions for RQ
# TODO: See how to merge weather dataset w/ mi fit data (probably just visualize via date) (tip to zoltan: look for previous 2 weeks, not predictions)
## Temperature, humidity, UV index, precipitation... Eindhoven, Tilburg, Eersal