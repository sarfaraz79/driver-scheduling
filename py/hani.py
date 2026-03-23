import pandas as pd

# load dataset
df = pd.read_csv("nyc_taxi_2024-01.csv")

# quick check
df.head()

###

print(df.shape)
print(df.columns)

df.head()

###

# checking data types
df.dtypes

###

# checking missing values
df.isna().sum()

###

# converting pickup and dropoff columns to proper datetime format
df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")

# check the result
df[["tpep_pickup_datetime", "tpep_dropoff_datetime"]].dtypes

###

# checking whether any datetime values became missing after conversion
df[["tpep_pickup_datetime", "tpep_dropoff_datetime"]].isna().sum()

###

# creating trip duration in minutes
df["trip_duration_min"] = (
    (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
)

# summary
df["trip_duration_min"].describe()

###

# counting potentially invalid trips
print("Negative duration trips:", (df["trip_duration_min"] < 0).sum())
print("Zero duration trips:", (df["trip_duration_min"] == 0).sum())

# Checking very long trips
print("Trips longer than 180 minutes:", (df["trip_duration_min"] > 180).sum())

###

# checking key numeric columns
numeric_cols = ["trip_distance", "fare_amount", "total_amount", "tip_amount"]

df[numeric_cols].describe()

###

# counting suspicious values
print("Negative trip distance:", (df["trip_distance"] < 0).sum())
print("Zero trip distance:", (df["trip_distance"] == 0).sum())

print("Negative fare amount:", (df["fare_amount"] < 0).sum())
print("Zero fare amount:", (df["fare_amount"] == 0).sum())

print("Negative total amount:", (df["total_amount"] < 0).sum())
print("Zero total amount:", (df["total_amount"] == 0).sum())

print("Negative tip amount:", (df["tip_amount"] < 0).sum())

###

# making a cleaned copy
df_clean = df.copy()

# keeping only trips with sensible core values
df_clean = df_clean[
    (df_clean["trip_duration_min"] > 0) &
    (df_clean["trip_distance"] >= 0) &
    (df_clean["fare_amount"] >= 0) &
    (df_clean["total_amount"] >= 0) &
    (df_clean["tip_amount"] >= 0)
]

# new shape
print("Original shape:", df.shape)
print("Cleaned shape:", df_clean.shape)
print("Rows removed:", df.shape[0] - df_clean.shape[0])

###

print("Negative duration trips:", (df_clean["trip_duration_min"] < 0).sum())
print("Zero duration trips:", (df_clean["trip_duration_min"] == 0).sum())
print("Negative fare amount:", (df_clean["fare_amount"] < 0).sum())
print("Negative total amount:", (df_clean["total_amount"] < 0).sum())
print("Negative tip amount:", (df_clean["tip_amount"] < 0).sum())

###

# extracting/making time-based features from pickup datetime
df_clean["pickup_hour"] = df_clean["tpep_pickup_datetime"].dt.hour
df_clean["pickup_dayofweek"] = df_clean["tpep_pickup_datetime"].dt.dayofweek
df_clean["pickup_month"] = df_clean["tpep_pickup_datetime"].dt.month

# weekend indicator (1 = weekend, 0 = weekday)
df_clean["is_weekend"] = df_clean["pickup_dayofweek"].isin([5, 6]).astype(int)

# check
df_clean[["pickup_hour", "pickup_dayofweek", "pickup_month", "is_weekend"]].head()

###

df_clean["pickup_hour"].unique()

###

# list of Manhattan taxi zone IDs (maybe use the lookup table, need to download the csv file (not provided))
manhattan_zones = [
    4, 12, 13, 24, 41, 42, 43, 45, 48, 50,
    68, 74, 75, 79, 87, 88, 90, 100, 107, 113,
    114, 116, 120, 125, 137, 140, 141, 142, 143,
    144, 148, 151, 152, 158, 161, 162, 163, 164,
    166, 170, 186, 194, 202, 209, 211, 224, 229,
    230, 231, 232, 233, 234, 236, 237, 238, 239,
    246, 249, 261, 262, 263
]

###

df_manhattan = df_clean[df_clean["PULocationID"].isin(manhattan_zones)].copy()

# result
print("Original rows:", df_clean.shape[0])
print("Manhattan rows:", df_manhattan.shape[0])

###

# creating a datetime rounded to the hour
df_manhattan["pickup_hour_dt"] = df_manhattan["tpep_pickup_datetime"].dt.floor("h")

# aggregating to zone-hour level
df_zone_hour = df_manhattan.groupby(
    ["PULocationID", "pickup_hour_dt"]
).agg(
    trip_count=("VendorID", "count"), # DEMAND VARIABLE (IMPORTANT)
    total_revenue=("total_amount", "sum"),
    avg_fare=("fare_amount", "mean"),
    avg_distance=("trip_distance", "mean")
).reset_index()

# result
df_zone_hour.head()

###

df_zone_hour.describe()
df_zone_hour.shape

###

# restricting to January 2024 only
df_manhattan = df_manhattan[
    (df_manhattan["tpep_pickup_datetime"] >= "2024-01-01") &
    (df_manhattan["tpep_pickup_datetime"] < "2024-02-01")
].copy()

# check new min/max
print(df_manhattan["tpep_pickup_datetime"].min())
print(df_manhattan["tpep_pickup_datetime"].max())

###

df_manhattan["pickup_hour_dt"] = df_manhattan["tpep_pickup_datetime"].dt.floor("h")

###

df_zone_hour = df_manhattan.groupby(
    ["PULocationID", "pickup_hour_dt"]
).agg(
    trip_count=("VendorID", "count"),
    total_revenue=("total_amount", "sum")
).reset_index()

###

all_hours = pd.date_range(
    start="2024-01-01",
    end="2024-01-31 23:00:00",
    freq="h"
)

###

all_zones = sorted(df_manhattan["PULocationID"].unique())

full_index = pd.MultiIndex.from_product(
    [all_zones, all_hours],
    names=["PULocationID", "pickup_hour_dt"]
)

df_zone_hour_full = (
    df_zone_hour
    .set_index(["PULocationID", "pickup_hour_dt"])
    .reindex(full_index)
    .reset_index()
)

# fill targets
df_zone_hour_full["trip_count"] = df_zone_hour_full["trip_count"].fillna(0)
df_zone_hour_full["total_revenue"] = df_zone_hour_full["total_revenue"].fillna(0)

###

df_zone_hour_full.shape

###

df_zone_hour_full.head()

###

df_zone_hour_full.describe()

###

# create time features from the hourly datetime
df_zone_hour_full["hour"] = df_zone_hour_full["pickup_hour_dt"].dt.hour
df_zone_hour_full["dayofweek"] = df_zone_hour_full["pickup_hour_dt"].dt.dayofweek
df_zone_hour_full["month"] = df_zone_hour_full["pickup_hour_dt"].dt.month
df_zone_hour_full["day"] = df_zone_hour_full["pickup_hour_dt"].dt.day

# weekend flag
df_zone_hour_full["is_weekend"] = df_zone_hour_full["dayofweek"].isin([5, 6]).astype(int)

# check
df_zone_hour_full.head()

###

df_zone_hour_full["day"].unique()

###

df_zone_hour_full[["pickup_hour_dt", "hour", "dayofweek", "day", "month", "is_weekend"]].head()

###

# sorting first so lags are created in the right order
df_zone_hour_full = df_zone_hour_full.sort_values(["PULocationID", "pickup_hour_dt"])

# lag features for trip_count
df_zone_hour_full["trip_count_lag_1"] = df_zone_hour_full.groupby("PULocationID")["trip_count"].shift(1)
df_zone_hour_full["trip_count_lag_24"] = df_zone_hour_full.groupby("PULocationID")["trip_count"].shift(24)

# lag features for total_revenue
df_zone_hour_full["total_revenue_lag_1"] = df_zone_hour_full.groupby("PULocationID")["total_revenue"].shift(1)
df_zone_hour_full["total_revenue_lag_24"] = df_zone_hour_full.groupby("PULocationID")["total_revenue"].shift(24)

# check
df_zone_hour_full.head(30)

###

# rolling mean for trip_count (past 3 hours)
df_zone_hour_full["trip_count_roll_3"] = (
    df_zone_hour_full.groupby("PULocationID")["trip_count"]
    .rolling(window=3)
    .mean()
    .reset_index(level=0, drop=True)
)

# rolling mean for trip_count (past 24 hours)
df_zone_hour_full["trip_count_roll_24"] = (
    df_zone_hour_full.groupby("PULocationID")["trip_count"]
    .rolling(window=24)
    .mean()
    .reset_index(level=0, drop=True)
)

# rolling mean for revenue
df_zone_hour_full["revenue_roll_3"] = (
    df_zone_hour_full.groupby("PULocationID")["total_revenue"]
    .rolling(window=3)
    .mean()
    .reset_index(level=0, drop=True)
)

df_zone_hour_full["revenue_roll_24"] = (
    df_zone_hour_full.groupby("PULocationID")["total_revenue"]
    .rolling(window=24)
    .mean()
    .reset_index(level=0, drop=True)
)

# check
df_zone_hour_full.head(30)

###

import matplotlib.pyplot as plt

# average demand by hour
hourly_demand = df_zone_hour_full.groupby("hour")["trip_count"].mean()

plt.figure()
hourly_demand.plot()
plt.title("Average Demand by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Trip Count")
plt.show()

###

hourly_revenue = df_zone_hour_full.groupby("hour")["total_revenue"].mean()

plt.figure()
hourly_revenue.plot()
plt.title("Average Revenue by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average Revenue")
plt.show()

###

dow_demand = df_zone_hour_full.groupby("dayofweek")["trip_count"].mean()

plt.figure()
dow_demand.plot()
plt.title("Average Demand by Day of Week")
plt.xlabel("Day of Week (0=Mon)")
plt.ylabel("Average Trip Count")
plt.show()

###

top_zones = df_zone_hour_full.groupby("PULocationID")["trip_count"].mean().sort_values(ascending=False).head(10)

top_zones.plot(kind="bar")
plt.title("Top 10 Zones by Average Demand")
plt.ylabel("Average Trip Count")
plt.show()

###

plt.scatter(df_zone_hour_full["trip_count"], df_zone_hour_full["total_revenue"], alpha=0.1)
plt.xlabel("Trip Count")
plt.ylabel("Total Revenue")
plt.title("Demand vs Revenue")
plt.show()

###

print(df_zone_hour_full.shape)
print(df_zone_hour_full.columns)
print(df_zone_hour_full.dtypes)

###

df_zone_hour_full.isna().sum()

###

# checking duplicate zone hours:
df_zone_hour_full.duplicated(subset=["PULocationID", "pickup_hour_dt"]).sum()

###

# checking date range
print(df_zone_hour_full["pickup_hour_dt"].min())
print(df_zone_hour_full["pickup_hour_dt"].max())

###

# checking zones and hours count:
print("Number of unique zones:", df_zone_hour_full["PULocationID"].nunique())
print("Number of unique hours:", df_zone_hour_full["pickup_hour_dt"].nunique())

###

# checking core targets for impossible values
print("Negative trip_count:", (df_zone_hour_full["trip_count"] < 0).sum())
print("Negative total_revenue:", (df_zone_hour_full["total_revenue"] < 0).sum())

###

# summary of target variables
df_zone_hour_full[["trip_count", "total_revenue"]].describe()

###

# lag missingness
lag_cols = ["trip_count_lag_1", "trip_count_lag_24", "total_revenue_lag_1", "total_revenue_lag_24"]
df_zone_hour_full[lag_cols].isna().sum()

###

# checking order:
df_zone_hour_full[["PULocationID", "pickup_hour_dt"]].head(10)

###



###

