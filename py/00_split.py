###
import datetime
from glob import glob

import pandas as pd
import sklearn as sk

###

def dbg(x):
    print(x)
    return x

rides = pd.concat([
    pd.read_csv(
        dbg(f),
        dtype=dict(
            VendorID=int,
            passenger_count=float,
            trip_distance=float,
            RatecodeID="Int64", # some are missing
            store_and_fwd_flag=str,
            PULocationID=int,
            DOLocationID=int,
            payment_type=int,
            fare_amount=float,
            extra=float,
            mta_tax=float,
            tip_amount=float,
            tolls_amount=float,
            improvement_surcharge=float,
            total_amount=float,
            congestion_surcharge=float,
            Airport_fee=float,
        ),
        parse_dates=[
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
        ],
    )
    for f in sorted(glob("../data/*.csv"))
], ignore_index=True)

###md

# Drop rides that start outside of Manhattan

###

zones = pd.read_csv("../dataset-split/taxi_zone_lookup.csv")
zones = set(zones[zones["Borough"] == "Manhattan"]["LocationID"])

rides = rides[rides["PULocationID"].isin(zones)]

###md

# # Do the split
#
# * Group by week of the month
#   * This grouping strikes a balance of granularity and not being correlated
#     to the target variables; compared to day of the week, hour of day, which
#     is expected to be correlated.
# * Find groups such that ride duration has similar distributions between
#   test and train datasets.
#   * Using the target variable is ideal, but since we have two targets,
#     demand and revenue, trip duration is reasonably correlated to both
#     of these:
#     * longer, in time, trips probably reflect traffic and thus demand.
#     * longer, in time, trips will cost more.
#   * Another reasonable choice is distance, but that's likely less correlated
#     to demand.

###

[(train_idx, test_idx)] = sk.model_selection.GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=2718,
).split(
    rides,
    y=(
        rides["tpep_dropoff_datetime"] - rides["tpep_pickup_datetime"]
    ).map(lambda x: x.seconds / 60).astype(int),
    groups=(
        rides["tpep_pickup_datetime"].map(lambda x: x.day) / 7
    ).astype(int),
)

###

# https://stackoverflow.com/a/17511341
def ceildiv(a, b):
    return -(a // -b)

def save_split(df, prefix, rows_per_file=2300000):
    for i in range(0, ceildiv(len(df), rows_per_file)):
        df[i*rows_per_file:(i+1)*rows_per_file].to_csv(dbg(f"{prefix}_{i:02}.csv"), index=False)

# a sample dataset for faster testing
save_split(rides.iloc[train_idx].sample(1800000), "../dataset-split/sample")
# real training dataset
save_split(rides.iloc[train_idx], "../dataset-split/train")
# test dataset, only used at the very end
save_split(rides.iloc[test_idx], "../dataset-split/test")

