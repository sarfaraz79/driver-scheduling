###
import numpy as np
import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import os
import glob

###

if "get_ipython" not in globals() or "get_ipython" in globals() and get_ipython().__class__.__name__ == "TerminalInteractiveShell":
    matplotlib.use("Agg")

def plot(figsize=(6.4, 4.8), **kwargs):
    assert DATA_SUBSET != "test"

    plt.gcf().set_size_inches(*figsize)
    if "get_ipython" not in globals() or get_ipython().__class__.__name__ == "TerminalInteractiveShell":
        plt.savefig("a.png", **kwargs)
        plt.clf()
        plt.cla()
    else:
        plt.show()
    plt.close()

def dbg(x):
    print(x)
    return x

###

def load_split(pat):
    return pd.concat([
        pd.read_csv(
            dbg(name),
            dtype=dict(
                VendorID=int,
                passenger_count=float,
                trip_distance=float,
                RatecodeID="category",
                store_and_fwd_flag=str,
                PULocationID=int, # "category",
                DOLocationID=int, # "category",
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
        for name in sorted(glob.glob(f"../dataset-split/{pat}"))
    ], ignore_index=True)

###

zones = pd.read_csv("../dataset-split/taxi_zone_lookup.csv")
zones = set(zones[zones["Borough"] == "Manhattan"]["LocationID"])

###

# DATA_SUBSET = "sample"
DATA_SUBSET = "train"

# NOTE: **DO NOT USE FOR GENERATING GRAPHS**
# This is only allowed here so that the actual data cleaning does not
# have to be copied
# DATA_SUBSET = "test"

rides = load_split(f"{DATA_SUBSET}_*")
# rides = rides[rides["PULocationID"].isin(zones)]

###

rides.head()

###

rides.describe()

###

print(rides.isna().mean() * 100)

###

missing_by_vendor = (
    rides
    .groupby("VendorID")
    .agg(lambda x: x.isna().mean())
    * 100
).round(5)

missing_by_vendor = missing_by_vendor.loc[:, (missing_by_vendor > 0).any()]

print(missing_by_vendor)

###

###

rides.info()

# Convert Y/N to boolean
rides['store_and_fwd_flag'] = (
    rides['store_and_fwd_flag']
    .map({'Y': True, 'N': False})
    .astype('boolean') # just for reference clarity
)

###

rides.dtypes

###md

# Add some computed features
# * duration, in hours
# * avg speed
#
# Generally, this kind of thing should be part of feature engineering
# but these features can indicate invalid data (eg. unrealistic speeds or negative
# durations) and we'd like to remove
# that as early as possible.

###

rides = rides.assign(
    duration_hours=lambda x:
        (
            x['tpep_dropoff_datetime'] - x['tpep_pickup_datetime']
        ).dt.total_seconds() / 3600,
).assign(
    avg_speed_mph=lambda x:
        x['trip_distance'] / x['duration_hours'].where(lambda x: x > 0),
    fare_per_hour=lambda x:
        x["fare_amount"] / x["duration_hours"].where(lambda x: x > 0),
)

###

# https://stackoverflow.com/a/17511341
def ceildiv(a, b):
    return -(a // -b)

def nplots(n, cols):
    if DATA_SUBSET == "test":
        return None

    rows = ceildiv(n, cols)
    figsize=(6 * cols, 4 * rows)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for i, ax in enumerate(axes.flatten()):
        if i >= n:
            fig.delaxes(ax)
            continue

        yield i, ax

    plt.tight_layout()
    plot(figsize=figsize)


# label the cateogricals
raw_categorical = [
    "VendorID",
    "RatecodeID",
    "payment_type",
]

for i, ax in nplots(len(raw_categorical), 2):
    feature = raw_categorical[i]

    counts = rides[feature].value_counts(dropna=False).sort_index()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(feature)
    ax.set_xlabel(feature)
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45)

###

# check distributions of target variables of the target variables

###

raw_numeric = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "avg_speed_mph",
    "duration_hours",
    "extra",
    "mta_tax",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "Airport_fee",
    "fare_per_hour",
]

# Construct some binned histograms of the numerical variables so we can see the distribution and observe outliers.

for i, ax in nplots(len(raw_numeric), 4):
    feature = raw_numeric[i]
    ax.hist(rides[feature].dropna(), bins=50)
    ax.set_title(feature)
    ax.set_xlabel(feature)
    ax.set_ylabel("count")

###md

# Same but with log scale

###

for i, ax in nplots(len(raw_numeric), 4):
    feature = raw_numeric[i]
    ax.hist(rides[feature].dropna(), bins=50, log=True)
    ax.set_title(feature)
    ax.set_xlabel(feature)
    ax.set_ylabel("count")

###



###

# Based on the above, we want to investigate trip_distance, fare_amount, mta_tax, total_amount, duration_hours, avg_speed_mph

# All negative values must be removed.

# Need to determine cut-off points for the others, to prevent enormous outliers from biasing the models.
# Below, we try some potential cut-offs and oberve ho

print("Counts of extreme / invalid values:\n")

checks = {
    "total_amount > 1000"  : ("total_amount"  , lambda x: x >  1000),
    "total_amount < 0.0"   : ("total_amount"  , lambda x: x <= 0.0),
    "avg_speed_mph > 150"  : ("avg_speed_mph" , lambda x: x >  150),
    "avg_speed_mph < 1"    : ("avg_speed_mph" , lambda x: x <  1),
    "duration_hours > 12"  : ("duration_hours", lambda x: x >  12),
    "duration_hours < 0.01": ("duration_hours", lambda x: x <  0.01),
}

for name, (col, condition) in checks.items():
    full_col = rides[col]
    flagged = full_col[condition(full_col)]
    count = len(flagged)

    if count > 0:
        # quantile position in the FULL distribution
        q_low = (full_col <= flagged.min()).mean()
        q_high = (full_col <= flagged.max()).mean()

        print(
            f"{name}: {count} | "
            f"value range: {flagged.min():.4f} to {flagged.max():.4f} | "
            f"full-sample quantile range: {q_low:.4f} to {q_high:.4f}"
        )
    else:
        print(f"{name}: {count}")

###

def get_constraint_violations(df, pad=1):

    violations = {}
    # These are the cut-offs we apply to the data. They are decided and refined using statistical assessment in 3.0 and checked using 3.2
    # The conditions describe what data will be flagged
    possible_checks = {
        'passenger_count': lambda x: x < 0,
        'trip_distance': lambda x: (x <= 0) | (x > 40 * pad), # to ensure the taxi actually moved
        'fare_amount': lambda x: (x < 0) | (x > 100 * pad),
        'fare_per_hour': lambda x: (x < 0) | (x > 1000 * pad),
        'extra': lambda x: x < 0,
        'mta_tax': lambda x: x < 0,
        'tip_amount': lambda x: (x < 0) | (x > 40 * pad),
        'tolls_amount': lambda x: (x < 0) | (x > 40 * pad),
        'improvement_surcharge': lambda x: x < 0,
        'total_amount': lambda x: (x < 0) | (x > 250 * pad),
        'congestion_surcharge': lambda x: x < 0,
        'Airport_fee': lambda x: x < 0,
        'duration_hours': lambda x: (x < 0.01) | (x > 2 * pad), # to remove any entries which took zero time
        'avg_speed_mph': lambda x: (x <= 0) | (x > 40 * pad), # to remove any entries which suggest the taxi has moved unrealistically far in a given time.
        'tpep_pickup_datetime': lambda x: (x < "2024-01-01") | (x >= "2025-01-01")
    }

    # Column-based checks
    for col, condition in possible_checks.items():
        if col in df.columns:
            mask = condition(df[col])
            violations[col] = df.index[mask].tolist()

    # Datetime check
    if 'tpep_pickup_datetime' in df.columns and 'tpep_dropoff_datetime' in df.columns:
        mask = df['tpep_dropoff_datetime'] < df['tpep_pickup_datetime']
        violations['pickup_before_dropoff'] = df.index[mask].tolist()

    # Combine all violating indices
    all_bad_indices = set()
    for v in violations.values():
        all_bad_indices.update(v)

    return violations, all_bad_indices

# ##

# clean the dataset for real now

# Get the violation df and the row numbers of offending data entries
violations, bad_idx = get_constraint_violations(rides)
rides_clean = rides.drop(index=bad_idx)

# Print a summary in order to check how much is flagged. We want this to be relatively small compared to sample size.
pad = max(len(v) for v in violations.keys())
print(f"{"violation":<{pad + 1}}count")
for k, v in violations.items():
    print(f"{k:<{pad + 1}}{len(v)}")
print(f"Total bad rows: {len(bad_idx)}")
print(f"Ratio bad rows: {len(bad_idx) / len(rides)}")

###

# plot with padding to see the edges of what's being removed
# pad multiplies upper bound of gamma-distributed variables by 4

_, bad_idx_pad = get_constraint_violations(rides, pad=4)
print(f"Total bad rows: {len(bad_idx_pad)}")
print(f"Ratio bad rows: {len(bad_idx_pad) / len(rides)}")
rides_pad = rides.drop(index=bad_idx_pad)

for i, ax in nplots(len(raw_numeric), 4):
    feature = raw_numeric[i]
    ax.hist(
        [
            rides_pad[feature].dropna(),
            rides_clean[feature].dropna(),
        ],
        label=["upper bound x4", "upper bound"],
        bins=50,
        log=True,
    )
    ax.set_title(feature)
    ax.set_xlabel(feature)
    ax.set_ylabel("count")

###md

# The above graph looks extreme, but it is in log scale. In a linear scale
# it looks more appropriate. More dat is often better, but these variables
# have very long tails which can throw off mechanisms like scaling and makes
# learning correlations difficult.
#
# Future work would be to try to non-linearly compress the outlying values
# instead of throwing them out entirely.
#
# All things considered, the discarded data is <3% of the training data

###


for i, ax in nplots(len(raw_numeric), 4):
    feature = raw_numeric[i]
    ax.hist(rides_clean[feature].dropna(), bins=50)
    ax.set_title(feature)
    ax.set_xlabel(feature)
    ax.set_ylabel("count")

###

rides_clean.dtypes
rides_clean.shape

###

def save_split(df, prefix, rows_per_file=1800000):
    for i in range(0, ceildiv(len(df), rows_per_file)):
        df[i*rows_per_file:(i+1)*rows_per_file].to_csv(dbg(f"{prefix}_{i:02}.csv"), index=False)

save_split(
    rides_clean.sort_values("tpep_pickup_datetime"),
    f"../dataset-clean/{DATA_SUBSET}",
)

###

