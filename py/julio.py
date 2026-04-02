# TODO: pull cleaning code from Toby and Hani


import glob

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

###

if get_ipython().__class__.__name__ == "TerminalInteractiveShell":
    matplotlib.use("Agg")

def plot(figsize=(6.4, 4.8), **kwargs):
    plt.gcf().set_size_inches(*figsize)
    if get_ipython().__class__.__name__ == "TerminalInteractiveShell":
        plt.savefig("a.png", **kwargs)
        plt.clf()
        plt.cla()
    else:
        plt.show()

def dbg(x):
    print(x)
    return x

###

def load_clean(pat):
    return pd.concat([
        pd.read_csv(
            name,
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
        for name in sorted(glob.glob(f"../dataset-clean/{pat}"))
    ])

rides_clean = load_clean("rides_sample.csv").assign(
    pickup_hour=lambda x: x["tpep_pickup_datetime"].dt.floor("1h"),
    pickup_time_minute=lambda x:
        x["tpep_pickup_datetime"].dt.hour * 60 +
        x["tpep_pickup_datetime"].dt.minute,
    fare_per_hour=lambda x:
        x["fare_amount"] / x["duration_hours"],
)

###

def group_zone_hour(
    df,
    aggs,
):
    """
    Aggregrate trips into zone-hour statistics

    df : pd.DataFrame
        Trip-level dataframe
    aggs : function, str, list, dict or None
        Argument to pd.DataFrameGroupBy.agg, specify which columsn and
        how aggregation should be done

    Returns:
    pd.DataFrame
        Dataframe with zone-hour and aggregated values
    """

    df = (
        df
            .groupby(["PULocationID", "pickup_hour"], observed=True)
            .agg(aggs)
            .reset_index()
    )
    df.columns = [(f"{x}_{y}" if y else x) for (x, y) in df.columns]

    return df

rides_agg = group_zone_hour(
    rides_clean,
    dict(
        duration_hours=["sum", "count"],
        trip_distance="sum",
        fare_amount="sum",
        total_amount="sum",
        fare_per_hour="mean",
    ),
)

###

def with_lags(
    df,
    feature,
    agg,
    window="2h",
    offsets=["1h"],
):
    agged = df.groupby("PULocationID")[["pickup_hour", feature]].rolling(
        window,
        on="pickup_hour",
    ).agg(agg)

    for offset in offsets:
        df = pd.merge(
            df,
            agged.assign(
                pickup_hour=agged["pickup_hour"] +
                    (offset if not isinstance(offset, str) else pd.Timedelta(offset)),
            ),
            how="left",
            on=["PULocationID", "pickup_hour"],
            suffixes=("", f"-{agg}-{window}-{offset}"),
        )

    return df

rides_lag = rides_agg

for feat in ["duration_hours_sum", "fare_amount_sum", "total_amount_sum", "fare_per_hour_mean"]:
    for (window, offsets) in [
        ("1h", ["1h", "2h", "3h"]),
        ("24h", ["1d", "2d", "3d", "7d"]),
    ]:
        rides_lag = with_lags(
            rides_lag,
            feat,
            "mean",
            window=window,
            offsets=offsets,
        )

rides_lag

###

rides["amount_per_min"].hist()
plot()


###

rides[rides["amount_per_min"] > 100][
    [
        "PULocationID",
        "DOLocationID",
        "fare_amount",
        "trip_duration_min",
        "avg_speed_mph",
        "amount_per_min",
    ]
]

###

rides[rides["PULocationID"].astype(str) == rides["DOLocationID"].astype(str)][
    [
        "RatecodeID",
        "PULocationID",
        "DOLocationID",
        "fare_amount",
        "trip_duration_min",
        "avg_speed_mph",
        "amount_per_min",
    ]
]

###

x = rides[
    (rides["RatecodeID"] == 1) &
    (rides["PULocationID"].astype(str) != rides["DOLocationID"].astype(str)) &
    (rides["trip_duration_min"] > 1)
][
    [
        "RatecodeID",
        "PULocationID",
        "DOLocationID",
        "fare_amount",
        "trip_duration_min",
        "avg_speed_mph",
        "amount_per_min",
    ]
]
x["amount_per_min"].hist(bins=20)
plot()

###

rides[rides["PULocationID"].astype(str) == rides["DOLocationID"].astype(str)][
    "amount_per_min"
].sample(100)

###

plot()

###

# rides_sorted = rides.sort_values(["tpep_pickup_datetime", "PULocationID"]).reset_index(drop=True)

KEYS = ["PULocationID", "pickup_hour"]

def norm_time(
    df,
    aggs,
):
    return df.groupby(KEYS).agg(aggs).reset_index()

rides_ts = norm_time(
    rides.assign(
        fare_per_min=lambda x:
            x["total_amount"] / x["trip_duration_min"],
    ),
    dict(
        trip_duration_min="sum",
        trip_distance="sum",
        total_amount="sum",
        fare_per_min="mean",
    ),
)

###

def with_lags(
    df,
    feature,
    agg,
    window="2h",
    offsets=["1h"],
):
    agged = df.groupby("PULocationID")[["pickup_hour", feature]].rolling(
        window,
        on="pickup_hour",
    ).agg(agg)

    for offset in offsets:
        df = pd.merge(
            df,
            agged.assign(
                pickup_hour=agged["pickup_hour"] +
                    (offset if not isinstance(offset, str) else pd.Timedelta(offset)),
            ),
            on=KEYS,
            suffixes=("", f"-{agg}-{window}-{offset}"),
        )

    return df


###

with_lags(
    rides_ts,
    "total_amount",
    "mean",
).plot(
    kind="scatter",
    x="pickup_hour",
    y="total_amount-2h-1h",
    c="total_amount",
    s=0.5,
    #alpha=0.3,
    cmap="viridis",
)
plot(figsize=(10, 10))

### toby mean stdev corr

WINDOW = "1h"
OFFSET = "12h"

x = with_lags(
    rides_ts,
    "total_amount",
    "mean",
    window=WINDOW,
    offsets=[OFFSET],
)

x = with_lags(
    x,
    "trip_duration_min",
    "mean",
    window=WINDOW,
    offsets=[OFFSET],
)

x.plot(
    kind="scatter",
    x=f"total_amount-mean-{WINDOW}-{OFFSET}",
    y=f"trip_duration_min-mean-{WINDOW}-{OFFSET}",
    c="total_amount",
    s=0.5,
    alpha=0.3,
    cmap="viridis",
)
plot(figsize=(10, 10))

###

dbg(with_lags(
    rides_ts[
        rides_ts["fare_per_min"] < 20
    ],
    "fare_per_min",
    "mean",
    window="1h",
    offsets=["24h", "7d"],
)).plot(
    kind="scatter",
    x="fare_per_min-1h-24h",
    y="fare_per_min-1h-7d",
    c="fare_per_min",
    s=0.3,
    alpha=0.2,
    cmap="viridis",
    norm="log",
)
plot(figsize=(10, 10))

###

lag = with_lags(
    rides_ts.assign(
        fare_per_min=lambda x:
            x["total_amount"] / x["trip_duration_min"],
    ),
    "fare_per_min",
    "mean",
    offsets=["1h", "2h", "24h"],
)

lag_merge = pd.merge(
    rides,
    lag,
    on=["PULocationID", "pickup_hour"],
    suffixes=("", "_hour"),
)

lag_merge[
    (lag_merge["trip_duration_min"] < 200) &
    (lag_merge["fare_per_min-2h-1h"] < 10) &
    (lag_merge["fare_per_min-2h-24h"] < 10) &
    (lag_merge["trip_distance"] < 40)
].assign(
    fare_per_min=lambda x:
        x["total_amount"] / x["trip_duration_min"],
).plot(
    kind="scatter",
    x="fare_per_min-2h-1h",
    y="fare_per_min-2h-24h",
    c="fare_per_min",
    s=0.1,
    alpha=0.2,
    cmap="viridis",
    norm="log",
)
plot(figsize=(10, 10))

###

rides[(rides["trip_duration_min"] < 200) & (rides["fare_amount"] < 200) & (rides["trip_distance"] < 40)].plot(
    kind="scatter",
    x="trip_duration_min",
    y="trip_distance",
    c="fare_amount",
    s=0.5,
    alpha=0.3,
    cmap="viridis",
)
plot(figsize=(10, 10))

###

dbg(rides[(rides["trip_duration_min"] < 200) & (rides["fare_amount"] < 200)].groupby(
    "PULocationID",
).agg(dict(trip_distance="mean", trip_duration_min="mean"))).reset_index().plot(
    kind="bar",
    x="PULocationID",
    y=["trip_distance", "trip_duration_min"],
)
plot(figsize=(10, 10))

###

sns.catplot(
    data=pd.merge(
        rides[
            (rides["trip_duration_min"] < 200) & (rides["fare_amount"] < 200)
        ],
        rides.groupby("PULocationID").agg(count=pd.NamedAgg(column="PULocationID", aggfunc="count")).reset_index(),
    ),
    kind="boxen",
    y="PULocationID",
    x="trip_distance", # "trip_duration_min"],
    hue="count",
)
plot(figsize=(10, 10))

###

sns.catplot(
    data=pd.merge(
        rides[
            (rides["trip_duration_min"] < 200) & (rides["fare_amount"] < 200)
        ],
        rides.groupby("PULocationID").agg(count=pd.NamedAgg(column="PULocationID", aggfunc="count")).reset_index(),
    ),
    kind="boxen",
    y="PULocationID",
    x="trip_duration_min",
    hue="count",
)
plot(figsize=(10, 10))

###

dbg(
    rides[
        (rides["trip_duration_min"] < 100) & (rides["trip_distance"] < 40)
    ].assign(
        fare_per_min=lambda x: (x["total_amount"] / x["trip_duration_min"]).fillna(0),
    )
).plot(
    kind="scatter",
    x="trip_duration_min",
    y="trip_distance",
    c="fare_per_min",
    s=0.5,
    alpha=0.3,
    cmap="viridis",
    norm="log",
)

plot(figsize=(10, 10))

###

sns.heatmap(
    dbg(
        rides[
            (rides["trip_duration_min"] < 100) & (rides["trip_distance"] < 40)
        ].assign(
            fare_per_min=lambda x: (x["total_amount"] / x["trip_duration_min"]).fillna(0),
        ).groupby(["PULocationID", "DOLocationID"])["fare_per_min"].mean().reset_index()
    ).pivot(
        index="PULocationID",
        columns="DOLocationID",
        values="fare_per_min",
    ),
    norm="log",
    square=True,
)

plot(figsize=(10, 10))

###

dbg(rides.assign(
    fare_per_min=lambda x: (x["total_amount"] / x["trip_duration_min"]).fillna(0),
).groupby(["PULocationID"])["fare_per_min"].median().reset_index()).plot(
    kind="bar",
    x="PULocationID",
    y="fare_per_min",
)

plot()

###

dbg(rides.assign(
    fare_per_min=lambda x: (x["total_amount"] / x["trip_duration_min"]).fillna(0),
).groupby(["DOLocationID"])["fare_per_min"].median().reset_index()).plot(
    kind="bar",
    x="DOLocationID",
    y="fare_per_min",
)

plot(figsize=(15, 10))

###

sns.heatmap(
    dbg(rides.assign(
        hour_of_day=lambda x: x["pickup_hour"].dt.hour,
        fare_per_min=lambda x: (x["total_amount"] / x["trip_duration_min"]).fillna(0),
    ).groupby(["PULocationID", "DOLocationID"])["fare_per_min"].mean().reset_index()).pivot(
        index="PULocationID",
        columns="DOLocationID",
        values="fare_per_min",
    ),
)

plot()

###

fig, ax = plt.subplots(2)

for ax, loc in zip(ax, ["PULocationID", "DOLocationID"]):
    sns.heatmap(
        dbg(rides.assign(
            hour_of_day=lambda x: x["pickup_hour"].dt.hour,
            fare_per_min=lambda x: (x["total_amount"] / x["trip_duration_min"]).fillna(0),
        ).groupby([loc, "hour_of_day"])["fare_per_min"].mean().reset_index()).pivot(
            index=loc,
            columns="hour_of_day",
            values="fare_per_min",
        ),
        norm="log",
        ax=ax,
    )

plot(figsize=(10, 15))

###

x = with_lags(
    rides_ts.assign(
        hour_of_day=lambda x: x["pickup_hour"].dt.hour,
    ),
    "total_amount",
    "mean",
)
sns.heatmap(
    dbg(rides.assign(
        hour_of_day=lambda x: x["pickup_hour"].dt.hour,
        fare_per_min=lambda x: (x["total_amount"] / x["trip_duration_min"]).fillna(0),
    ).groupby(["PULocationID", "hour_of_day"])["fare_per_min"].mean().reset_index()).pivot(
        index="PULocationID",
        columns="hour_of_day",
        values="fare_per_min",
    ),
)

plot(figsize=(10, 10))

###

rides.groupby(["pickup_hour"])["fare_amount"].sum().plot.line()
plot()

