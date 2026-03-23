
# TODO: pull cleaning code from Toby and Hani

import matplotlib
import matplotlib.pyplot as plt

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

rides = manhat_rides[
    manhat_rides["tpep_pickup_datetime"].map(lambda x: x.year == 2024)
].assign(
    pickup_hour=manhat_rides["tpep_pickup_datetime"].dt.floor("1h"),
    trip_duration_min=(manhat_rides["tpep_dropoff_datetime"] - manhat_rides["tpep_pickup_datetime"]).dt.seconds / 60,
)

###

# rides_sorted = rides.sort_values(["tpep_pickup_datetime", "PULocationID"]).reset_index(drop=True)

KEYS = ["PULocationID", "pickup_hour"]

def norm_time(
    df,
    aggs,
):
    return df.groupby(KEYS).agg(aggs).reset_index()

rides_ts = norm_time(
    rides,
    dict(
        trip_duration_min="sum",
        trip_distance="sum",
        total_amount="sum",
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
            suffixes=("", f"-{window}-{offset}"),
        )

    return df

with_lags(
    rides_ts,
    "trip_duration_min",
    "mean",
)

###

rides.groupby(["pickup_hour"])["fare_amount"].sum().plot.line()
plot()

