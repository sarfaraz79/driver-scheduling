import datetime
from glob import glob

import pandas as pd
import sklearn as sk

rides = pd.concat([
    pd.read_csv(
        f,
        dtype=dict(
            VendorID=int,
            tpep_pickup_datetime=str,
            tpep_dropoff_datetime=str,
            passenger_count=float,
            trip_distance=float,
            RatecodeID=float,
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
    )
    for f in glob("data/*.csv")
])

rides = (
    rides
    .assign(
        tpep_pickup_datetime=rides[["tpep_pickup_datetime"]].map(
            datetime.datetime.fromisoformat,
        ),
        tpep_dropoff_datetime=rides[["tpep_dropoff_datetime"]].map(
            datetime.datetime.fromisoformat,
        ),
    )
)

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

rides.iloc[train_idx].to_csv("rides_train.csv")
rides.iloc[test_idx].to_csv("rides_test.csv")

