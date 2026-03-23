import pandas as pd
import sklearn as sk
import numpy as np

###

rides = pd.read_csv(
    "../dataset/rides_train_sample.csv",
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

###

# Find Manhattan zone IDs for filtering up next
zones = pd.read_csv('../dataset/taxi_zone_lookup.csv')
manhat_zones = zones[zones['Borough'] == 'Manhattan']['LocationID']

###

# rides.shape
# rides.info()
rides.describe()
#rides.head()

# zones.info()
#manhat_zones.head()

###

manhat_rides = rides[rides['PULocationID'].isin(manhat_zones)].copy()

###

# manhat_rides.shape
# manhat_rides.info()
manhat_rides.describe()

###

print(manhat_rides.isna().mean() * 100)

###

missing_by_vendor = (
    manhat_rides
    .isna()
    .groupby(manhat_rides['VendorID'])
    .sum() * 100 / len(manhat_rides)
).round(5)

missing_by_vendor = missing_by_vendor.loc[:, (missing_by_vendor > 0).any()]

print(missing_by_vendor)

###

# Convert to datetime
manhat_rides['tpep_pickup_datetime'] = pd.to_datetime(
    manhat_rides['tpep_pickup_datetime'],
    errors='coerce'
)

manhat_rides['tpep_dropoff_datetime'] = pd.to_datetime(
    manhat_rides['tpep_dropoff_datetime'],
    errors='coerce'
)

# Convert Y/N to boolean
manhat_rides['store_and_fwd_flag'] = (
    manhat_rides['store_and_fwd_flag']
    .map({'Y': True, 'N': False})
    .astype('boolean') # just for reference clarity
)

###

manhat_rides.dtypes

###

# Duration in hours
manhat_rides['duration_hours'] = (
    (manhat_rides['tpep_dropoff_datetime'] - manhat_rides['tpep_pickup_datetime'])
    .dt.total_seconds() / 3600
)

# Average speed (mph)
manhat_rides['avg_speed_mph'] = (
    manhat_rides['trip_distance'] /
    manhat_rides['duration_hours'].replace(0, np.nan)
)

###

def get_constraint_violations(df):

    violations = {}
    # These are the cut-offs we apply to the data. They are decided and refined using statistical assessment in 3.2
    # The conditions describe what data will be flagged
    possible_checks = {
        'passenger_count': lambda x: x < 0,
        'trip_distance': lambda x: (x <= 0) | (x > 300), # to ensure the taxi actually moved
        'fare_amount': lambda x: (x < 0) | (x > 3000),
        'extra': lambda x: x < 0,
        'mta_tax': lambda x: x < 0,
        'tip_amount': lambda x: x < 0,
        'tolls_amount': lambda x: x < 0,
        'improvement_surcharge': lambda x: x < 0,
        'total_amount': lambda x: x < 0,
        'congestion_surcharge': lambda x: x < 0,
        'Airport_fee': lambda x: x < 0,
        'duration_hours': lambda x: (x <= 0) | (x > 12), # to remove any entries which took zero time
        'avg_speed_mph': lambda x: (x < 0.1) | (x > 150) # to remove any entries which suggest the taxi has moved unrealistically far in a given time.
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

###

# Get the violation df and the row numbers of offending data entries
violations, bad_idx = get_constraint_violations(manhat_rides)

# Print a summary in order to check how much is flagged. We want this to be relatively small compared to sample size.
print({k: len(v) for k, v in violations.items()})
print(f"Total bad rows: {len(bad_idx)}")

###

#Inspect the bad rows in case of suspicion
manhat_rides.loc[list(bad_idx)].head()

###

# Remove the data rows that were identified as having constraint issues
manhat_rides_clean = manhat_rides.drop(index=bad_idx).copy()

#Quick check
print(manhat_rides.shape)
print(manhat_rides_clean.shape)

###

# Have a look at statistics
manhat_rides_clean.describe().loc[['mean', 'std', 'max', 'min']]

###

# Check toll amounts manually as they seem a bit high. All checks out.
print(manhat_rides_clean[manhat_rides_clean['tolls_amount'] > 50]['tolls_amount'])

###

import numpy as np
import pandas as pd

# Numerical features to be tested (judged using above describe() print and particularly maxima
# All maxima are satisfactory. 'tolls_amount' needed extra analysis, but all seems in order. We now Z-test the remaining features.
# Note fare_amount <= total_amount
iqrtest_features = [
    'trip_distance',
    'total_amount',
]

def get_iqr_violations(df, features, k=1.5):

    # Store flagged row indices separately for each feature
    violations = {}
    # Store flagged row indices separately for each feature
    flagged_rows = []
    # Store all unique bad row indices across all tested features
    bad_idx = set()

    # Loop through each feature that's selected
    for col in features:
        # Skip if not in the dataframe
        if col not in df.columns:
            continue

        # If the column has no non-missing values, record no violations
        x = df[col].dropna()
        if x.empty:
            violations[col] = []
            continue

        # Quantiles
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        # Interquantile range (IQR)
        iqr = q3 - q1

       # Thresholds
        lower = q1 - k * iqr
        upper = q3 + k * iqr

        # Flag rows lying outside acceptable range
        mask = (df[col] < lower) | (df[col] > upper)
        idx_list = df.index[mask.fillna(False)].tolist()

        # Store violating row numbers
        violations[col] = idx_list
        # Add bad row numbers to the set
        bad_idx.update(idx_list)

        # Store some info on each flagged row (for review)
        for idx in idx_list:
            flagged_rows.append({
                'row_index': idx,
                'feature': col,
                'value': df.at[idx, col]
            })

    flagged_df = pd.DataFrame(flagged_rows)

    return violations, flagged_df, bad_idx

###

violations_iqr, flagged_iqr_df, bad_idx_iqr = get_iqr_violations(
    manhat_rides_clean,
    features=iqrtest_features,
    k=5 # adjust for sensitivity
)

###

flagged_iqr_df.sort_values(by='value')

###

# Have a look at the number of flagged entries in each selected feature.
print({k: len(v) for k, v in violations_iqr.items()})
print(f"Total unique bad rows: {len(bad_idx_iqr)}")

###

# Inspect original rows (very many)
manhat_rides_clean[['duration_hours', 'avg_speed_mph', 'trip_distance', 'total_amount', 'fare_amount', 'PULocationID', 'DOLocationID']].loc[list(bad_idx_iqr)].sort_values(by='total_amount', ascending=False)

###

