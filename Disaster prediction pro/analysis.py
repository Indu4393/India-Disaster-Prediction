import pandas as pd
import numpy as np
import json

# ─── 1. FLOOD / DISASTER DATA ───────────────────────────
floods = pd.read_csv(r'C:\Users\kuruv\OneDrive\Documents\Disaster Prediction datasets\flood_risk_dataset_india.csv')  # rename your file to this
print("=" * 50)
print("FLOOD DATASET")
print("Shape:", floods.shape)
print(floods.head())
print(floods.dtypes)
print("Nulls:\n", floods.isnull().sum())

# ─── 2. RAINFALL DATA ───────────────────────────────────
rainfall = pd.read_csv(r"C:\Users\kuruv\OneDrive\Documents\Disaster Prediction datasets\rainfall in india 1901-2015.csv")  # rename your file to this
print("=" * 50)
print("RAINFALL DATASET")
print("Shape:", rainfall.shape)
print(rainfall.head())
print(rainfall.dtypes)
print("Nulls:\n", rainfall.isnull().sum())

# ─── 3. EARTHQUAKE DATA ─────────────────────────────────
earthquakes = pd.read_csv(r'C:\Users\kuruv\OneDrive\Documents\Disaster Prediction datasets\Indian_earthquake_data.csv')  # rename your file to this
print("=" * 50)
print("EARTHQUAKE DATASET")
print("Shape:", earthquakes.shape)
print(earthquakes.head())
print(earthquakes.dtypes)
print("Nulls:\n", earthquakes.isnull().sum())

# ─── 4. GEOJSON MAP ─────────────────────────────────────
with open(r'C:\Users\kuruv\OneDrive\Documents\Disaster Prediction datasets\INDIA_DISTRICTS.geojson.txt') as f:
    geo = json.load(f)

print("=" * 50)
print("GEOJSON MAP")
print("Type:", geo['type'])
print("Total districts:", len(geo['features']))
print("Sample district:", geo['features'][0]['properties']['district'])
print("Sample state:", geo['features'][0]['properties']['state'])


# Check 1 — class balance
print(floods['Flood Occurred'].value_counts(normalize=True) * 100)

# Check 2 — rainfall nulls after interpolation  
print("Nulls remaining:", rainfall.isnull().sum().sum())
# Interpolation fails if a subdivision has nulls at START or END
# (nothing to interpolate from). Fill those remaining with median.

rainfall = rainfall.sort_values(['SUBDIVISION','YEAR'])

# Step 1: interpolate within group
rainfall[['JAN','FEB','MAR','APR','MAY','JUN',
          'JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']] = \
    rainfall.groupby('SUBDIVISION')[
        ['JAN','FEB','MAR','APR','MAY','JUN',
         'JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL']
    ].transform(lambda x: x.interpolate(method='linear'))


# Step 2: fill any remaining with column median — pandas 2.0 safe
for col in ['JAN','FEB','MAR','APR','MAY','JUN',
            'JUL','AUG','SEP','OCT','NOV','DEC','ANNUAL',
            'Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']:
    rainfall[col] = rainfall[col].fillna(rainfall[col].median())

# Verify
print("Nulls remaining:", rainfall.isnull().sum().sum())  # Should be 0
# Check 3 — earthquake datetime fixed
print(earthquakes['Origin Time'].dtype)
earthquakes['Origin Time'] = earthquakes['Origin Time'].str.replace(' IST', '', regex=False)
earthquakes['Origin Time'] = pd.to_datetime(earthquakes['Origin Time'], format='%Y-%m-%d %H:%M:%S')

print(earthquakes['Origin Time'].dtype)  # Should print: datetime64[ns]
print(earthquakes['Origin Time'].head(3))
print("\n All 4 datasets loaded successfully!")

import os

# Create processed folder
os.makedirs('data/processed', exist_ok=True)

# Save all 3 cleaned datasets
floods.to_csv('data/processed/floods_clean.csv', index=False)
rainfall.to_csv('data/processed/rainfall_clean.csv', index=False)
earthquakes.to_csv('data/processed/earthquakes_clean.csv', index=False)

print(" floods_clean.csv saved —", floods.shape)
print(" rainfall_clean.csv saved —", rainfall.shape)
print(" earthquakes_clean.csv saved —", earthquakes.shape)
print(" GeoJSON already saved in data/raw/")
