import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load your data
# Example columns: 'location_id', 'lat', 'lon', 'timestamp', 'event' ('rent' or 'return')
df = pd.read_csv("python_scripts/synthetic_bike_data.csv", parse_dates=["timestamp"])

# Step 1: Aggregate demand/supply per location
df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.dayofweek  # Monday=0, Sunday=6

agg = df.groupby(["location_id", "event", "hour"])["timestamp"].count().unstack(fill_value=0)
agg.columns = [f"{col}_hour" for col in agg.columns]
agg.reset_index(inplace=True)

# Merge location coordinates
locations = df.groupby("location_id")[["lat", "lon"]].first().reset_index()
agg = agg.merge(locations, on="location_id")

# Step 2: Feature engineering
# Combine features: usage patterns + coordinates
features = agg.drop(columns=["location_id"])
features = features.drop(columns=["event"])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Step 3: KMeans clustering
k = 5  # you can tune this
kmeans = KMeans(n_clusters=k, random_state=42)
agg["cluster"] = kmeans.fit_predict(X_scaled)

# Step 4: Visualize result
plt.figure(figsize=(10, 6))
for c in range(k):
    subset = agg[agg["cluster"] == c]
    plt.scatter(subset["lon"], subset["lat"], label=f"Cluster {c}", alpha=0.6)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Bike Location Clusters by Usage + Proximity")
plt.legend()
plt.grid(True)
plt.show()
