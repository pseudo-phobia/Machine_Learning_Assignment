

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score


# STEP 1: Create sample transportation data

data = {
    "transport_mode": [
        "MTC bus", "MTC bus", "MTC bus",
        "College bus", "College bus", "College bus",
        "Train + bus", "Train + bus", "Train + bus",
        "Two-wheeler", "Two-wheeler", "Two-wheeler",
        "Walk", "Walk", "Walk",
        "MTC bus", "College bus", "Two-wheeler",
        "Train + bus", "Walk", "MTC bus",
        "College bus", "Two-wheeler", "Train + bus", "Walk"
    ],

    "distance_km": [
        18, 22, 15, 25, 30, 20,
        35, 40, 32, 8, 12, 15,
        0.8, 1.2, 0.5, 12, 18, 9,
        28, 1.5, 30, 12, 18, 45, 2
    ],

    "travel_time_min": [
        55, 65, 48, 60, 70, 50,
        75, 85, 70, 25, 30, 35,
        10, 14, 7, 40, 45, 22,
        68, 18, 80, 38, 40, 95, 20
    ],

    "waiting_time_min": [
        12, 15, 10, 5, 7, 4,
        15, 18, 12, 0, 0, 0,
        0, 0, 0, 8, 4, 0,
        10, 0, 20, 3, 0, 20, 0
    ],

    "transfers_per_trip": [
        1, 2, 1, 0, 0, 0,
        1, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0,
        1, 0, 2, 0, 0, 2, 0
    ]
}

df = pd.DataFrame(data)


# STEP 2: Separate numerical and categorical data

numerical_features = [
    "distance_km",
    "travel_time_min",
    "waiting_time_min",
    "transfers_per_trip"
]

categorical_features = ["transport_mode"]


# STEP 3: Prepare the data for machine learning

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", StandardScaler(), numerical_features),
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)


# STEP 4: Create the K-Means model

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("kmeans", KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        ))
    ]
)


# STEP 5: Train the model

X = df[numerical_features + categorical_features]

# EXPERIMENT: Find the best number of clusters

X_processed = preprocessor.fit_transform(X)

inertias = []
silhouette_scores = []
cluster_values = range(2, 7)

for k in cluster_values:

    temporary_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = temporary_model.fit_predict(X_processed)

    inertias.append(temporary_model.inertia_)

    score = silhouette_score(X_processed, labels)
    silhouette_scores.append(score)

    print(
        f"Clusters: {k} | "
        f"Inertia: {temporary_model.inertia_:.2f} | "
        f"Silhouette: {score:.3f}"
    )


# Plot the Elbow Method

plt.figure(figsize=(8, 5))

plt.plot(
    list(cluster_values),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid()
plt.show()


# Plot Silhouette Scores

plt.figure(figsize=(8, 5))

plt.plot(
    list(cluster_values),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.grid()
plt.show()


model.fit(X)

df["cluster"] = model.predict(X)


# STEP 6: Display clustering results

print("\nSTUDENT CLUSTERING RESULTS")
print(df.to_string(index=False))


# STEP 7: Analyse the clusters

print("\nCLUSTER PROFILES")

summary = df.groupby("cluster").agg(
    student_count=("transport_mode", "size"),
    average_distance=("distance_km", "mean"),
    average_travel_time=("travel_time_min", "mean"),
    average_waiting_time=("waiting_time_min", "mean")
)

print(summary.round(2))


# STEP 8: Check clustering quality

X_processed = model.named_steps["preprocessor"].transform(X)

score = silhouette_score(X_processed, df["cluster"])

print("\nSilhouette Score:", round(score, 3))


# STEP 9: Visualise the clusters

plt.figure(figsize=(9, 6))

plt.scatter(
    df["distance_km"],
    df["travel_time_min"],
    c=df["cluster"],
    cmap="viridis",
    s=100
)

plt.xlabel("Travel Distance (km)")
plt.ylabel("Travel Time (minutes)")
plt.title("SRMIST Student Transportation Clustering")

plt.colorbar(label="Cluster")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()
print("Reached Step 10")

# STEP 10: Compare the actual student groups
# Test K = 5 and K = 6

for k in [5, 6]:

    print(f"\n===== ANALYSING {k} CLUSTERS =====")

    test_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = test_model.fit_predict(X_processed)

    # Create a copy of our original dataset
    analysis_df = df.copy()

    # Assign cluster numbers
    analysis_df["cluster"] = labels

    # Analyse each cluster
    cluster_summary = analysis_df.groupby("cluster").agg(
        student_count=("transport_mode", "size"),

        dominant_transport=(
            "transport_mode",
            lambda x: x.mode().iloc[0]
        ),

        average_distance=("distance_km", "mean"),

        average_travel_time=("travel_time_min", "mean"),

        average_waiting_time=("waiting_time_min", "mean")
    )

    print(cluster_summary.round(2))
