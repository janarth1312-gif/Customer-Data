# ==========================================
# Customer Segmentation using K-Means
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ------------------------------------------
# Step 1: Generate Sample Customer Data
# ------------------------------------------

np.random.seed(42)

num_customers = 500

data = {
    "CustomerID": range(1, num_customers + 1),
    "Age": np.random.randint(18, 70, num_customers),
    "AnnualIncome": np.random.randint(20000, 150000, num_customers),
    "SpendingScore": np.random.randint(1, 100, num_customers),
    "PurchaseFrequency": np.random.randint(1, 30, num_customers),
    "OnlinePurchases": np.random.randint(0, 20, num_customers),
    "StorePurchases": np.random.randint(0, 15, num_customers)
}

df = pd.DataFrame(data)

# Save generated data
df.to_csv("Customer_Data.csv", index=False)

print("First 5 Records")
print(df.head())

# ------------------------------------------
# Step 2: Select Features
# ------------------------------------------

X = df[['Age',
        'AnnualIncome',
        'SpendingScore',
        'PurchaseFrequency',
        'OnlinePurchases',
        'StorePurchases']]

# ------------------------------------------
# Step 3: Standardize Data
# ------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------
# Step 4: Elbow Method
# ------------------------------------------

wcss = []

for i in range(1, 11):
    model = KMeans(n_clusters=i, random_state=42, n_init=10)
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# ------------------------------------------
# Step 5: K-Means Clustering
# ------------------------------------------

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# ------------------------------------------
# Step 6: Cluster Summary
# ------------------------------------------

print("\nCluster Summary\n")

summary = df.groupby("Cluster").mean(numeric_only=True)

print(summary)

# ------------------------------------------
# Step 7: Visualization
# ------------------------------------------

plt.figure(figsize=(8,6))

sns.scatterplot(
    x="AnnualIncome",
    y="SpendingScore",
    hue="Cluster",
    palette="Set1",
    data=df,
    s=80
)

plt.title("Customer Segmentation")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

# ------------------------------------------
# Step 8: Pair Plot
# ------------------------------------------

sns.pairplot(
    df,
    vars=[
        "AnnualIncome",
        "SpendingScore",
        "Age",
        "PurchaseFrequency"
    ],
    hue="Cluster"
)

plt.show()

# ------------------------------------------
# Step 9: Save Output
# ------------------------------------------

df.to_csv("Customer_Segments.csv", index=False)

print("\nFiles Created Successfully!")
print("1. Customer_Data.csv")
print("2. Customer_Segments.csv")

# ------------------------------------------
# Step 10: Cluster Counts
# ------------------------------------------

print("\nCustomers in Each Cluster\n")
print(df["Cluster"].value_counts())
