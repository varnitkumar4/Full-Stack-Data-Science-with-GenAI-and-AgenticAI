import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("Customer Segmentation using K-Means")

# Upload dataset
uploaded_file = st.file_uploader(r' ',type=["csv"])

if uploaded_file is not None:

    dataset = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(dataset.head())

    # Select features
    X = dataset.iloc[:, [3,4]].values

    # Elbow Method
    st.subheader("Elbow Method")

    wcss = []

    for i in range(1,11):
        kmeans = KMeans(n_clusters=i, init="k-means++", random_state=0, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(1,11), wcss, marker="o")
    ax1.set_title("Elbow Method")
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("WCSS")

    st.pyplot(fig1)

    # Select number of clusters
    k = st.slider("Select Number of Clusters",2,10,5)

    # Train KMeans
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=0, n_init=10)
    y_kmeans = kmeans.fit_predict(X)

    dataset["Cluster"] = y_kmeans

    st.subheader("Clustered Data")
    st.write(dataset.head())

    # Plot clusters
    fig2, ax2 = plt.subplots()

    colors = ["red","blue","green","cyan","magenta","yellow","black","orange","purple","brown"]

    for i in range(k):
        ax2.scatter(
            X[y_kmeans==i,0],
            X[y_kmeans==i,1],
            s=100,
            c=colors[i],
            label=f"Cluster {i+1}"
        )

    ax2.scatter(
        kmeans.cluster_centers_[:,0],
        kmeans.cluster_centers_[:,1],
        s=300,
        c="yellow",
        label="Centroids"
    )

    ax2.set_title("Customer Segments")
    ax2.set_xlabel("Annual Income")
    ax2.set_ylabel("Spending Score")

    ax2.legend()

    st.pyplot(fig2)
