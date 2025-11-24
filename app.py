import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Title
st.title("Unified ML Platform 🚀")

# Sidebar for dataset upload
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Handle missing values (NaN) column-wise
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])  # fill with most frequent
        else:
            df[col] = df[col].fillna(df[col].mean())     # fill with mean

    # Feature selection
    target = st.sidebar.selectbox("Select Target Column", df.columns)

    # Checkbox for selecting all features at once
    select_all = st.sidebar.checkbox("Select All Features")

    if select_all:
        features = [col for col in df.columns if col != target]
    else:
        features = st.sidebar.multiselect(
            "Select Feature Columns",
            [col for col in df.columns if col != target]
        )

    if features:
        X = df[features].copy()
        y = df[target].copy()

        # Encode categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Detect problem type
        if y.dtype == 'object' or len(y.unique()) <= 20:
            problem_type = "classification"
            y = LabelEncoder().fit_transform(y.astype(str))
        else:
            problem_type = "regression"

        # Train-test split (only for supervised)
        test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Algorithm selection
        algo_type = st.sidebar.radio("Select Task Type", 
                                     ["Classification", "Regression", "Clustering"])

        if algo_type == "Classification":
            algo = st.sidebar.selectbox(
                "Choose Algorithm",
                ["Logistic Regression", "Decision Tree Classifier",
                 "Random Forest Classifier", "SVM", "KNN Classifier"]
            )
            if algo == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif algo == "Decision Tree Classifier":
                model = DecisionTreeClassifier()
            elif algo == "Random Forest Classifier":
                model = RandomForestClassifier()
            elif algo == "SVM":
                model = SVC()
            elif algo == "KNN Classifier":
                k = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)
                model = KNeighborsClassifier(n_neighbors=k)

            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Results
            st.write("### Accuracy:", accuracy_score(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            # Actual vs Predicted table
            comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            st.write("### Actual vs Predicted Values")
            st.write(comparison_df.head(20))

        elif algo_type == "Regression":
            algo = st.sidebar.selectbox(
                "Choose Algorithm",
                ["Linear Regression", "Decision Tree Regressor",
                 "Random Forest Regressor", "SVR", "KNN Regressor"]
            )
            if algo == "Linear Regression":
                model = LinearRegression()
            elif algo == "Decision Tree Regressor":
                model = DecisionTreeRegressor()
            elif algo == "Random Forest Regressor":
                model = RandomForestRegressor()
            elif algo == "SVR":
                model = SVR()
            elif algo == "KNN Regressor":
                k = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)
                model = KNeighborsRegressor(n_neighbors=k)

            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Results
            st.write("### R² Score:", r2_score(y_test, y_pred))
            st.write("### Mean Squared Error:", mean_squared_error(y_test, y_pred))
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.7)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

            # Actual vs Predicted table
            comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
            st.write("### Actual vs Predicted Values")
            st.write(comparison_df.head(20))

        else:  # Clustering
            algo = st.sidebar.selectbox(
                "Choose Clustering Algorithm",
                ["KMeans", "DBSCAN", "Agglomerative Clustering"]
            )
            if algo == "KMeans":
                k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
                model = KMeans(n_clusters=k, random_state=42)
            elif algo == "DBSCAN":
                eps = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
                min_samples = st.sidebar.slider("Min Samples", 1, 10, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
            elif algo == "Agglomerative Clustering":
                k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
                model = AgglomerativeClustering(n_clusters=k)

            # Fit clustering model
            y_pred = model.fit_predict(X)

            # Results
            st.write("### Cluster Assignments")
            st.write(pd.DataFrame({"Cluster": y_pred}).head(20))

            # Scatter plot (first 2 features)
            if X.shape[1] >= 2:
                fig, ax = plt.subplots()
                scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, cmap="viridis", alpha=0.7)
                ax.set_xlabel(X.columns[0])
                ax.set_ylabel(X.columns[1])
                ax.set_title("Clustering Result (first 2 features)")
                plt.colorbar(scatter, ax=ax)
                st.pyplot(fig)
