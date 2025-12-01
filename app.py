import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
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

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    # Feature selection
    target = st.sidebar.selectbox("Select Target Column", df.columns)
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

        # Train-test split
        test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Algorithm selection
        algo_type = st.sidebar.radio("Select Task Type", ["Classification", "Regression", "Clustering"])

        # Select algorithm and hyperparameters
        if algo_type == "Classification":
            algo = st.sidebar.selectbox(
                "Choose Algorithm",
                ["Logistic Regression", "Decision Tree Classifier",
                 "Random Forest Classifier", "SVM", "KNN Classifier"]
            )
            hyperparams = {}
            if algo == "Decision Tree Classifier":
                hyperparams['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
            elif algo == "Random Forest Classifier":
                hyperparams['n_estimators'] = st.sidebar.slider("Number of Trees", 10, 200, 100)
                hyperparams['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
            elif algo == "SVM":
                hyperparams['C'] = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, 1.0)
            elif algo == "KNN Classifier":
                hyperparams['n_neighbors'] = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)

        elif algo_type == "Regression":
            algo = st.sidebar.selectbox(
                "Choose Algorithm",
                ["Linear Regression", "Decision Tree Regressor",
                 "Random Forest Regressor", "SVR", "KNN Regressor"]
            )
            hyperparams = {}
            if algo == "Decision Tree Regressor":
                hyperparams['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
            elif algo == "Random Forest Regressor":
                hyperparams['n_estimators'] = st.sidebar.slider("Number of Trees", 10, 200, 100)
                hyperparams['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
            elif algo == "SVR":
                hyperparams['C'] = st.sidebar.number_input("C (Regularization)", 0.01, 10.0, 1.0)
            elif algo == "KNN Regressor":
                hyperparams['n_neighbors'] = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)

        else:  # Clustering
            algo = st.sidebar.selectbox(
                "Choose Clustering Algorithm",
                ["KMeans", "DBSCAN", "Agglomerative Clustering"]
            )
            hyperparams = {}
            if algo == "KMeans":
                hyperparams['n_clusters'] = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)
            elif algo == "DBSCAN":
                hyperparams['eps'] = st.sidebar.slider("Epsilon (eps)", 0.1, 5.0, 0.5)
                hyperparams['min_samples'] = st.sidebar.slider("Min Samples", 1, 10, 5)
            elif algo == "Agglomerative Clustering":
                hyperparams['n_clusters'] = st.sidebar.slider("Number of Clusters (k)", 2, 10, 3)

        # Submit button to train model
        if st.sidebar.button("Train Model"):

            # Initialize model with hyperparameters
            if algo_type == "Classification":
                if algo == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif algo == "Decision Tree Classifier":
                    model = DecisionTreeClassifier(max_depth=hyperparams.get('max_depth'))
                elif algo == "Random Forest Classifier":
                    model = RandomForestClassifier(
                        n_estimators=hyperparams.get('n_estimators'),
                        max_depth=hyperparams.get('max_depth')
                    )
                elif algo == "SVM":
                    model = SVC(C=hyperparams.get('C'))
                elif algo == "KNN Classifier":
                    model = KNeighborsClassifier(n_neighbors=hyperparams.get('n_neighbors'))

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.write("### Accuracy:", accuracy_score(y_test, y_pred))
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                st.write("### Actual vs Predicted Values")
                comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                st.write(comparison_df.head(20))

                # Show Decision Tree Image
                if algo in ["Decision Tree Classifier"]:
                    fig, ax = plt.subplots(figsize=(12,6))
                    plot_tree(model, feature_names=features, class_names=True, filled=True, ax=ax)
                    st.pyplot(fig)

            elif algo_type == "Regression":
                if algo == "Linear Regression":
                    model = LinearRegression()
                elif algo == "Decision Tree Regressor":
                    model = DecisionTreeRegressor(max_depth=hyperparams.get('max_depth'))
                elif algo == "Random Forest Regressor":
                    model = RandomForestRegressor(
                        n_estimators=hyperparams.get('n_estimators'),
                        max_depth=hyperparams.get('max_depth')
                    )
                elif algo == "SVR":
                    model = SVR(C=hyperparams.get('C'))
                elif algo == "KNN Regressor":
                    model = KNeighborsRegressor(n_neighbors=hyperparams.get('n_neighbors'))

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.write("### R² Score:", r2_score(y_test, y_pred))
                st.write("### Mean Squared Error:", mean_squared_error(y_test, y_pred))
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, alpha=0.7)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title("Actual vs Predicted")
                st.pyplot(fig)

                st.write("### Actual vs Predicted Values")
                comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                st.write(comparison_df.head(20))

                # Show Regression Equation
                if algo == "Linear Regression":
                    equation = " + ".join([f"{coef:.3f}*{feat}" for coef, feat in zip(model.coef_, features)])
                    st.write(f"### Regression Equation:\n y = {model.intercept_:.3f} + {equation}")

            else:  # Clustering
                if algo == "KMeans":
                    model = KMeans(n_clusters=hyperparams.get('n_clusters'), random_state=42)
                elif algo == "DBSCAN":
                    model = DBSCAN(eps=hyperparams.get('eps'), min_samples=hyperparams.get('min_samples'))
                elif algo == "Agglomerative Clustering":
                    model = AgglomerativeClustering(n_clusters=hyperparams.get('n_clusters'))

                y_pred = model.fit_predict(X)
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
