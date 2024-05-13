import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class CustomerModel:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.model = None
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.fraud_propensity = {0: 0.05, 1: 0.10, 2: 0.15, 3: 0.20, 4: 0.01}

    def preprocess(self):
        numeric_cols = ['loan_amount', 'income', 'Credit_Score']
        X = self.data[numeric_cols]
        return self.pipeline.fit_transform(X)

    def train_model(self, n_clusters=5):
        X_preprocessed = self.preprocess()
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.model.fit(X_preprocessed)
        self.data['cluster'] = self.model.labels_

    def assign_labels(self):
        cluster_personas = {
            0: 'Low Risk', 
            1: 'Moderate Risk', 
            2: 'High Risk', 
            3: 'Very High Risk', 
            4: 'Premium Customer'
        }
        self.data['persona'] = self.data['cluster'].map(cluster_personas)

    def predict(self, new_data):
        input_df = pd.DataFrame([new_data])
        preprocessed_input = self.pipeline.transform(input_df)
        cluster_label = self.model.predict(preprocessed_input)
        persona = self.data['persona'][self.data['cluster'] == cluster_label[0]].iloc[0]
        fraud_propensity = self.fraud_propensity[cluster_label[0]]
        return int(cluster_label), persona, float(fraud_propensity) 