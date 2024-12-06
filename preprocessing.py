import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    # Charger et préparer les données
    data = pd.read_csv("only-data-set.csv")
    data.fillna(data.mean(numeric_only=True), inplace=True)
    data.dropna(inplace=True)

    # Calculer la distance entre origine et destination
    def calculate_distance(row):
        origin = (row["origin_lat"], row["origin_long"])
        destination = (row["destination_lat"], row["destination_long"])
        return geodesic(origin, destination).kilometers

    data["distance_km"] = data.apply(calculate_distance, axis=1)

    # Conversion des timestamps
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek

    # One-hot encoding
    categorical_features = ["transport_type", "weather_condition", "incident_type"]
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(data[categorical_features])

    # Normalisation
    scaler = MinMaxScaler()
    numerical_features = ["average_speed", "traffic_density", "temperature", "wind_speed", "distance_km", "hour"]
    normalized_features = scaler.fit_transform(data[numerical_features])

    # Préparer les features
    X = np.hstack([normalized_features, encoded_features])
    y = data["transport_type"]

    # Encodage des labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split en train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, encoder, label_encoder
