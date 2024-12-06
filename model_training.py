import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Fonction de prétraitement des données
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
    
    
    # Afficher les valeurs uniques de chaque colonne
    for column in data.columns:
        print(f"Valeurs uniques pour la colonne '{column}':")
        print(data[column].unique())
    print("\n")

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

# Charger et préparer les données
X_train, X_test, y_train, y_test, encoder, label_encoder = preprocess_data()

# Convertir y_train et y_test en one-hot
y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

# Construire le modèle
model = Sequential()
model.add(Dense(64, input_dim=18, activation='relu'))  # Nous spécifions que l'entrée est un vecteur de 18 caractéristiques
model.add(Dropout(0.4))  #eviter le surapprentissage
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(y_train_onehot.shape[1], activation='softmax'))  # Le nombre de classes pour la sortie


# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping pour eviter le surapprentissage
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)

# Entraîner le modèle
history = model.fit(X_train, y_train_onehot, validation_data=(X_test, y_test_onehot),
                    epochs=30, batch_size=64, callbacks=[early_stopping, reduce_lr])

# Sauvegarder le modèle
model.save("model.h5")  # Sauvegarde du modèle
print("Le modèle a été sauvegardé avec succès.")
import joblib
joblib.dump(encoder, 'encoder.pkl')


# Optionnel: Évaluation du modèle
test_loss, test_acc = model.evaluate(X_test, y_test_onehot)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
