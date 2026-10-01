import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


def execute_pipeline():
    print("[MLOps Pipeline] Starting pipeline execution...")

    # 1. Charger le dataset Iris sous forme de tableau pandas
    raw_data = load_iris(as_frame=True)
    df = raw_data.frame

    # 2. Séparer les colonnes : X = mesures, y = espèce (dernière colonne)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 3. Couper en 80 % entraînement / 20 % test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Créer et entraîner le modèle Random Forest (100 arbres)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    # 5. Créer le dossier models/ s'il n'existe pas, puis sauvegarder le modèle
    os.makedirs('models', exist_ok=True)
    with open('models/iris_model.pkl', 'wb') as f:
        pickle.dump(classifier, f)

    # 6. Afficher le score sur les données de test
    print(f"[MLOps Pipeline] Target model successfully cached! Score: {classifier.score(X_test, y_test):.4f}")


# Ce bloc lance la fonction seulement si on exécute ce fichier directement
if __name__ == '__main__':
    execute_pipeline()