# train_svm.py
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import os

if __name__ == "__main__":
    data = np.load('dataset/bilstm_embeddings.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(probability=True))
    ])

    param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf'],
        'svc__gamma': ['scale', 'auto']
    }

    gs = GridSearchCV(pipe, param_grid, cv=3, scoring='f1_macro', verbose=2, n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f" Best Params: {gs.best_params_}")

    best_model = gs.best_estimator_
    preds = best_model.predict(X_val)

    print("\n Evaluation Report:\n", classification_report(y_val, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_val, preds))

    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/svm_pipeline.pkl')
    print(" SVM model saved to models/svm_pipeline.pkl")
