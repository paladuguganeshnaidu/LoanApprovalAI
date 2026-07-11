"""Model evaluation utilities."""
import joblib
from sklearn.metrics import classification_report

from train import X_test

from train import y_test
from sklearn.metrics import classification_report, confusion_matrix
    
model = joblib.load("saved_models/random_forest.pkl")

y_pred = model.predict( X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))