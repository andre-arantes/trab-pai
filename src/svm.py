from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import scipy
import csv
from sklearn.svm import SVC
from sklearn.calibration import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from PIL import Image


path_input_dir = Path("data")

path_data = (
                path_input_dir / "dataset_liver_bmodes_steatosis_assessment_IJCARS.mat"
            )

data = scipy.io.loadmat(str(path_data))
data_array = data["data"]
images = data_array["images"]
classes = data_array["class"]
fat = data_array["fat"]

HEALTHY = "Paciente Saudável"

X = []
y = []  
groups = []

with open("src/dados_classificador.csv", newline="") as csvfile:
    reader=csv.reader(csvfile)
    rows=[r for r in reader]
    for i in range(len(images[0])):
        state = 0 if rows[i][1] == HEALTHY else 1
        for j in range(len(images[0][i])):
            X.append(images[0][i][j])
            y.append(state)
            groups.append(i)

logo = LeaveOneGroupOut()

X = np.array(X)
X = X.reshape(X.shape[0], -1)

model = SVC(kernel='linear', random_state=42)

y_pred = cross_val_predict(model, X, y, cv=logo, groups=groups)

cm = confusion_matrix(y, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y, y_pred)
sensitivity = tp / (tp + fn) 
specificity = tn / (tn + fp) 

print("Shallow Classifier")
print(f"Accuracy: {accuracy:.2f}")
print(f"Sensitivity (Recall for Positive Class): {sensitivity:.2f}")
print(f"Specificity (Recall for Negative Class): {specificity:.2f}")
print("\nConfusion Matrix:")
print(cm)
