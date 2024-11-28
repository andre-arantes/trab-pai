from pathlib import Path
import pickle

from matplotlib import pyplot as plt
import numpy as np
import scipy
import csv
from sklearn.svm import SVC
from sklearn.calibration import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from PIL import Image


HEALTHY = "Paciente Saudável"

path_input_dir = Path("images")
X = []
y = []
groups = []
csvIndex = 1
with open("src/dados_hu.csv", newline="") as csvfile:
    reader=csv.reader(csvfile)
    rows=[r for r in reader]
    for index, path in enumerate(sorted(path_input_dir.iterdir(), key=lambda m: int(m.name.split('_')[-1]))):
        for imagePath in path.iterdir():
            state = 0 if rows[csvIndex][1] == HEALTHY else 1
            image = Image.open(imagePath).resize((224, 224)).convert('RGB')
            image = np.array(image)
            X.append(image)
            y.append(state)
            groups.append(index)
            csvIndex += 1

X = np.array(X)
y = np.array(y)
X = X.reshape(X.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = SVC(kernel='linear', random_state=42)

model.fit(X_train, y_train)

file = open("svmModel/svm_model.pkl", "wb")
pickle.dump(model, file)

# logo = LeaveOneGroupOut()

# X = np.array(X)
# X = X.reshape(X.shape[0], -1)

# model = SVC(kernel='linear', random_state=42)

# y_pred = cross_val_predict(model, X, y, cv=logo, groups=groups)

# cm = confusion_matrix(y, y_pred, labels=[0, 1])
# tn, fp, fn, tp = cm.ravel()

# accuracy = accuracy_score(y, y_pred)
# sensitivity = tp / (tp + fn) 
# specificity = tn / (tn + fp) 

# print("Shallow Classifier")
# print(f"Accuracy: {accuracy:.2f}")
# print(f"Sensitivity (Recall for Positive Class): {sensitivity:.2f}")
# print(f"Specificity (Recall for Negative Class): {specificity:.2f}")
# print("\nConfusion Matrix:")
# print(cm)
