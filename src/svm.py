from pathlib import Path

import numpy as np
import scipy
from sklearn.svm import SVC
from sklearn.calibration import cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import LeaveOneGroupOut


path_input_dir = Path("data")

path_data = (
                path_input_dir / "dataset_liver_bmodes_steatosis_assessment_IJCARS.mat"
            )

data = scipy.io.loadmat(str(path_data))
data_array = data["data"]
images = data_array["images"]
classes = data_array["class"]
fat = data_array["fat"]

X = []
y = []
groups = []

for i in range(len(images[0])):
    for j in range(len(images[0][i])):
        X.append(images[0][i][j])
        y.append(int(classes[0][i][0][0]))
        groups.append(i)

logo = LeaveOneGroupOut()

X = np.array(X)
X = X.reshape(X.shape[0], -1)

model = SVC(kernel='linear', random_state=42)

# Predict using cross_val_predict
y_pred = cross_val_predict(model, X, y, cv=logo, groups=groups)

# Calculate confusion matrix
cm = confusion_matrix(y, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

# Calculate metrics
accuracy = accuracy_score(y, y_pred)
sensitivity = tp / (tp + fn)  # Recall for positive class
specificity = tn / (tn + fp)  # Recall for negative class

# Output metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Sensitivity (Recall for Positive Class): {sensitivity:.2f}")
print(f"Specificity (Recall for Negative Class): {specificity:.2f}")
print("\nConfusion Matrix:")
print(cm)
