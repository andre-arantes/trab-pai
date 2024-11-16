from pathlib import Path

import scipy
from sklearn import svm
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

for train, test in logo.split(X, y, groups=groups):
    clf = svm.SVC(kernel='linear', C=1).fit(train, test)
