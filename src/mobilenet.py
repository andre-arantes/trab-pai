import csv
from pathlib import Path
from PIL import Image

from cv2 import imread
import numpy as np
from keras.src.applications.mobilenet_v2 import preprocess_input
from keras.src.applications.mobilenet_v2 import MobileNetV2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from keras.src.losses import sparse_categorical_crossentropy
from keras.src.optimizers import Adam
import tensorflow as tf
from keras.src.callbacks import CSVLogger


HEALTHY = "Paciente Saudável"
path_input_dir = Path("images")
X = []
y = []
groups = []

with open("src/dados_classificador.csv", newline="") as csvfile:
    reader=csv.reader(csvfile)
    rows=[r for r in reader]
    for index, path in enumerate(sorted(path_input_dir.iterdir(), key=lambda m: int(m.name.split('_')[-1]))):
        state = 0 if rows[index][1] == HEALTHY else 1
        for imagePath in path.iterdir():
            image = Image.open(imagePath).resize((224, 224)).convert('RGB')
            image = np.array(image)
            X.append(image)
            y.append(state)
            groups.append(index)

X = np.array(X)


logo = LeaveOneGroupOut()
csv_logger = CSVLogger("model_history_log.csv", append=True)

acc_per_fold = []
loss_per_fold = []
all_true_labels = []
all_pred_labels = []
for i, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for index in train_index:
        X_train.append(X[index])
        y_train.append(y[index])
    for index in test_index:
        X_test.append(X[index])
        y_test.append(y[index])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    model = MobileNetV2(weights='imagenet')
    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {i} ...')

    
    # Fit data to model
    X_train = preprocess_input(X_train)
    history = model.fit(X_train, y_train,
                batch_size=50,
                epochs=20,
                verbose=1, callbacks=[csv_logger])

    # Generate generalization metrics
    X_test = preprocess_input(X_test)
    scores = model.evaluate(X_test, y_test, verbose=0)

    # Get predictions and convert to class labels
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
    y_true_classes = np.argmax(y_test, axis=1)  # Convert one-hot encoded to class labels
    
    # Append true and predicted labels
    all_true_labels.extend(y_true_classes)
    all_pred_labels.extend(y_pred_classes)


# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
conf_matrix = confusion_matrix(all_true_labels, all_pred_labels)
print('Confusion Matrix:')
print(conf_matrix)