import csv
from pathlib import Path
from PIL import Image

from cv2 import imread
from matplotlib import pyplot as plt
import numpy as np
from keras.src.applications.mobilenet_v2 import preprocess_input
from keras.src.applications.mobilenet_v2 import MobileNetV2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from keras.src.losses import sparse_categorical_crossentropy
from keras.src.optimizers import Adam
import tensorflow as tf
from keras.src.callbacks import CSVLogger
from keras.src.layers import Dropout
from keras.src.layers import Dense
from keras.src.models import Model
from keras.src.layers import Conv2D
from keras.src.layers import MaxPooling2D
from keras.src.layers import Flatten
from keras.src.layers import BatchNormalization
from keras.src.regularizers import L2
HEALTHY = "Paciente Saudável"



path_input_dir = Path("images")
X = []
y = []
groups = []
csvIndex = 1
with open("src/dados_classificador.csv", newline="") as csvfile:
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

logo = LeaveOneGroupOut()

acc_per_fold = []
loss_per_fold = []
conf_matrices = []
for i, (train_index, test_index) in enumerate(logo.split(X, y, groups)):
    path = "mobileNetHistory/logs/model_history_log_fold_" + str(i) + ".csv"
    csv_logger = CSVLogger(path, append=True)
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
    base_model = MobileNetV2(weights='imagenet',include_top=False)
    for layer in base_model.layers[:120]:  # Congele as 100 primeiras camadas
        layer.trainable = False 
    model = base_model.output 
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = BatchNormalization()(model)
    model = Dense(128, activation='relu', kernel_regularizer=L2(0.001))(model)
    model = Dropout(0.5)(model)
    model = Dense(64, activation='relu', kernel_regularizer=L2(0.001))(model)
    model = Dropout(0.5)(model)
    output = Dense(2, activation='softmax')(model)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {i} ...')

    
    # Fit data to model
    X_train = preprocess_input(X_train)
    X_test = preprocess_input(X_test)
    history = model.fit(X_train, y_train,
                batch_size=32,
                epochs=20,
                verbose=1, callbacks=[csv_logger], validation_data=(X_test, y_test), )

    # Generate generalization metrics
    scores = model.evaluate(X_test, y_test, verbose=0)
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Get predictions and convert to class labels
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test, y_pred_classes)
    conf_matrices.append(cm)

    plt.clf()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("mobileNetHistory/graphs/loss_fold_" + str(i) + ".png")
    plt.clf()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("mobileNetHistory/graphs/acc_fold_" + str(i) + ".png")


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
avg_conf_matrix = np.sum(conf_matrices, axis=0)
print("Average Confusion Matrix:")
print(avg_conf_matrix)
