import matplotlib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd

matplotlib.use('Qt5Agg')

# Altere o path para o path do arquivo dataset_liver_bmodes_steatosis_assessment_IJCARS.mat no seu computador
path_input_dir = Path('/home/andrelinux/cc6/pai/trab-pai/data')
path_data = path_input_dir / 'dataset_liver_bmodes_steatosis_assessment_IJCARS.mat'

data = scipy.io.loadmat(path_data)

data_array = data['data']
images = data_array['images']

n=1
m=5
imagem = images[0][n][m]
print(imagem.shape)

plt.figure(figsize=(9,9))
plt.imshow(imagem, cmap='gray')
plt.axis('off')  
plt.show()

