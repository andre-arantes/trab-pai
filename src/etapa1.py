import matplotlib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.io
import tkinter as tk
from tkinter import messagebox

matplotlib.use('Qt5Agg')

def initial_menu():
    try:
        n = int(entry_n.get())
        m = int(entry_m.get())
        
        # Altere o path para o path do arquivo dataset_liver_bmodes_steatosis_assessment_IJCARS.mat no seu computador
        path_input_dir = Path('/home/andrelinux/cc6/pai/trab-pai/data')
        path_data = path_input_dir / 'dataset_liver_bmodes_steatosis_assessment_IJCARS.mat'

        data = scipy.io.loadmat(path_data)
        data_array = data['data']
        images = data_array['images']

        # Obter a imagem com os índices fornecidos
        imagem = images[0][n][m]

        # Exibir a imagem
        plt.figure(figsize=(9, 9))
        plt.imshow(imagem, cmap='gray')
        plt.axis('off')  
        plt.show()

    except Exception as e:
        messagebox.showerror("Erro", str(e))

# Configuração da janela principal
root = tk.Tk()
root.title("Visualizador de Imagens")
root.geometry("300x300")

# Labels e campos de entrada
label_n = tk.Label(root, text="Número do paciente:")
label_n.pack(pady=5)

entry_n = tk.Entry(root)
entry_n.pack(pady=5)

label_m = tk.Label(root, text="Imagem:")
label_m.pack(pady=5)

entry_m = tk.Entry(root)
entry_m.pack(pady=5)

# Botão para mostrar a imagem
button_show = tk.Button(root, text="Mostrar Imagem", command=initial_menu)
button_show.pack(pady=20)

# Inicialização da interface gráfica
root.mainloop()