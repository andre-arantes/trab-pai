import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
from tkinter import Tk, Button, Label, filedialog
from PIL import Image, ImageTk

def carregar_imagem_mat(caminho):
    # Carregar o arquivo .mat
    dados = loadmat(caminho)
    for chave in dados:
        if isinstance(dados[chave], np.ndarray):
            imagem = dados[chave]
            break
    return imagem

def carregar_imagem(caminho):
    extensao = os.path.splitext(caminho)[-1].lower()
    
    if extensao == ".mat":
        imagem = carregar_imagem_mat(caminho)
    elif extensao in [".png", ".jpg", ".jpeg"]:
        imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError("Formato de arquivo não suportado.")
    
    return imagem

def exibir_imagem(imagem):
    img_pil = Image.fromarray(imagem)
    img_tk = ImageTk.PhotoImage(img_pil)
    
    label_imagem.config(image=img_tk)
    label_imagem.image = img_tk  # Para evitar que a imagem seja descartada

def exibir_histograma(imagem):
    histograma, bins = np.histogram(imagem.flatten(), 256, [0, 256])
    
    plt.figure()
    plt.plot(histograma, color='black')
    plt.title("Histograma de tons de cinza")
    plt.xlabel("Intensidade de pixel")
    plt.ylabel("Número de pixels")
    plt.xlim([0, 256])
    plt.show()

def carregar_arquivo():
    caminho = filedialog.askopenfilename(title="Selecione a imagem", 
                                         filetypes=[("Todos os arquivos", "*.png;*.jpg;*.jpeg;*.mat"),
                                                    ("Arquivos PNG", "*.png"),
                                                    ("Arquivos JPG", "*.jpg;*.jpeg"),
                                                    ("Arquivos MAT", "*.mat")])
    if caminho:
        try:
            global imagem
            imagem = carregar_imagem(caminho)
            exibir_imagem(imagem)
        except Exception as e:
            label_imagem.config(text=f"Erro ao carregar a imagem: {e}")

def mostrar_histograma():
    if imagem is not None:
        exibir_histograma(imagem)
    else:
        label_imagem.config(text="Nenhuma imagem foi carregada ainda.")

# Configurando a interface Tkinter
root = Tk()
root.title("Visualizador de Imagens e Histograma")

# Botão para carregar a imagem
btn_carregar = Button(root, text="Carregar Imagem", command=carregar_arquivo)
btn_carregar.pack(pady=10)

# Botão para exibir o histograma
btn_histograma = Button(root, text="Exibir Histograma", command=mostrar_histograma)
btn_histograma.pack(pady=10)

# Label onde a imagem será exibida
label_imagem = Label(root)
label_imagem.pack(pady=10)

# Variável global para armazenar a imagem carregada
imagem = None

# Iniciar a aplicação
root.mainloop()
