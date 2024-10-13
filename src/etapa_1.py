import matplotlib
import matplotlib.pyplot as plt
import scipy.io
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from PIL import Image, ImageTk
import os
import numpy as np
import csv

matplotlib.use('TkAgg')

class ProcessaImagem:
    def __init__(self, root):
        self.root = root
        self.img = None
        self.canvas_img = None
        self.start_x = None
        self.start_y = None
        self.contagem_roi = 0
        self.current_image_index = 0
        self.images = None
        self.numero_paciente = None
        self.rect = None
        self.roi_media_figado = None
        self.indice_HI = None
        self.dicionario = []
        self.nome_arquivo_csv = "Dados"
        self.cord_x = None
        self.cord_Y = None
        self.controler = None
        self.setup_menu()

    def setup_menu(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        label = tk.Label(frame, text="Informe o número do paciente:")
        label.pack(side=tk.LEFT, padx=5)

        self.entry_n = tk.Entry(frame)
        self.entry_n.pack(side=tk.LEFT, padx=5)

        btn_load = tk.Button(frame, text="Carregar imagem", command=self.initial_menu)
        btn_load.pack(side=tk.LEFT, padx=5)

        self.canvas_img = tk.Canvas(self.root)
        self.canvas_img.pack(pady=20)

        self.create_csv()

        self.canvas_img.bind("<ButtonPress-1>", self.selecionar_roi)
        
        self.canvas_img.bind("<ButtonPress-1>", self.selecionar_roi)
        


    def initial_menu(self):
        try:
            self.numero_paciente = int(self.entry_n.get())
            
            # Altere o path para o path do arquivo dataset_liver_bmodes_steatosis_assessment_IJCARS.mat no seu computador
            path_input_dir = Path('C:/Users/uni34536/Documents/PC/trab-pai-main/data')
            path_data = path_input_dir / 'dataset_liver_bmodes_steatosis_assessment_IJCARS.mat'

            data = scipy.io.loadmat(path_data)
            data_array = data['data']
            self.images = data_array['images']
            
            self.mostrar_imagem(self.images[0][self.numero_paciente][self.current_image_index])
            self.atualizar_roi_label()
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def mostrar_imagem(self, image):
        self.img = Image.fromarray(image)
        img_tk = ImageTk.PhotoImage(self.img)
        
        self.canvas_img.config(width=self.img.width, height=self.img.height)
        self.canvas_img.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas_img.image = img_tk

    def selecionar_roi(self, event):
        self.start_x, self.start_y = event.x, event.y
        end_x, end_y = event.x, event.y
        
        self.salvar_roi(self.start_x, self.start_y, end_x, end_y)
        self.desenhar_retangulo(self.start_x, self.start_y)
        self.atualizar_roi_label()

        if self.controler != 0:
            self.gerenciar_csv(self.numero_paciente, self.cord_x, self.cord_y, self.start_x, self.start_y, self.indice_HI)
    
    def gerenciar_csv(self, numero_paciente, cord_x, cord_y, posicao_X, posicao_Y, indice_HI):
        nova_linha = {'Nome do Arquivo': f"PATIENT_{numero_paciente}", 'Roi Fígado X': cord_x, 'Roi Fígado Y': cord_y, 'Roi Rim X': posicao_X, 'Roi Rim Y': posicao_Y, 'Índice hepatorenal (HI)': indice_HI}
        self.dicionario.append(nova_linha)
        
        # Acrescenta a nova linha ao arquivo CSV
        with open(self.nome_arquivo_csv, mode='a', newline='', encoding='utf-8') as arquivo_csv:
            escritor = csv.writer(arquivo_csv)
            escritor.writerow([f"PATIENT_{numero_paciente}", cord_x, cord_y, posicao_X, posicao_Y, indice_HI])
    
    def create_csv(self):
        # Cria o arquivo CSV e escreve o cabeçalho (se necessário)
        with open(self.nome_arquivo_csv, mode='w', newline='', encoding='utf-8') as arquivo_csv:
            escritor = csv.writer(arquivo_csv)
            escritor.writerow(['Nome do Arquivo', 'Roi Fígado X', 'Roi Fígado Y', 'Roi Rim Y', 'Roi Rim Y', 'Índice hepatorenal (HI)'])  # Cabeçalho

    def salvar_roi(self, x1, y1, x2, y2):
        roi = self.img.crop((x1, y1, x1 + 28, y1 + 28))
        roi = roi.resize((28, 28), Image.Resampling.LANCZOS)
        
        patient_dir = f"../images/PATIENT_{self.numero_paciente}"
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)
        
        self.contagem_roi += 1
        if self.contagem_roi == 1:
            roi.save(os.path.join(patient_dir, f"ROI_{self.numero_paciente}_{self.current_image_index + 1}.png"))
            messagebox.showinfo("Sucesso", f"ROI do fígado selecionado")
            self.cord_x = x1
            self.cord_y = y1
            self.controler = 0

            
        if self.contagem_roi == 2:
            self.controler = 1
            messagebox.showinfo("Sucesso", f"ROI do rim selecionado")

        print(self.contagem_roi)
        print(self.cord_x, self.cord_y)
        self.gerar_indice_HI(roi, self.contagem_roi)

        self.gerar_histograma(roi, patient_dir, self.contagem_roi, self.numero_paciente, self.current_image_index + 1)
        if self.contagem_roi == 2:
            self.contagem_roi = 0
            self.current_image_index += 1
            if self.current_image_index < len(self.images[0][self.numero_paciente]):
                self.canvas_img.delete("all")
                self.mostrar_imagem(self.images[0][self.numero_paciente][self.current_image_index])
            else:
                messagebox.showinfo("Info", "Não há mais imagens disponíveis para este paciente.")

    def desenhar_retangulo(self, x, y):
        x1 = x
        y1 = y
        x2 = x + 28
        y2 = y + 28

        if self.rect:
            self.canvas_img.delete(self.rect)
        self.rect = self.canvas_img.create_rectangle(x1, y1, x2, y2, outline="green", width=2)

    def atualizar_roi_label(self):
        roi_label = f"ROI {self.contagem_roi + 1}"
        self.root.title(f"Visualizador de Imagens - {roi_label}")

    def gerar_histograma(self, roi, patient_dir, contagem_roi, numero_paciente, index_imagem):
        roi_gray = roi.convert("L")
        
        histogram = roi_gray.histogram()
        
        plt.figure()
        plt.bar(range(256), histogram, width=1, color='black')
        plt.title(f"Histograma referente ao ROI{contagem_roi}_{numero_paciente}_{index_imagem}")
        plt.xlabel("Tom de cinza")
        plt.ylabel("Frequência")
        
        histogram_path = os.path.join(patient_dir, f"Histograma_{numero_paciente}_{index_imagem}.png")
        plt.savefig(histogram_path)
        plt.close()
    
    def gerar_indice_HI(self, roi, contagem_roi):
        roi_rim_media = 0
        if contagem_roi == 1:
            roi_figado_cinza = roi.convert("L")
            roi_figado_array = np.array(roi_figado_cinza)
            self.roi_figado_media = np.mean(roi_figado_array)
        if contagem_roi == 2:
            roi_rim_cinza = roi.convert("L")
            roi_rim_array = np.array(roi_rim_cinza)
            roi_rim_media = np.mean(roi_rim_array)

        if roi_rim_media != 0:
            self.indice_HI = roi_rim_media / self.roi_figado_media
            messagebox.showinfo("Índice HI", f"O índice HI é: {self.indice_HI:.2f}")


root = tk.Tk()
root.title("Visualizador de Imagens")
root.geometry("800x800")

app = ProcessaImagem(root)

root.mainloop()