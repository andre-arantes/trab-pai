import csv
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import scipy
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Classe do Menu Principal
class MenuPrincipal(tk.Frame):
    def __init__(self, root, controller):
        tk.Frame.__init__(self, root)
        
        # Título do menu principal
        titulo = tk.Label(self, text="Trabalho de PAI", font=("Arial", 24))
        titulo.pack(pady=20)

        # Botão para o submenu "Separar ROI's"
        btn_separar = tk.Button(self, text="Separar ROI's", font=("Arial", 14), 
                                command=lambda: controller.mostrar_frame(MenuSepararROIs))
        btn_separar.pack(pady=10)

        # Botão para o submenu "Visualizar ROI's"
        btn_visualizar = tk.Button(self, text="Visualizar ROI's", font=("Arial", 14), 
                                   command=lambda: controller.mostrar_frame(MenuVisualizarROIs))
        btn_visualizar.pack(pady=10)

# Classe do Menu Separar ROI's
class MenuSepararROIs(tk.Frame):
    def __init__(self, root, controller):
        tk.Frame.__init__(self, root)
        self.root = root
        self.controller = controller
        
        # Inicializa as variáveis necessárias
        self.img = None
        self.canvas_img = None
        self.start_x = None
        self.start_y = None
        self.contagem_roi = 0
        self.index_img = 0
        self.images = None
        self.numero_paciente = None
        self.rect = None
        self.roi_rim_media = 0
        self.indice_HI = None
        self.dicionario = []
        # self.nome_arquivo_csv = "../images/Dados.csv"
        self.classe_paciente = ''
        self.cord_x = None
        self.cord_Y = None

        self.setup_menu()

    def setup_menu(self):
        frame = tk.Frame(self)
        frame.pack(pady=20)

        # Label e campo de entrada para o número do paciente
        label = tk.Label(frame, text="Informe o número do paciente:")
        label.pack(side=tk.LEFT, padx=5)
        
        self.entry_n = tk.Entry(frame)
        self.entry_n.pack(side=tk.LEFT, padx=5)

        # Botão para carregar a imagem
        btn_load = tk.Button(frame, text="Carregar imagem", command=self.initial_menu)
        btn_load.pack(side=tk.LEFT, padx=5)

        # Canvas para mostrar a imagem
        self.canvas_img = tk.Canvas(self)
        self.canvas_img.pack(pady=20)
        

        # Bind dos eventos de mouse para seleção da ROI
        self.canvas_img.bind("<ButtonPress-1>", self.selecionar_roi)
        self.canvas_img.bind("<ButtonPress-1>", self.selecionar_roi)

        # Adicionar o botão de Voltar
        btn_voltar = tk.Button(self, text="Voltar ao Menu Principal", 
                               command=lambda: self.controller.mostrar_frame(MenuPrincipal))
        btn_voltar.pack(pady=20)


    def initial_menu(self):
        try:
            self.numero_paciente = int(self.entry_n.get())
            
            # Altere o path para o path do arquivo dataset_liver_bmodes_steatosis_assessment_IJCARS.mat no seu computador
            path_input_dir = Path('C:/Users/uni34536/Documents/PC/trab-pai-main/data')
            path_data = path_input_dir / 'dataset_liver_bmodes_steatosis_assessment_IJCARS.mat'

            if not path_data.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {path_data}")

            data = scipy.io.loadmat(str(path_data))
            data_array = data['data']
            self.images = data_array['images']
    
            self.mostrar_imagem(self.images[0][self.numero_paciente][self.index_img])
            self.atualizar_label_roi()
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def selecionar_roi(self, event):
        self.start_x, self.start_y = event.x, event.y
        # print(f"start_x: {self.start_x}, start_y: {self.start_y}")
        self.salvar_roi(self.start_x, self.start_y)
        self.desenhar_retangulo(self.start_x, self.start_y)
        self.atualizar_label_roi()
        print(self.controler)
        self.criar_csv()  # Método para criar o arquivo CSV se necessário  
        if self.controler != 0:
            self.gerenciar_csv(self.numero_paciente, self.cord_x, self.cord_y, self.start_x, self.start_y, self.indice_HI, self.classe_paciente)

    def gerenciar_csv(self, numero_paciente, cord_x, cord_y, posicao_X, posicao_Y, indice_HI, classe_paciente):

        if numero_paciente <= 16:
            classe_paciente = 'Paciente Saudável'
        else:
            classe_paciente = 'Paciente com Esteatose'

        nova_linha = {'Nome do Arquivo': f"PATIENT_{numero_paciente}", 'Roi Fígado X': cord_x, 'Roi Fígado Y': cord_y, 'Roi Rim X': posicao_X, 'Roi Rim Y': posicao_Y, 'Índice hepatorenal (HI)': indice_HI, 'Classe': classe_paciente}
        self.dicionario.append(nova_linha)
        
        with open(f"../images/PATIENT_{self.numero_paciente}/Dados.csv", mode='a', newline='', encoding='utf-8') as arquivo_csv:
            escritor = csv.writer(arquivo_csv)
            escritor.writerow([f"PATIENT_{numero_paciente}", cord_x, cord_y, posicao_X, posicao_Y, indice_HI, classe_paciente])
    
    def criar_csv(self):
        if not os.path.exists(f"../images/PATIENT_{self.numero_paciente}/Dados.csv"):
            with open(f"../images/PATIENT_{self.numero_paciente}/Dados.csv", mode='w', newline='', encoding='utf-8') as arquivo_csv:
                escritor = csv.writer(arquivo_csv)
                escritor.writerow(['Nome do Arquivo', 'Roi Fígado X', 'Roi Fígado Y', 'Roi Rim X', 'Roi Rim Y', 'Índice hepatorenal (HI)', 'Classe'])  # Cabeçalho
    
    
    def mostrar_imagem(self, image):
        self.img = Image.fromarray(image)
        img_tk = ImageTk.PhotoImage(self.img)
        
        self.canvas_img.config(width=self.img.width, height=self.img.height)
        self.canvas_img.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas_img.image = img_tk
    def desenhar_retangulo(self, x, y):
        x1 = x
        y1 = y
        x2 = x + 28
        y2 = y + 28

        if self.rect:
            self.canvas_img.delete(self.rect)
        self.rect = self.canvas_img.create_rectangle(x1, y1, x2, y2, outline="green", width=2)

    def atualizar_label_roi(self):
        roi_label = f"ROI {self.contagem_roi + 1}"
        self.root.title(f"Visualizador de Imagens - {roi_label}")

    def salvar_roi(self, x1, y1):
        roi = self.img.crop((x1, y1, x1 + 28, y1 + 28))
        roi = roi.resize((28, 28), Image.Resampling.LANCZOS)
        
        patient_dir = f"../images/PATIENT_{self.numero_paciente}/ROI{self.index_img + 1}"
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)
        
        self.contagem_roi += 1
        # Primeiro salva o ROI do rim depois do fígado
        if self.contagem_roi == 1: # Salva o ROI do RIM 
            self.controler = 0
            self.cord_x = x1
            self.cord_y = y1

            nome_arquivo = f"ROI_{self.numero_paciente}_{self.index_img + 1}.png"
            roi.save(os.path.join(patient_dir, nome_arquivo))
            messagebox.showinfo("Sucesso", f"ROI do rim selecionado")

            self.gerar_indice_HI(roi, self.contagem_roi)
            self.gerar_histograma(roi, patient_dir, self.contagem_roi, self.numero_paciente, self.index_img + 1, True)

        if self.contagem_roi == 2: # Salva o ROI do fígado 
            self.controler = 1
            self.gerar_indice_HI(roi, self.contagem_roi)
            self.contagem_roi = 0
            print(self.indice_HI)
            roi_ajustado = self.ajustar_roi(roi, self.indice_HI)
            self.index_img += 1
            nome_arquivo = f"ROI_{self.numero_paciente}_{self.index_img}.png"
            roi_ajustado.save(os.path.join(patient_dir, nome_arquivo))
            messagebox.showinfo("Sucesso", f"ROI do fígado selecionado")

            self.gerar_histograma(roi, patient_dir, self.contagem_roi, self.numero_paciente, self.index_img, False)
            self.gerar_indice_HI(roi, self.contagem_roi)

            if self.index_img < len(self.images[0][self.numero_paciente]):
                self.canvas_img.delete("all")
                self.mostrar_imagem(self.images[0][self.numero_paciente][self.index_img])
            else:
                messagebox.showinfo("Info", "Não há mais imagens disponíveis para este paciente.")

    def gerar_histograma(self, roi, patient_dir, contagem_roi, numero_paciente, index_imagem, is_rim):
        roi_gray = roi.convert("L")
        
        histograma = roi_gray.histogram()

        total_pixels = sum(histograma)
        soma_tons_cinza = sum(valor * contagem for valor, contagem in enumerate(histograma))
        media_tons_cinza = soma_tons_cinza / total_pixels

        
        plt.figure()
        plt.bar(range(256), histograma, width=1, color='black')
        plt.title(f"Histograma referente ao ROI{contagem_roi}_{numero_paciente}_{index_imagem}")
        plt.xlabel("Brightness")
        plt.ylabel("Count")

        x_max = plt.gca().get_xlim()[1]
        y_max = plt.gca().get_ylim()[1]


        plt.text(x=x_max * 0.75, y=y_max * 0.9, s=f'Média: {media_tons_cinza:.2f}', fontsize=12, color='black')

        if(is_rim):
            histogram_path = os.path.join(patient_dir, f"RIM-Histograma_{numero_paciente}_{index_imagem}.png")
            plt.savefig(histogram_path)
            plt.close()
        else:
            histogram_path = os.path.join(patient_dir, f"Histograma_{numero_paciente}_{index_imagem}.png")
            plt.savefig(histogram_path)
            plt.close()
    
    def gerar_indice_HI(self, roi, contagem_roi):
        roi_figado_media = 0
        if contagem_roi == 1:
            roi_rim_cinza = roi.convert("L")
            roi_rim_array = np.array(roi_rim_cinza)
            self.roi_rim_media = np.mean(roi_rim_array)
        if contagem_roi == 2:
            roi_figado_cinza = roi.convert("L")
            roi_figado_array = np.array(roi_figado_cinza)
            roi_figado_media = np.mean(roi_figado_array)

        if roi_figado_media != 0:
            self.indice_HI = roi_figado_media / self.roi_rim_media
            messagebox.showinfo("Índice HI", f"O índice HI é: {self.indice_HI:.2f}")
    
    def ajustar_roi(self, roi, HI):
        # Converte a imagem para modo 'L' (tons de cinza)
        roi = roi.convert("L")
        pixels = roi.load()

        # Itera sobre cada pixel e ajusta o valor multiplicando pelo HI
        for i in range(roi.size[0]):
            for j in range(roi.size[1]):
                pixel_valor = pixels[i, j]
                novo_valor = int(pixel_valor * HI)
                # Garante que o novo valor esteja entre 0 e 255
                pixels[i, j] = min(255, max(0, novo_valor))
            
        return roi


        

# Classe do Menu Visualizar ROI's
class MenuVisualizarROIs(tk.Frame):
    def __init__(self, root, controller):
        tk.Frame.__init__(self, root)
        
        # Título do submenu
        label = tk.Label(self, text="Visualizar ROI's", font=("Arial", 20))
        label.pack(pady=20)

        # Botão de Voltar para o menu principal
        btn_voltar = tk.Button(self, text="Voltar", font=("Arial", 14), 
                               command=lambda: controller.mostrar_frame(MenuPrincipal))
        btn_voltar.pack(pady=10)

# Controlador da aplicação
class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Menu Principal - Trabalho de PAI")
        self.geometry("600x400")
        
        # Dicionário para armazenar os frames
        self.frames = {}

        # Adiciona cada frame ao dicionário
        for F in (MenuPrincipal, MenuSepararROIs, MenuVisualizarROIs):
            frame = F(self, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        # Mostra o menu principal
        self.mostrar_frame(MenuPrincipal)

    def mostrar_frame(self, nome_frame):
        frame = self.frames[nome_frame]
        frame.tkraise()

# Inicia a aplicação
if __name__ == "__main__":
    app = App()
    app.mainloop()
