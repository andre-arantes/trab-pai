import csv
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from PIL import Image, ImageTk
import os

matplotlib.use('TkAgg')

class ProcessaImagem:
    def __init__(self, root):
        self.root = root
        self.img = None
        self.canvas_img = None
        self.start_x = None
        self.start_y = None
        self.contagem_roi = 0
        self.index_img = 0
        self.images = None
        self.numero_paciente = None
        self.rect = None

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

        self.canvas_img.bind("<ButtonPress-1>", self.selecionar_roi)
        self.canvas_img.bind("<ButtonRelease-1>", self.selecionar_roi)

    def initial_menu(self):
        try:
            self.numero_paciente = int(self.entry_n.get())
            
            # Altere o path para o path do arquivo dataset_liver_bmodes_steatosis_assessment_IJCARS.mat no seu computador
            path_input_dir = Path('/home/andrelinux/cc6/pai/trab-pai/data')
            path_data = path_input_dir / 'dataset_liver_bmodes_steatosis_assessment_IJCARS.mat'

            data = scipy.io.loadmat(path_data)
            data_array = data['data']
            self.images = data_array['images']
            
            self.mostrar_imagem(self.images[0][self.numero_paciente][self.index_img])
            self.atualizar_label_roi()
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
        print(f"start_x: {self.start_x}, start_y: {self.start_y}")
        self.salvar_roi(self.start_x, self.start_y)
        self.desenhar_retangulo(self.start_x, self.start_y)
        self.atualizar_label_roi()

    def salvar_roi(self, x1, y1):
        roi = self.img.crop((x1, y1, x1 + 28, y1 + 28))
        roi = roi.resize((28, 28), Image.Resampling.LANCZOS)
        
        patient_dir = f"../images/PATIENT_{self.numero_paciente}"
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)
        
        self.contagem_roi += 1
        if self.contagem_roi == 1:
            nome_arquivo = f"ROI_{self.numero_paciente}_{self.index_img + 1}.png"
            roi.save(os.path.join(patient_dir, nome_arquivo))
            messagebox.showinfo("Sucesso", f"ROI do fígado selecionado")
            self.gerar_histograma(roi, patient_dir, self.contagem_roi, self.numero_paciente, self.index_img + 1, False)
        if self.contagem_roi == 2:
            nome_arquivo = f"ROI_{self.numero_paciente}_{self.index_img + 1}.png"
            roi.save(os.path.join(patient_dir, nome_arquivo))
            messagebox.showinfo("Sucesso", f"ROI do rim selecionado")
            self.gerar_histograma(roi, patient_dir, self.contagem_roi, self.numero_paciente, self.index_img + 1, True)


        if self.contagem_roi == 2:
            self.contagem_roi = 0
            self.index_img += 1
            if self.index_img < len(self.images[0][self.numero_paciente]):
                self.canvas_img.delete("all")
                self.mostrar_imagem(self.images[0][self.numero_paciente][self.index_img])
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

    def atualizar_label_roi(self):
        roi_label = f"ROI {self.contagem_roi + 1}"
        self.root.title(f"Visualizador de Imagens - {roi_label}")

    def gerar_histograma(self, roi, patient_dir, contagem_roi, numero_paciente, index_imagem, is_rim):
        # MUDAR DESSE LUGAR, APENAS TESTE
        # novo_dado = pd.DataFrame({'nome do arquivo': [nome_do_arquivo], 'total': [total]})

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



        # try:
        #     df_existente = pd.read_csv(nome_arquivo)

        #     # Concatenando o novo dado ao DataFrame existente
        #     df_atualizado = pd.concat([df_existente, novo_dado], ignore_index=True)
        # except FileNotFoundError:
        #     # Caso o arquivo não exista, o novo dado será o DataFrame completo
        #     df_atualizado = novo_dado

        # # Salvando o DataFrame atualizado de volta no arquivo CSV
        # df_atualizado.to_csv(nome_arquivo, index=False)



        if(is_rim):
            histogram_path = os.path.join(patient_dir, f"RIM-Histograma_{numero_paciente}_{index_imagem}.png")
            plt.savefig(histogram_path)
            plt.close()
        else:
            histogram_path = os.path.join(patient_dir, f"Histograma_{numero_paciente}_{index_imagem}.png")
            plt.savefig(histogram_path)
            plt.close()


root = tk.Tk()
root.title("Visualizador de Imagens")
root.geometry("800x800")

app = ProcessaImagem(root)

root.mainloop()