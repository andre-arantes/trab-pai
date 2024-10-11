import matplotlib
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
        self.roi_count = 0
        self.current_image_index = 0
        self.images = None
        self.patient_number = None
        self.rect = None

        self.setup_menu()

    def setup_menu(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        label = tk.Label(frame, text="Informe o número do paciente:")
        label.pack(side=tk.LEFT, padx=5)

        self.entry_n = tk.Entry(frame)
        self.entry_n.pack(side=tk.LEFT, padx=5)

        btn_load = tk.Button(frame, text="OK", command=self.initial_menu)
        btn_load.pack(side=tk.LEFT, padx=5)

        self.canvas_img = tk.Canvas(self.root, width=800, height=800)
        self.canvas_img.pack(pady=20)

        self.canvas_img.bind("<ButtonPress-1>", self.selecionar_roi)
        self.canvas_img.bind("<ButtonRelease-1>", self.selecionar_roi)

    def initial_menu(self):
        try:
            self.patient_number = int(self.entry_n.get())
            
            # Altere o path para o path do arquivo dataset_liver_bmodes_steatosis_assessment_IJCARS.mat no seu computador
            path_input_dir = Path('/home/andrelinux/cc6/pai/trab-pai/data')
            path_data = path_input_dir / 'dataset_liver_bmodes_steatosis_assessment_IJCARS.mat'

            data = scipy.io.loadmat(path_data)
            data_array = data['data']
            self.images = data_array['images']
            
            self.mostrar_imagem(self.images[0][self.patient_number][self.current_image_index])
            self.atualizar_roi_label()
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def mostrar_imagem(self, image):
        self.img = Image.fromarray(image)
        self.img = self.img.resize((600, 600), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(self.img)
        
        self.canvas_img.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas_img.image = img_tk

    def selecionar_roi(self, event):
        self.start_x, self.start_y = event.x, event.y
        end_x, end_y = event.x, event.y
        self.salvar_roi(self.start_x, self.start_y, end_x, end_y)
        self.desenhar_retangulo(self.start_x, self.start_y)
        self.atualizar_roi_label()

    def salvar_roi(self, x1, y1, x2, y2):
        roi = self.img.crop((x1, y1, x1 + 28, y1 + 28))
        roi = roi.resize((28, 28), Image.Resampling.LANCZOS)
        
        patient_dir = f"../images/PATIENT_{self.patient_number}"
        if not os.path.exists(patient_dir):
            os.makedirs(patient_dir)
        
        save_path = os.path.join(patient_dir, f"PAT_{self.patient_number}_IMG{self.current_image_index + 1}_ROI{self.roi_count + 1}.png")
        roi.save(save_path)
        self.roi_count += 1
        if self.roi_count == 1:
            messagebox.showinfo("Sucesso", f"ROI do fígado disponível em: {save_path}")
        if self.roi_count == 2:
            messagebox.showinfo("Sucesso", f"ROI do rim disponível em: {save_path}")

        if self.roi_count == 2:
            self.roi_count = 0
            self.current_image_index += 1
            if self.current_image_index < len(self.images[0][self.patient_number]):
                self.canvas_img.delete("all")
                self.mostrar_imagem(self.images[0][self.patient_number][self.current_image_index])
            else:
                messagebox.showinfo("Info", "Não há mais imagens disponíveis para este paciente.")

    def desenhar_retangulo(self, x, y):
        x1 = x
        y1 = y
        x2 = x + 28
        y2 = y + 28

        if self.rect:
            self.canvas_img.delete(self.rect)
        self.rect = self.canvas_img.create_rectangle(x1, y1, x2, y2, outline="red", width=2)

    def atualizar_roi_label(self):
        roi_label = f"ROI {self.roi_count + 1}"
        self.root.title(f"Visualizador de Imagens - {roi_label}")

root = tk.Tk()
root.title("Menu")
root.geometry("850x850")

app = ProcessaImagem(root)

root.mainloop()