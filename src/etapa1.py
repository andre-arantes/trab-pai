import csv
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from PIL import Image, ImageTk
import os
import numpy as np
import io
import functools
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
import platform
import cv2
import pandas as pd

matplotlib.use("TkAgg")


class ROI:
    def __init__(self, x, y, width=28, height=28):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.img = None
        self.canvas_img = None
        self.canvas_hist = None
        self.canvas_features = None
        self.roi_count = 0
        self.index_img = 0
        self.images = None
        self.roi_images = []
        self.patient_number = None
        self.rect = None
        self.list = []
        self.nome_arquivo_csv = "Dados.csv"
        self.file_name_classifier = "dados_classificador.csv"
        self.file_name_glmc = "dados_glmc.csv"
        self.patient_class = ""
        self.zoom_level = 1.0
        self.adjusted_roi_liver_img = None
        self.initial_menu()

    def initial_menu(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        label = tk.Label(frame, text="Informe o número do paciente:")
        label.pack(side=tk.LEFT, padx=5)

        self.entry_n = tk.Entry(frame)
        self.entry_n.pack(side=tk.LEFT, padx=5)

        btn_load = tk.Button(frame, text="Visualizar paciente", command=self.setup_menu)
        btn_load.pack(side=tk.LEFT, padx=5)

    def setup_menu(self):
        try:
            self.patient_number = int(self.entry_n.get())
            # Rode o código a partir da rota trab-pai/src
            path_input_dir = Path("../data")
            path_data = (
                path_input_dir / "dataset_liver_bmodes_steatosis_assessment_IJCARS.mat"
            )

            if not path_data.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {path_data}")

            data = scipy.io.loadmat(str(path_data))
            data_array = data["data"]
            self.images = data_array["images"]

            num_patients = len(self.images[0])
            if self.patient_number >= num_patients:
                raise IndexError(
                    f"Paciente de número {self.patient_number} está fora do limite. Número total de pacientes: {num_patients - 1}"
                )
            self.index_img = 0

            self.main_menu()

        except IndexError as e:
            messagebox.showerror("Paciente não encontrado!", str(e))
        except ValueError:
            messagebox.showerror("Erro!", "O campo deve conter um número.")

    def main_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        btn_img_visualization = tk.Button(
            frame, text="Menu de visualização", command=self.visualization_menu
        )
        btn_img_visualization.pack(side=tk.LEFT, padx=5)

        btn_cut = tk.Button(frame, text="Cortar Roi", command=self.cut_roi_menu)
        btn_cut.pack(side=tk.LEFT, padx=5)

        btn_visualize_roi = tk.Button(
            frame, text="Visualizar Roi", command=self.visualize_roi_menu
        )
        btn_visualize_roi.pack(side=tk.LEFT, padx=5)

        btn_compute_glcm = tk.Button(
            frame, text="Computar GLCM", command=self.compute_glcm
        )
        btn_compute_glcm.pack(side=tk.LEFT, padx=5)

        btn_roi_caracterization = tk.Button(
            frame, text="Caracterizar ROI", command=self.caracterize_roi
        )
        btn_roi_caracterization.pack(side=tk.LEFT, padx=5)

        btn_classificate_img = tk.Button(
            frame, text="Classificar imagem", command=self.classificate_img
        )

        btn_classificate_img = tk.Button(
            frame, text="Escolher outro paciente", command=self.reset_variables
        )    

        btn_classificate_img.pack(side=tk.LEFT, padx=5)

    def reset_variables(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        self.img = None
        self.canvas_img = None
        self.canvas_hist = None
        self.canvas_glcm = None
        self.canvas_homo = None
        self.canvas_ent = None
        self.roi_count = 0
        self.index_img = 0
        self.images = None
        self.roi_images = []
        self.patient_number = None
        self.rect = None
        self.list = []
        self.nome_arquivo_csv = "Dados.csv"
        self.patient_class = ""
        self.zoom_level = 1.0
        self.initial_menu()


    
    def visualization_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        display_frame = tk.Frame(self.root)
        display_frame.pack(pady=20)

        self.canvas_img = tk.Canvas(display_frame, width=400, height=400)
        self.canvas_img.grid(row=0, column=0, padx=10, pady=10)

        self.canvas_hist = tk.Canvas(display_frame, width=400, height=400)
        self.canvas_hist.grid(row=0, column=1, padx=10, pady=10)

        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        self.update_header_roi_number()

        btn_prev = tk.Button(frame, text="Voltar imagem", command=self.prev_image)
        btn_prev.pack(side=tk.LEFT, padx=5)

        btn_next = tk.Button(frame, text="Próxima imagem", command=self.next_image)
        btn_next.pack(side=tk.LEFT, padx=5)

        btn_menu = tk.Button(frame, text="Voltar ao menu", command=self.main_menu)
        btn_menu.pack(side=tk.LEFT, padx=5)

        self.display_image(self.images[0][self.patient_number][self.index_img])
        self.display_histogram(self.images[0][self.patient_number][self.index_img])

        if platform.system() == "Linux":
            self.canvas_img.bind(
                "<Button-4>",
                functools.partial(
                    self.zoom, image=self.images[0][self.patient_number][self.index_img]
                ),
            )
            self.canvas_img.bind(
                "<Button-5>",
                functools.partial(
                    self.zoom, image=self.images[0][self.patient_number][self.index_img]
                ),
            )
        else:
            self.canvas_img.bind(
                "<MouseWheel>",
                functools.partial(
                    self.zoom, image=self.images[0][self.patient_number][self.index_img]
                ),
            )

    def zoom(self, event, image):
        if event.delta > 0:  # Zoom in
            self.zoom_level *= 1.1
        else:  # Zoom out
            self.zoom_level *= 0.9

        # display the image again with the new zoom level
        self.display_image(image)

    def cut_roi_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        display_frame = tk.Frame(self.root)
        display_frame.pack(pady=20)

        self.canvas_img = tk.Canvas(display_frame, width=400, height=400)
        self.canvas_img.grid(row=0, column=0, padx=10, pady=10)

        self.canvas_hist = tk.Canvas(display_frame, width=400, height=400)
        self.canvas_hist.grid(row=0, column=1, padx=10, pady=10)

        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        self.update_header_roi_number()

        self.display_image(self.images[0][self.patient_number][self.index_img])

        self.canvas_img.bind("<ButtonPress-1>", lambda event: self.select_roi(event))
        self.canvas_img.bind("<ButtonPress-1>", lambda event: self.select_roi(event))

    def visualize_roi_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        display_frame = tk.Frame(self.root)
        display_frame.pack(pady=20)

        self.canvas_img = tk.Canvas(display_frame, width=400, height=400)
        self.canvas_img.grid(row=0, column=0, padx=10, pady=10)

        self.canvas_hist = tk.Canvas(display_frame, width=400, height=400)
        self.canvas_hist.grid(row=0, column=1, padx=10, pady=10)

        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        self.update_header_roi_number()

        patient_dir = os.path.abspath(f"../images/PATIENT_{self.patient_number}/")
        roi_files = [f for f in os.listdir(patient_dir) if f.startswith("ROI_")]

        if not roi_files:
            messagebox.showinfo("Info", "Não há ROIs salvos para este paciente.")
            return

        for roi_file in roi_files:
            roi_path = os.path.join(patient_dir, roi_file)
            roi_img = Image.open(roi_path).convert("RGB")
            self.roi_images.append(roi_img)

        btn_prev = tk.Button(frame, text="Voltar ROI", command=self.prev_roi)
        btn_prev.pack(side=tk.LEFT, padx=5)

        btn_next = tk.Button(frame, text="Próximo ROI", command=self.next_roi)
        btn_next.pack(side=tk.LEFT, padx=5)

        btn_next = tk.Button(frame, text="Zoom", command=self.zoom_roi)
        btn_next.pack(side=tk.LEFT, padx=5)

        btn_menu = tk.Button(frame, text="Voltar ao menu", command=self.main_menu)
        btn_menu.pack(side=tk.LEFT, padx=5)

        self.display_image(np.array(self.roi_images[self.index_img]))
        self.display_histogram(np.array(self.roi_images[self.index_img]))

        if platform.system() == "Linux":
            self.canvas_img.bind(
                "<Button-4>",
                functools.partial(
                    self.zoom, image=np.array(self.roi_images[self.index_img])
                ),
            )
            self.canvas_img.bind(
                "<Button-5>",
                functools.partial(
                    self.zoom, image=np.array(self.roi_images[self.index_img])
                ),
            )
        else:
            self.canvas_img.bind(
                "<MouseWheel>",
                functools.partial(
                    self.zoom, image=np.array(self.roi_images[self.index_img])
                ),
            )

    def compute_glcm(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        display_frame = tk.Frame(self.root)
        display_frame.pack(pady=20)

        self.canvas_features = tk.Canvas(display_frame, width=400, height=400)
        self.canvas_features.grid(row=0, column=0, padx=10, pady=10)

        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        self.update_header_roi_number()

        patient_dir = os.path.abspath(f"../images/PATIENT_{self.patient_number}/")
        roi_files = [f for f in os.listdir(patient_dir) if f.startswith("ROI_")]

        if not roi_files:
            messagebox.showinfo("Info", "Não há ROIs salvos para este paciente.")
            return

        for roi_file in roi_files:
            roi_path = os.path.join(patient_dir, roi_file)
            roi_img = Image.open(roi_path).convert("L")
            self.roi_images.append(roi_img)

        btn_prev = tk.Button(frame, text="Voltar ROI", command=self.prev_glcm)
        btn_prev.pack(side=tk.LEFT, padx=5)

        btn_next = tk.Button(frame, text="Próximo ROI", command=self.next_glcm)
        btn_next.pack(side=tk.LEFT, padx=5)

        btn_menu = tk.Button(frame, text="Voltar ao menu", command=self.main_menu)
        btn_menu.pack(side=tk.LEFT, padx=5)

        glcm = self.glcm(np.array(self.roi_images[self.index_img]))
        features = self.glcm_features(glcm)
        entropy = shannon_entropy(glcm)
        self.display_features(features, entropy)

        features_csv = self.glcm_features_csv(glcm, entropy)
        self.create_glcm_csv(features_csv, entropy, self.patient_number, self.file_name_glmc)
        
        
        # print(f"GLCM: {glcm}")
        # print(f"features: {features}")
        # print(f"entropy: {entropy}")

        frame = tk.Frame(self.root)
        frame.pack(pady=20)
        label = tk.Label(frame, text="TODO")
        label.pack(side=tk.LEFT, padx=5)
        btn_menu = tk.Button(frame, text="Voltar ao menu", command=self.main_menu)
        btn_menu.pack(side=tk.LEFT, padx=5)

    def create_glcm_csv(self, features, entropy, patient_number, file_name_glmc):
        df = pd.DataFrame(features)
        df.to_csv(file_name_glmc, index=False)
        print(f"Arquivo '{file_name_glmc}' salvo com sucesso.")

    
    def prev_glcm(self):
        if self.index_img > 0:
            self.index_img -= 1
            glcm = self.glcm(np.array(self.roi_images[self.index_img]))
            features = self.glcm_features(glcm)
            entropy = shannon_entropy(glcm)

            self.display_features(features, entropy)
            self.update_header_roi_number()
        else:
            messagebox.showinfo("Fim das imagens", "Essa é o primeiro ROI.")

    def next_glcm(self):
        num_images_per_patient = len(self.images[0][self.patient_number])
        if self.index_img < num_images_per_patient - 1:
            self.index_img += 1
            glcm = self.glcm(np.array(self.roi_images[self.index_img]))
            features = self.glcm_features(glcm)
            entropy = shannon_entropy(glcm)

            self.display_features(features, entropy)
            self.update_header_roi_number()
        else:
            messagebox.showinfo("Fim das imagens", "Essa é a último ROI.")

    def display_features(self, features, entropy):
        # Creating subplots
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle("GLCM Features", fontsize=8)
        plt.figtext(0.5, 0.02, f"Shannon Entropy: {entropy}", ha="center", fontsize=12)

        # Plotting each feature
        feature_names = list(features.keys())
        for i, feature_name in enumerate(feature_names):
            ax = axes[i // 2, i % 2]
            ax.bar(range(len(features[feature_name])), features[feature_name])
            ax.set_title(feature_name)
            ax.set_xlabel("Distance/Angle Index")
            ax.set_ylabel(feature_name)
            ax.set_xticks(range(len(features[feature_name])))
            ax.set_xticklabels(
                [
                    f'Distance {d}, Angle {"{:.2f}".format(a)}'
                    for d in [1, 2, 4, 8]
                    for a in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
                ],
                rotation=45,
                fontsize=6,
            )

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        hist_img = Image.open(buf)
        hist_tk = ImageTk.PhotoImage(hist_img)

        self.canvas_features.config(width=hist_img.width, height=hist_img.height)
        self.canvas_features.create_image(0, 0, anchor=tk.NW, image=hist_tk)
        self.canvas_features.image = hist_tk

        buf.close()

        plt.close(fig)

    def glcm(self, image):
        return graycomatrix(
            image,
            [1, 2, 4, 8],
            [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True,
        )

    def glcm_features(self, glcm):
        contrast = graycoprops(glcm, "contrast").flatten()
        energy = graycoprops(glcm, "energy").flatten()
        homogeneity = graycoprops(glcm, "homogeneity").flatten()
        correlation = graycoprops(glcm, "correlation").flatten()
        glcm_features = {
            "Contrast": contrast,
            "Energy": energy,
            "Homogeneity": homogeneity,
            "Correlation": correlation,
        }
        return glcm_features
    
    def glcm_features_csv(self, glcm, entropy):
        contrast = graycoprops(glcm, "contrast").flatten()
        energy = graycoprops(glcm, "energy").flatten()
        homogeneity = graycoprops(glcm, "homogeneity").flatten()
        correlation = graycoprops(glcm, "correlation").flatten()
        

        glcm_features = {
            "Número do Paciente": f"PATIENT_{self.patient_number}",
            "Número da ROI:": self.index_img,
            "Contrast": contrast,
            "Energy": energy,
            "Homogeneity": homogeneity,
            "Correlation": correlation,
            "Entropia": entropy
        }
        return glcm_features

    def caracterize_roi(self):
        patient_dir = os.path.abspath(f"../images/PATIENT_{self.patient_number}/")
        for widget in self.root.winfo_children():
            widget.destroy()

        self.create_classifier_csv()

        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        roi_files = [f for f in os.listdir(patient_dir) if f.startswith("ROI_")]

        for roi_file in roi_files:
            roi_path = os.path.join(patient_dir, roi_file)
            roi_img = Image.open(roi_path).convert("RGB")
            hu_moments = self.hu_moment_invariants(np.array(roi_img))
            
            if self.patient_number <= 16 :
                input_line = f"PATIENT_{self.patient_number},Paciente Saudável"
            else:
                input_line = f"PATIENT_{self.patient_number},Paciente com Esteatose"

            
            print("Momentos Invariantes de Hu:")
            for i, moment in enumerate(hu_moments, 1):
                input_line = input_line + f",{moment}"
                # print(f"Hu[{i}]: {moment}")

            self.classifier_add_line_csv(input_line)

        btn_menu = tk.Button(frame, text="Voltar ao menu", command=self.main_menu)
        btn_menu.pack(side=tk.LEFT, padx=5)

    def classificate_img(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.root)
        frame.pack(pady=20)
        label = tk.Label(frame, text="TODO")
        label.pack(side=tk.LEFT, padx=5)
        btn_menu = tk.Button(frame, text="Voltar ao menu", command=self.main_menu)
        btn_menu.pack(side=tk.LEFT, padx=5)

    def display_image(self, image):
        self.img = Image.fromarray(image)
        self.img = self.img.resize(
            (
                int(self.img.width * self.zoom_level),
                int(self.img.height * self.zoom_level),
            ),
            Image.Resampling.LANCZOS,
        )
        img_tk = ImageTk.PhotoImage(self.img)

        self.canvas_img.config(width=self.img.width, height=self.img.height)
        self.canvas_img.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas_img.image = img_tk

    def display_histogram(self, image):
        image = Image.fromarray(image)
        histogram, bin_edges = np.histogram(image, bins=256, range=(0, 255))

        fig, ax_hist = plt.subplots(figsize=(5, 4))

        ax_hist.clear()
        ax_hist.plot(bin_edges[0:-1], histogram, color="black")
        ax_hist.set_title("Histogram")
        ax_hist.set_xlim(0, 250)
        y_95th_percentile = np.percentile(histogram, 99)
        y_max_limit = y_95th_percentile * 2.0
        ax_hist.set_ylim(0, y_max_limit)
        ax_hist.set_xlabel("Brightness")
        ax_hist.set_ylabel("Count")

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        hist_img = Image.open(buf)
        hist_tk = ImageTk.PhotoImage(hist_img)

        self.canvas_hist.config(width=hist_img.width, height=hist_img.height)
        self.canvas_hist.create_image(0, 0, anchor=tk.NW, image=hist_tk)
        self.canvas_hist.image = hist_tk

        buf.close()

        plt.close(fig)

    def display_roi(self, image):
        self.img = Image.fromarray(image)
        plt.figure(figsize=(8, 8))
        plt.imshow(self.img, cmap='gray')
        plt.title('ROI')
        plt.grid(False)
        plt.show()
    
    def prev_image(self):
        if self.index_img > 0:
            self.index_img -= 1
            self.display_image(self.images[0][self.patient_number][self.index_img])
            self.display_histogram(self.images[0][self.patient_number][self.index_img])
            self.update_header_roi_number()
        else:
            messagebox.showinfo("Fim das imagens", "Essa é a primeira imagem.")

    def next_image(self):
        num_images_per_patient = len(self.images[0][self.patient_number])
        if self.index_img < num_images_per_patient - 1:
            self.index_img += 1
            self.display_image(self.images[0][self.patient_number][self.index_img])
            self.display_histogram(self.images[0][self.patient_number][self.index_img])
            self.update_header_roi_number()
        else:
            messagebox.showinfo("Fim das imagens", "Essa é a última imagem.")

    def prev_roi(self):
        if self.index_img > 0:
            self.index_img -= 1
            self.display_image(np.array(self.roi_images[self.index_img]))
            self.display_histogram(np.array(self.roi_images[self.index_img]))
            self.update_header_roi_number()
        else:
            messagebox.showinfo("Fim das imagens", "Essa é o primeiro ROI.")

    def next_roi(self):
        num_images_per_patient = len(self.images[0][self.patient_number])
        if self.index_img < num_images_per_patient - 1:
            self.index_img += 1
            self.display_image(np.array(self.roi_images[self.index_img]))
            self.display_histogram(np.array(self.roi_images[self.index_img]))
            self.update_header_roi_number()
        else:
            messagebox.showinfo("Fim das imagens", "Essa é o último ROI.")

    def zoom_roi(self):
        self.display_roi(np.array(self.roi_images[self.index_img]))
  
    
    def select_roi(self, event):
        global liver_roi, kidney_roi
        self.roi_count += 1
        if self.is_liver_roi():
            liver_roi = ROI(event.x, event.y)
            self.draw_rectangle(liver_roi)
            messagebox.showinfo("Sucesso", "ROI do figado selecionado")
        else:
            kidney_roi = ROI(event.x, event.y)
            self.create_csv()
            self.draw_rectangle(kidney_roi)
            messagebox.showinfo("Sucesso", "ROI do rim selecionado")
            self.cut_roi(liver_roi, kidney_roi)

    def cut_roi(self, liver_roi, kidney_roi):
        roi_liver_img = self.crop_img(liver_roi)
        liver_grayscale_mean = self.calculate_grayscale_mean(roi_liver_img)

        roi_kidney_img = self.crop_img(kidney_roi)
        kidney_grayscale_mean = self.calculate_grayscale_mean(roi_kidney_img)

        self.update_header_roi_number()
        if self.roi_count == 2:
            HI_index = self.make_HI_index(liver_grayscale_mean, kidney_grayscale_mean)
            self.adjusted_roi_liver_img = self.adjust_liver_roi(roi_liver_img, HI_index)
            self.save_roi(self.adjusted_roi_liver_img)
            self.update_csv(
                self.patient_number,
                liver_roi.x,
                liver_roi.y,
                kidney_roi.x,
                kidney_roi.y,
                HI_index,
                self.patient_class,
            )
            if self.index_img + 1 < len(self.images[0][self.patient_number]):
                self.canvas_img.delete("all")
                self.index_img += 1
                self.display_image(self.images[0][self.patient_number][self.index_img])
                self.roi_count = 0
            else:
                messagebox.showinfo(
                    "Info", "Não há mais imagens disponíveis para este paciente."
                )
                frame = tk.Frame(self.root)
                frame.pack(pady=20)
                btn_menu = tk.Button(
                    frame, text="Voltar ao menu", command=self.main_menu
                )
                btn_menu.pack(side=tk.LEFT, padx=5)

    def calculate_grayscale_mean(self, roi_img):
        grayscale_image = roi_img.convert("L")
        grayscale_array = np.array(grayscale_image)
        greyscale_mean = np.mean(grayscale_array)
        return greyscale_mean

    def crop_img(self, roi):
        roi_img = self.img.crop(
            (
                roi.x,
                roi.y,
                roi.x + roi.width,
                roi.y + roi.height,
            )
        )
        roi_img = roi_img.resize((roi.width, roi.height), Image.Resampling.LANCZOS)
        return roi_img

    def save_roi(self, roi_img):
        patient_dir = os.path.abspath(f"../images/PATIENT_{self.patient_number}/")
        if not os.path.exists(patient_dir):
            os.mkdir(patient_dir)
        roi_path = os.path.join(
            patient_dir, f"ROI_{self.patient_number}_{self.index_img}.png"
        )

        plt.imshow(roi_img, cmap="gray")
        plt.axis("off")
        plt.savefig(roi_path, bbox_inches="tight", pad_inches=0)
        plt.close()

    def adjust_liver_roi(self, roi_liver_img, HI_index):
        img_array = np.array(roi_liver_img, dtype=np.float32)

        adjusted_array = img_array * HI_index

        adjusted_array = np.clip(adjusted_array, 0, 255).astype(np.uint8)

        adjusted_img = Image.fromarray(adjusted_array)

        return adjusted_img

    def update_csv(
        self,
        patient_number,
        liver_roi_x,
        liver_roi_y,
        kidney_roi_x,
        kidney_roi_y,
        indice_HI,
        classe_paciente,
    ):
        if patient_number <= 16:
            classe_paciente = "Paciente Saudável"
        else:
            classe_paciente = "Paciente com Esteatose"

        nova_linha = {
            "Nome do Arquivo": f"PATIENT_{patient_number}",
            "Roi Fígado X": liver_roi_x,
            "Roi Fígado Y": liver_roi_y,
            "Roi Rim X": kidney_roi_x,
            "Roi Rim Y": kidney_roi_y,
            "Índice hepatorenal (HI)": indice_HI,
            "Classe": classe_paciente,
        }
        self.list.append(nova_linha)

        with open(
            self.nome_arquivo_csv, mode="a", newline="", encoding="utf-8"
        ) as arquivo_csv:
            writer = csv.writer(arquivo_csv)
            writer.writerow(
                [
                    f"PATIENT_{patient_number}",
                    liver_roi_x,
                    liver_roi_y,
                    kidney_roi_x,
                    kidney_roi_y,
                    indice_HI,
                    classe_paciente,
                ]
            )
    def classifier_add_line_csv(
        self,
        input_line
        
    ):

        line_in_list = input_line.split(",")
        with open(
            self.file_name_classifier, mode="a", newline="", encoding="utf-8"
        ) as arquivo_csv:
            writer = csv.writer(arquivo_csv, delimiter=",")
            writer.writerow(line_in_list)

    def create_csv(self):
        if not os.path.exists(self.nome_arquivo_csv):
            with open(
                self.nome_arquivo_csv, mode="w", newline="", encoding="utf-8"
            ) as arquivo_csv:
                writer = csv.writer(arquivo_csv)
                writer.writerow(
                    [
                        "Nome do Arquivo",
                        "Roi Fígado X",
                        "Roi Fígado Y",
                        "Roi Rim X",
                        "Roi Rim Y",
                        "Índice hepatorenal (HI)",
                        "Classe",
                    ]
                )

    def create_classifier_csv(self):
        if not os.path.exists(self.file_name_classifier):
            with open(
                self.file_name_classifier, mode="w", newline="", encoding="utf-8"
            ) as arquivo_csv:
                writer = csv.writer(arquivo_csv)
                writer.writerow(
                    [
                        "Nome do Arquivo",
                        "Classe",
                        "Hu 1",
                        "Hu 2",
                        "Hu 3",
                        "Hu 4",
                        "Hu 5",
                        "Hu 6",
                        "Hu 7",
                    ]
                )

    def draw_rectangle(self, roi):
        x1 = roi.x
        y1 = roi.y
        x2 = roi.x + 28
        y2 = roi.y + 28

        if self.rect:
            self.canvas_img.delete(self.rect)

        self.rect = self.canvas_img.create_rectangle(
            x1, y1, x2, y2, outline="green", width=2
        )

    def is_liver_roi(self):
        return self.roi_count == 1

    def update_header_roi_number(self):
        roi_label = f"ROI {self.index_img + 1}"
        self.root.title(f"Visualizador de Imagens - {roi_label}")

    def create_histogram(
        self,
        roi_image,
        patient_dir,
        roi_count,
        patient_number,
        index_image,
        greyscale_mean,
    ):
        roi_gray = roi_image.convert("L")

        histogram = roi_gray.histogram()

        plt.figure()
        plt.bar(range(256), histogram, width=1, color="black")
        plt.title(
            f"Histograma referente ao ROI{roi_count}_{patient_number}_{index_image}"
        )
        plt.xlabel("Brightness")
        plt.ylabel("Count")

        x_max = plt.gca().get_xlim()[1]
        y_max = plt.gca().get_ylim()[1]

        plt.text(
            x=x_max * 0.75,
            y=y_max * 0.9,
            s=f"Média: {greyscale_mean:.2f}",
            fontsize=12,
            color="black",
        )

        if self.is_liver_roi():
            histogram_path = os.path.join(
                patient_dir, f"Histograma_{patient_number}_{index_image}.png"
            )
            plt.savefig(histogram_path)
            plt.close()
        else:
            histogram_path = os.path.join(
                patient_dir, f"RIM-Histograma_{patient_number}_{index_image}.png"
            )
            plt.savefig(histogram_path)
            plt.close()

    def make_HI_index(self, liver_grayscale_mean, kidney_grayscale_mean):
        HI_index = liver_grayscale_mean / kidney_grayscale_mean
        messagebox.showinfo("Índice HI", f"O índice HI é: {HI_index:.2f}")
        return HI_index

    def hu_moment_invariants(self, roi_img):
        image = cv2.cvtColor(np.array(roi_img), cv2.COLOR_RGB2GRAY)

        feature = cv2.HuMoments(cv2.moments(image)).flatten()

        return feature


root = tk.Tk()
root.title("Visualizador de Imagens")
root.attributes("-fullscreen", True)

app = ImageProcessor(root)

root.mainloop()
