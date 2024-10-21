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
        self.roi_count = 0
        self.index_img = 0
        self.images = None
        self.patient_number = None
        self.rect = None
        self.list = []
        self.nome_arquivo_csv = "Dados.csv"
        self.patient_class = ""
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

            path_input_dir = Path("/home/andrelinux/cc6/pai/trab-pai/data")
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
            messagebox.showerror(
                "Invalid input. Patient number must be an integer.",
                "Invalid input. Patient number must be an integer.",
            )

    def main_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        # TODO: zoom e olhar histograma
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
        btn_classificate_img.pack(side=tk.LEFT, padx=5)

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

        self.display_image(self.images[0][self.patient_number][self.index_img])
        self.display_histogram(self.images[0][self.patient_number][self.index_img])

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

        frame = tk.Frame(self.root)
        frame.pack(pady=20)
        label = tk.Label(frame, text="TODO")
        label.pack(side=tk.LEFT, padx=5)

    def compute_glcm(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.root)
        frame.pack(pady=20)
        label = tk.Label(frame, text="TODO")
        label.pack(side=tk.LEFT, padx=5)

    def caracterize_roi(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.root)
        frame.pack(pady=20)
        label = tk.Label(frame, text="TODO")
        label.pack(side=tk.LEFT, padx=5)

    def classificate_img(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.root)
        frame.pack(pady=20)
        label = tk.Label(frame, text="TODO")
        label.pack(side=tk.LEFT, padx=5)

    def display_image(self, image):
        self.img = Image.fromarray(image)
        img_tk = ImageTk.PhotoImage(self.img)

        self.canvas_img.config(width=self.img.width, height=self.img.height)
        self.canvas_img.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas_img.image = img_tk

    def display_histogram(self, image):
        if len(image.shape) != 2:
            raise ValueError("Expected a 2D array for grayscale image data.")

        image = Image.fromarray(image)
        histogram, bin_edges = np.histogram(image, bins=256, range=(0, 255))

        fig, ax_hist = plt.subplots(figsize=(5, 4))

        ax_hist.clear()
        ax_hist.plot(bin_edges[0:-1], histogram, color="black")
        ax_hist.set_title("Histogram")
        ax_hist.set_xlim(0, 255)
        ax_hist.set_ylim(0, 4000)
        ax_hist.set_xlabel("Pixel value")
        ax_hist.set_ylabel("Frequency")

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

    def select_roi(self, event):
        global liver_roi, kidney_roi
        self.roi_count += 1
        if self.is_liver_roi():
            liver_roi = ROI(event.x, event.y)
            self.draw_rectangle(liver_roi)
            messagebox.showinfo("Sucesso", "ROI do figado selecionado")
        else:
            kidney_roi = ROI(event.x, event.y)
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
            adjusted_roi_liver_img = self.adjust_liver_roi(roi_liver_img, HI_index)
            self.save_roi(adjusted_roi_liver_img)
            # self.update_csv(
            #     self.patient_number,
            #     self.coord_x,
            #     self.coord_y,
            #     self.x,
            #     self.y,
            #     self.HI_index,
            #     self.patient_class,
            # )
            if self.index_img < len(self.images[0][self.patient_number]):
                self.canvas_img.delete("all")
                self.index_img += 1
                self.display_image(self.images[0][self.patient_number][self.index_img])
                self.roi_count = 0
            else:
                messagebox.showinfo(
                    "Info", "Não há mais imagens disponíveis para este paciente."
                )

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
        x,
        y,
        posicao_X,
        posicao_Y,
        indice_HI,
        classe_paciente,
    ):
        if patient_number <= 16:
            classe_paciente = "Paciente Saudável"
        else:
            classe_paciente = "Paciente com Esteatose"

        nova_linha = {
            "Nome do Arquivo": f"PATIENT_{patient_number}",
            "Roi Fígado X": x,
            "Roi Fígado Y": y,
            "Roi Rim X": posicao_X,
            "Roi Rim Y": posicao_Y,
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
                    x,
                    y,
                    posicao_X,
                    posicao_Y,
                    indice_HI,
                    classe_paciente,
                ]
            )

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


root = tk.Tk()
root.title("Visualizador de Imagens")
root.geometry("1600x800")

app = ImageProcessor(root)

root.mainloop()
