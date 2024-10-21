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

matplotlib.use("TkAgg")


class ROI:
    global media_1
    global media_2

    def __init__(self, x_liver, y_liver, x_kidney, y_kidney, width=28, height=28):
        self.x_liver = x_liver
        self.y_liver = y_liver
        self.x_kidney = x_kidney
        self.y_kidney = y_kidney
        self.width = width
        self.height = height
        self.greyscale_mean = None

    def calculate_grayscale_mean(self, roi_img):
        grayscale_image = roi_img.convert("L")
        grayscale_array = np.array(grayscale_image)
        return np.mean(grayscale_array)

    def select_roi(self, processor, x_liver, y_liver, x_kidney, y_kidney):
        if processor.roi_count == 1:
            processor.draw_rectangle(x_liver, y_liver)
            messagebox.showinfo("Sucesso", "ROI do rim selecionado")
            roi_liver_img = processor.img.crop(
                (
                    self.x_liver,
                    self.y_liver,
                    self.x_liver + self.width,
                    self.y_liver + self.height,
                )
            )
            roi_liver_img = roi_liver_img.resize(
                (self.width, self.height), Image.Resampling.LANCZOS
            )
            processor.grayscale_mean = self.calculate_grayscale_mean(roi_liver_img)
            print(processor.grayscale_mean)

        if processor.roi_count == 2:
            processor.draw_rectangle(x_liver, y_liver)
            roi_img = processor.img.crop(
                (self.x, self.y, self.x + self.width, self.y + self.height)
            )
            roi_img = roi_img.resize(
                (self.width, self.height), Image.Resampling.LANCZOS
            )
            processor.grayscale_mean = self.calculate_grayscale_mean(roi_img)

            print("anv", x_liver, y_liver, x_kidney, y_kidney)

        processor.update_header_roi_number()
        if processor.roi_count == 2:
            processor.update_csv(
                processor.patient_number,
                processor.coord_x,
                processor.coord_y,
                self.x,
                self.y,
                processor.HI_index,
                processor.patient_class,
            )
            if processor.index_img < len(processor.images[0][processor.patient_number]):
                processor.canvas_img.delete("all")
                processor.display_image(
                    processor.images[0][processor.patient_number][
                        processor.index_img + 1
                    ]
                )
                processor.roi_count = 0
            else:
                messagebox.showinfo(
                    "Info", "Não há mais imagens disponíveis para este paciente."
                )

    def save_roi(self, processor):
        roi_img = processor.img.crop(
            (self.x, self.y, self.x + self.width, self.y + self.height)
        )
        roi_img = roi_img.resize((self.width, self.height), Image.Resampling.LANCZOS)

        processor.grayscale_mean = self.calculate_grayscale_mean(roi_img)

        patient_dir = os.path.abspath(f"../images/PATIENT_{processor.patient_number}/")
        if not os.path.exists(patient_dir):
            os.mkdir(patient_dir)

        processor.roi_count += 1
        # if processor.is_liver_roi():
        #     processor.liver_grayscale = self.calculate_grayscale_mean(roi_img)
        #     media_1 = processor.liver_grayscale
        #     print("media 1", media_1)
        #     processor.create_histogram(
        #         roi_img,
        #         patient_dir,
        #         processor.roi_count,
        #         processor.patient_number,
        #         processor.index_img + 1,
        #         processor.liver_grayscale,
        #     )
        #     messagebox.showinfo("Sucesso", "ROI do fígado selecionado")
        # else:
        # processor.kidney_grayscale = self.calculate_grayscale_mean(roi_img)
        # media_2 = processor.kidney_grayscale
        # print("media 2", media_2)
        # messagebox.showinfo("Sucesso", "ROI do rim selecionado")
        # processor.create_histogram(
        #     roi_img,
        #     patient_dir,
        #     processor.roi_count,
        #     processor.patient_number,
        #     processor.index_img + 1,
        #     processor.kidney_grayscale,
        # )

        # processor.make_HI_index(self, media_1, media_2)


class ImageProcessor:
    def __init__(self, root):
        self.root = root
        self.img = None
        self.canvas_img = None
        self.roi_x = None
        self.roi_y = None
        self.roi_count = 0
        self.index_img = 0
        self.images = None
        self.patient_number = None
        self.rect = None
        self.HI_index = None
        self.list = []
        self.nome_arquivo_csv = "Dados.csv"
        self.patient_class = ""
        self.coord_x = None
        self.coord_y = None
        self.liver_grayscale = None
        self.kidney_grayscale = None
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
        self.canvas_img.bind("<ButtonPress-1>", lambda event: self.select_roi(event))
        self.canvas_img.bind("<ButtonPress-1>", lambda event: self.select_roi(event))

    def initial_menu(self):
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

            self.display_image(self.images[0][self.patient_number][self.index_img])
            self.update_header_roi_number()
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def display_image(self, image):
        self.img = Image.fromarray(image)
        img_tk = ImageTk.PhotoImage(self.img)

        self.canvas_img.config(width=self.img.width, height=self.img.height)
        self.canvas_img.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas_img.image = img_tk

    def select_roi(self, event):
        global x_liver, y_liver, x_kidney, y_kidney
        self.roi_count += 1
        if self.is_liver_roi():
            x_liver, y_liver = event.x, event.y
        else:
            x_kidney, y_kidney = event.x, event.y
            roi = ROI(x_liver, y_liver, x_kidney, y_kidney)
            roi.select_roi(self, x_liver, y_liver, x_kidney, y_kidney)

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

    def adjust_liver_roi(self, roi, HI):
        if HI is None:
            raise ValueError("HI (Hepatorenal Index) is not set.")

        roi = roi.convert("L")
        pixels = roi.load()

        for i in range(roi.size[0]):
            for j in range(roi.size[1]):
                pixel_valor = pixels[i, j]
                novo_valor = int(pixel_valor * HI)
                pixels[i, j] = min(255, max(0, novo_valor))

        return roi

    def draw_rectangle(self, x, y):
        x1 = x
        y1 = y
        x2 = x + 28
        y2 = y + 28

        if self.rect:
            self.canvas_img.delete(self.rect)
        if self.is_liver_roi():
            self.rect = self.canvas_img.create_rectangle(
                x1, y1, x2, y2, outline="green", width=2
            )

    def is_liver_roi(self):
        return self.roi_count == 1

    def update_header_roi_number(self):
        roi_label = f"ROI {self.roi_count + 1}"
        self.root.title(f"Visualizador de Ultrassons - {roi_label}")

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
        print("liver_grayscale_mean", liver_grayscale_mean)
        print("kidney_grayscale_mean", kidney_grayscale_mean)
        self.HI_index = liver_grayscale_mean / kidney_grayscale_mean
        messagebox.showinfo("Índice HI", f"O índice HI é: {self.HI_index:.2f}")


root = tk.Tk()
root.title("Visualizador de Imagens")
root.geometry("800x800")

app = ImageProcessor(root)

root.mainloop()
