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
        self.canvas_hist = None
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
        self.zoom_level = 1.0
        self.img_width = None
        self.img_height = None
        self.inicial_menu()

    def inicial_menu(self):
        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        label = tk.Label(frame, text="Informe o número do paciente:")
        label.pack(side=tk.LEFT, padx=5)

        self.entry_n = tk.Entry(frame)
        self.entry_n.pack(side=tk.LEFT, padx=5)

        btn_load = tk.Button(frame, text="visualizar paciente", command=self.setup_menu)
        btn_load.pack(side=tk.LEFT, padx=5)

        self.canvas_img = tk.Canvas(self.root)
        self.canvas_img.pack(pady=20)
        self.create_csv()
        self.canvas_img.bind("<ButtonPress-1>", lambda event: self.select_roi(event))
        self.canvas_img.bind("<ButtonPress-1>", lambda event: self.select_roi(event))

    def visualization_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        display_frame = tk.Frame(self.root)
        display_frame.pack(pady=20)

        self.canvas_img = tk.Canvas(
            display_frame, width=400, height=400
        )  # Adjust size based on your image
        self.canvas_img.grid(
            row=0, column=0, padx=10, pady=10
        )  # Use grid for better layout control

        self.canvas_hist = tk.Canvas(
            display_frame, width=400, height=400
        )  # Adjust size for the histogram
        self.canvas_hist.grid(row=0, column=1, padx=10, pady=10)

        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        self.update_header_roi_number()

        # Create buttons for image navigation
        btn_prev = tk.Button(frame, text="Previous Image", command=self.prev_image)
        btn_prev.pack(side=tk.LEFT, padx=5)

        btn_next = tk.Button(frame, text="Next Image", command=self.next_image)
        btn_next.pack(side=tk.LEFT, padx=5)

        self.canvas_img.bind("<MouseWheel>", self.on_mouse_wheel)

        # Display the first image for the selected patient
        self.display_image(self.images[0][self.patient_number][self.index_img])
        self.display_histogram(self.images[0][self.patient_number][self.index_img])

    def on_mouse_wheel(self, event):
        if event.delta > 0:  # Scroll up
            self.zoom_in()
        else:  # Scroll down
            self.zoom_out()

    def main_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        btn_img_visualization = tk.Button(
            frame, text="Visualization menu", command=self.visualization_menu
        )
        btn_img_visualization.pack(side=tk.LEFT, padx=5)

        btn_cut = tk.Button(frame, text="Cut Roi", command=self.cut_roi_menu)
        btn_cut.pack(side=tk.LEFT, padx=5)

        btn_visualize_roi = tk.Button(
            frame, text="Cut Roi", command=self.visualize_roi_menu
        )
        btn_visualize_roi.pack(side=tk.LEFT, padx=5)

        btn_compute_glcm = tk.Button(
            frame, text="Compute GLCM", command=self.compute_glcm
        )
        btn_compute_glcm.pack(side=tk.LEFT, padx=5)

        btn_roi_caracterization = tk.Button(
            frame, text="Caracterize ROI", command=self.caracterize_roi
        )
        btn_roi_caracterization.pack(side=tk.LEFT, padx=5)

        btn_classificate_img = tk.Button(
            frame, text="Classificate image", command=self.classificate_img
        )
        btn_classificate_img.pack(side=tk.LEFT, padx=5)

    def classificate_img(self):
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

    def compute_glcm(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.root)
        frame.pack(pady=20)
        label = tk.Label(frame, text="TODO")
        label.pack(side=tk.LEFT, padx=5)

    def visualize_roi_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.root)
        frame.pack(pady=20)
        label = tk.Label(frame, text="TODO")
        label.pack(side=tk.LEFT, padx=5)

    def cut_roi_menu(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.root)
        frame.pack(pady=20)
        label = tk.Label(frame, text="TODO")
        label.pack(side=tk.LEFT, padx=5)

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
                    f"Patient number {self.patient_number} is out of bounds. Max patient number: {num_patients - 1}"
                )

            num_images_per_patient = len(self.images[0][self.patient_number])
            self.index_img = 0

            if self.index_img >= num_images_per_patient:
                raise IndexError(f"Patient have no Exams!!")

            self.main_menu()
        except IndexError as e:
            messagebox.showerror("Patient not found!!", str(e))
        except ValueError:
            messagebox.showerror(
                "Invalid input. Patient number must be an integer.",
                "Invalid input. Patient number must be an integer.",
            )

    def display_image(self, image):
        self.img = Image.fromarray(image)
        img_tk = ImageTk.PhotoImage(self.img)

        # Store original dimensions for zooming
        if self.img_width is None or self.img_height is None:
            self.img_width, self.img_height = self.img.size

        # Scale the image according to the zoom level
        new_size = (
            int(self.img_width * self.zoom_level),
            int(self.img_height * self.zoom_level),
        )
        scaled_img = self.img.resize(new_size, Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(scaled_img)

        self.canvas_img.config(width=scaled_img.width, height=scaled_img.height)
        self.canvas_img.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas_img.image = img_tk

    def zoom_in(self):
        self.zoom_level *= 1.2  # Increase zoom level
        self.display_image(
            self.images[0][self.patient_number][self.index_img]
        )  # Redraw the image with updated zoom

    def zoom_out(self):
        self.zoom_level /= 1.2  # Decrease zoom level
        self.display_image(self.images[0][self.patient_number][self.index_img])

    def display_histogram(self, image):
        if len(image.shape) != 2:
            raise ValueError("Expected a 2D array for grayscale image data.")

        image = Image.fromarray(image)
        # Calculate histogram
        histogram, bin_edges = np.histogram(image, bins=256, range=(0, 255))

        # Create a matplotlib figure to display the histogram
        fig, ax_hist = plt.subplots(figsize=(5, 4))

        ax_hist.clear()
        ax_hist.plot(bin_edges[0:-1], histogram, color="black")
        ax_hist.set_title("Histogram")
        ax_hist.set_xlim(0, 255)
        ax_hist.set_ylim(0, 4000)
        ax_hist.set_xlabel("Pixel value")
        ax_hist.set_ylabel("Frequency")

        # Save the histogram plot to a memory buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        # Load the image from the buffer and convert it for Tkinter
        hist_img = Image.open(buf)
        hist_tk = ImageTk.PhotoImage(hist_img)

        # Display the histogram in the Tkinter canvas
        self.canvas_hist.config(width=hist_img.width, height=hist_img.height)
        self.canvas_hist.create_image(0, 0, anchor=tk.NW, image=hist_tk)
        self.canvas_hist.image = hist_tk  # Keep a reference to avoid garbage collection

        buf.close()  # Close the buffer

        plt.close(fig)

    def prev_image(self):
        if self.index_img > 0:
            self.index_img -= 1
            self.display_image(self.images[0][self.patient_number][self.index_img])
            self.display_histogram(self.images[0][self.patient_number][self.index_img])
            self.update_header_roi_number()
        else:
            messagebox.showinfo("End of Images", "This is the first image.")

    def next_image(self):
        num_images_per_patient = len(self.images[0][self.patient_number])
        if self.index_img < num_images_per_patient - 1:
            self.index_img += 1
            self.display_image(self.images[0][self.patient_number][self.index_img])
            self.display_histogram(self.images[0][self.patient_number][self.index_img])
            self.update_header_roi_number()
        else:
            messagebox.showinfo("End of Images", "This is the last image.")

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
        print("liver_grayscale_mean", liver_grayscale_mean)
        print("kidney_grayscale_mean", kidney_grayscale_mean)
        self.HI_index = liver_grayscale_mean / kidney_grayscale_mean
        messagebox.showinfo("Índice HI", f"O índice HI é: {self.HI_index:.2f}")


root = tk.Tk()
root.title("Visualizador de Imagens")
root.geometry("1600x800")

app = ImageProcessor(root)

root.mainloop()
