from image_preprocessor import GalactiClass_PreProcessing
from morphology_detector import GalactiClass_MorphologyDetector
from helpers import GalacticClass_Helpers
from common_imports import *
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import cv2

class Main:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Galaxy Morphology Detector")
        
        # initial window size
        self.root.geometry("1024x1024")  

        self.morphology_detector = GalactiClass_MorphologyDetector()
        self.setup_ui()

    def setup_ui(self):
        # Upload image button
        upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        upload_button.pack()

    def upload_image(self):
        filepaths = filedialog.askopenfilenames(filetypes=[("JPEG files", "*.jpeg"), ("PNG files", "*.png")])
        self.classify_image(filepaths)

    def classify_image(self, filepaths):
        num_files = len(filepaths)
        fig, axs = plt.subplots(1, num_files, figsize=(5 * num_files, 6), squeeze=False)
        axs = axs.flatten()

        for i, file_path in enumerate(filepaths):
            # Read and process image
            image = cv2.imread(file_path)
            confidences = self.morphology_detector.detect_morphology(image)

            # Determine the highest confidence
            galaxy_type = max(confidences, key=confidences.get)

            # Plot the result image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axs[i].imshow(image_rgb)
            
            # Display confidence score and galaxy type in the title
            axs[i].set_title(f"Galaxy: {i+1}\nElliptical: {confidences['elliptical']:.2f}\nSpiral: {confidences['spiral']:.2f}\nIrregular: {confidences['irregular']:.2f}\nType: {galaxy_type.capitalize()}", pad=20)  # Add padding for the title

        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    main = Main()
    main.run()