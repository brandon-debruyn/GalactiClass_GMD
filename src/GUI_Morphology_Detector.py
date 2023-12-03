import tkinter as tk
from tkinter import filedialog
from common_imports import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from morphology_detector import GalactiClass_MorphologyDetector


classifier = GalactiClass_MorphologyDetector()

def upload_image():
    filepaths = filedialog.askopenfilenames(filetypes=[("JPEG files", "*.jpeg"), ("PNG files", "*.png")])
    classify_image(filepaths)

def classify_image(filepaths):
    num_files = len(filepaths)
    fig, axs = plt.subplots(1, num_files, figsize=(5 * num_files, 6), squeeze=False)
    axs = axs.flatten()

    for i, file_path in enumerate(filepaths):
        # Read and process image
        image = cv2.imread(file_path)
        confidences = classifier.detect_morphology(image)

        # Determine the highest confidence
        galaxy_type = max(confidences, key=confidences.get)

        # Plot the result image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axs[i].imshow(image_rgb)
        
        # Display confidence score and galaxy type in the title
        axs[i].set_title(f"Galaxy: {i+1}\nElliptical: {confidences['elliptical']:.2f}\nSpiral: {confidences['spiral']:.2f}\nIrregular: {confidences['irregular']:.2f}\nType: {galaxy_type.capitalize()}", pad=20)  # Add padding for the title

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

# Create the root window
root = tk.Tk()
root.title("Galaxy Morphology Detector")

# Upload image button
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack()

# Run app
root.mainloop()

