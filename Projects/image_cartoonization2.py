import cv2
import tkinter as tk
from tkinter import PhotoImage, filedialog
from PIL import ImageDraw, Image, ImageTk


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera App")

        self.detect_human_btn = tk.Button(root, text="Detect Human", command=self.detect_human)
        self.detect_human_btn.pack()

        self.open_cartoonized_btn = tk.Button(root, text="Open Cartoonized Image", command=self.open_cartoonized)
        self.open_cartoonized_btn.pack()
        self.open_cartoonized_btn["state"] = "disabled"

        self.video_source = 0  # You may need to change this to the appropriate camera source (e.g., 1 for an external camera)
        self.vid = cv2.VideoCapture(self.video_source)

        self.canvas = tk.Canvas(root, width=self.vid.get(3), height=self.vid.get(4))
        self.canvas.pack()

        self.update()
        self.faces = []

    def capture_photo(self):
        ret, frame = self.vid.read()
        if ret:
            cv2.imwrite("captured_photo.jpg", frame)
            print("Photo captured as 'captured_photo.jpg'")
            self.open_cartoonized_btn["state"] = "disabled"
        else:
            print("Failed to capture photo")

    def detect_human(self):
        self.capture_photo()  # Capture a fresh photo
        self.process_image("captured_photo.jpg")

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        if len(self.faces) > 0:
            image_pillow = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image_pillow)
            for (x, y, w, h) in self.faces:
                draw.rectangle([x, y, x + w, y + h], outline="green", width=2)
            image_pillow.show()
            self.open_cartoonized_btn["state"] = "active"

    def cartoonize_image(self, image_path):
        image = cv2.imread(image_path)

        cartoon_image = cv2.stylization(image, sigma_s=90, sigma_r=0.35)

        cv2.imwrite("cartoonized_photo.jpg", cartoon_image)
        print("Cartoonized photo saved as 'cartoonized_photo.jpg'")

    def open_cartoonized(self):
        self.cartoonize_image("captured_photo.jpg")
        image_pillow = Image.open("cartoonized_photo.jpg")
        image_pillow.show()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.root.after(10, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


root = tk.Tk()
app = CameraApp(root)
root.mainloop()
