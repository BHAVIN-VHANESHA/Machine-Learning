import cv2
import numpy as np

def cartoonize_image(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a bilateral filter to reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=300, sigmaSpace=300)

    # Apply an edge-preserving filter to identify and enhance edges
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)

    # Combine the edges with the original image
    cartoon = cv2.bitwise_and(img, img, mask=edges)

    return cartoon

cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Apply cartoonization to the frame
    cartoon_frame = cartoonize_image(frame)

    # Display the original and cartoonized frames
    cv2.imshow('Original', frame)
    cv2.imshow('Cartoonized', cartoon_frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
