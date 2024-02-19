import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image

# print(cv2.__version__)

# Display Image Directly
# Image(filename=r"D:\ML\OpenCV\bugatti.jpg")

# Set threshold to infinity to avoid truncation
np.set_printoptions(threshold=np.inf)


''' The function cv2.imread() is used to read an image.
img = cv2.imread(r"D:\\ML\\OpenCV\\chessboard.jpeg")
# Print the image data (pixel values), element of a 2D numpy array. Each pixel value is 8-bits [0,255]
print(img)

# Display Image attributes
# print the size  of image
print("Image size (H, W) is:", img.shape)
# print data-type of image
print("Data type of image is:", img.dtype)

# The function cv2.imshow() is used to display an image in a window.
# cv2.imshow('original image', img)
# '''


''' Converting image into gray scale image
img_grayscale = cv2.imread(r"D:\\ML\\OpenCV\\chessboard.jpeg", 0)  # cv2.IMREAD_GRAYSCALE
cv2.imshow('grayscale image', img_grayscale)
print(img_grayscale)
# '''

''' Display Images using Matplotlib
plt.imshow(img_grayscale)
plt.show()
# Even though the image was read in as a gray scale image, it won't necessarily display in gray scale when using
# imshow(). matplotlib uses different color maps, and it's possible that the gray scale color map is not set.
# Set color map to gray scale for proper rendering.
plt.imshow(img_grayscale, cmap='gray')
plt.show()
# '''


''' Working with colour image
img_color = cv2.imread(r"D:\\ML\\OpenCV\\coca-cola.png", 1)  # cv2.IMREAD_COLOR
cv2.imshow('color image', img_color)
# print(img_color)
print("Image size (H, W, C) is:", img_color.shape)
print("Data type of image is:", img_color.dtype)
plt.imshow(img_color)
plt.show()
# The color displayed above is different from the actual image. This is because matplotlib expects the image in RGB
# format whereas OpenCV stores images in BGR format. Thus, for correct display, we need to reverse the channels of
# the image.
coke_img_channels_reversed = img_color[:, :, ::-1]
plt.imshow(coke_img_channels_reversed)
plt.show()
# '''

# img_unchanged = cv2.imread(r"D:\\ML\\OpenCV\\bugatti.jpg", -1)  # cv2.IMREAD_UNCHANGED
# cv2.imshow('unchanged image', img_unchanged)
# print(img_unchanged)


''' Splitting and Merging Color Channels
img_bgr = cv2.imread(r"D:\\ML\\OpenCV\\input-image-for-demo-throughout-1024x682.jpg", cv2.IMREAD_COLOR)
b, g, r = cv2.split(img_bgr)
plt.figure(figsize=[20, 5])
plt.subplot(141)
plt.imshow(r, cmap="gray")
plt.title("Red Channel")
plt.subplot(142)
plt.imshow(g, cmap="gray")
plt.title("Green Channel")
plt.subplot(143)
plt.imshow(b, cmap="gray")
plt.title("Blue Channel")

# Merge the individual channels into a BGR image
imgMerged = cv2.merge((b, g, r))
# Show the merged output
plt.subplot(144)
plt.imshow(imgMerged[:, :, ::-1])
plt.title("Merged Output")
plt.show()
# '''

# Converting to different Color Spaces
# ''' 1. BGR to RGB
bgr_rgb = cv2.imread(r"/OpenCV/Dataset/input-image-for-demo-throughout-1024x682.jpg", cv2.COLOR_BGR2RGB)
plt.imshow(bgr_rgb)
plt.show()
# '''

# ''' 2. HSV color space
img_hsv = cv2.cvtColor(bgr_rgb, cv2.COLOR_BGR2HSV)

# Split the image into the B,G,R components
h, s, v = cv2.split(img_hsv)

# Show the channels
plt.figure(figsize=[20,5])
plt.subplot(141)
plt.imshow(h, cmap="gray")
plt.title("H Channel")

plt.subplot(142)
plt.imshow(s, cmap="gray")
plt.title("S Channel")

plt.subplot(143)
plt.imshow(v, cmap="gray")
plt.title("V Channel")

plt.subplot(144)
plt.imshow(img_hsv)
plt.title("Original")
plt.show()
# '''

# ''' Modifying individual Channel
h_new = h + 10
img_merged = cv2.merge((h_new, s, v))
img_rgb = cv2.cvtColor(img_merged, cv2.COLOR_HSV2RGB)

# Show the channels
plt.figure(figsize=[20,5])
plt.subplot(141)
plt.imshow(h, cmap="gray")
plt.title("H Channel")

plt.subplot(142)
plt.imshow(s, cmap="gray")
plt.title("S Channel")

plt.subplot(143)
plt.imshow(v, cmap="gray")
plt.title("V Channel")

plt.subplot(144)
plt.imshow(img_rgb)
plt.title("Original")
plt.show()
# '''


# waitKey() waits for a key press to close the window and 0 specifies indefinite loop
cv2.waitKey(0)

# cv2.destroyAllWindows() simply destroys all the windows we created.
cv2.destroyAllWindows()

# The function cv2.imwrite() is used to save an image.
cv2.imwrite('Dataset/sprinkling_rose.jpg', img_rgb)

# read the image as Color
img_sr_bgr = cv2.imread("Dataset/sprinkling_rose.jpg", cv2.IMREAD_COLOR)
print("img_NZ_bgr shape (H, W, C) is:", img_sr_bgr.shape)

# read the image as Grayscale
img_sr_gry = cv2.imread("Dataset/sprinkling_rose.jpg", cv2.IMREAD_GRAYSCALE)
print("img_NZ_gry shape (H, W) is:", img_sr_gry.shape)
