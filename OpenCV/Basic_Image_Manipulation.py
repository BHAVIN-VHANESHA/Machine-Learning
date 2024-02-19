import cv2
import numpy as np
import matplotlib.pyplot as plt


# Set threshold to infinity to avoid truncation
np.set_printoptions(threshold=np.inf)


''' 
img_gray = cv2.imread(r"D:\\ML\\OpenCV\\chessboard.jpeg", 0)
print(img_gray)
plt.imshow(img_gray, cmap='gray')
plt.show()
# '''


''' Accessing Individual Pixels
# print the first pixel of the first black box
print(img_gray[0, 0])
# print the first white pixel to the right of the first black box
print(img_gray[0, 30])

# Modifying Image Pixels
img_modify = img_gray.copy()
# img_modify[2, 2] = 200
# img_modify[2, 3] = 200
# img_modify[3, 2] = 200
# img_modify[3, 3] = 200

# Same as above
img_modify[2:10, 2:10] = 200

print(img_modify)
plt.imshow(img_modify, cmap="gray")
plt.show()
# '''


''' Cropping Images
img_crop = cv2.imread(r"D:\\ML\\OpenCV\\input-image-for-demo-throughout-1024x682.jpg", 1)
plt.imshow(img_crop)
# plt.show()

img_rgb = img_crop[:, :, ::-1]
plt.imshow(img_rgb)
# plt.show()

# Crop out the middle region of the image
cropped_region = img_rgb[150:600, 300:700]
plt.imshow(cropped_region)
# plt.show()


# Resizing Images
# Using the function resize()
# Method 1: Specifying Scaling Factor using fx and fy
resize_img = cv2.resize(cropped_region, None, fx=2, fy=2)
plt.imshow(resize_img)
# plt.show()

# Method 2: Specifying exact size of the output image
resized_cropped_region = cv2.resize(cropped_region, dsize=[100, 200], interpolation=cv2.INTER_AREA)
plt.imshow(resized_cropped_region)
# plt.show()

# Resize while maintaining aspect ratio
desired_width = 100
aspect_ratio = desired_width / cropped_region.shape[1]
desired_height = int(cropped_region.shape[0] * aspect_ratio)
dim = (desired_width, desired_height)
# Resize image
resized_cropped_region = cv2.resize(cropped_region, dsize=dim, interpolation=cv2.INTER_AREA)
plt.imshow(resized_cropped_region)
# plt.show()

# Let's actually show the (cropped) resized image.
# Swap channel order
resized_cropped_region_2x = resize_img[:, :, ::-1]

# Save resized image to disk
cv2.imwrite("resized_cropped_region_2x.png", resized_cropped_region_2x)

# Display the cropped and resized image
img_saved = cv2.imread(r"D:\\ML\\OpenCV\\resized_cropped_region_2x.png")
cv2.imshow('resized-cropped image', img_saved)

# Swap channel order
cropped = cropped_region[:, :, ::-1]

# Save cropped 'region'
cv2.imwrite("cropped_region.png", cropped)

# Display the cropped and resized image
cv2.imshow(r"D:\\ML\\OpenCV\\cropped_region.png", cropped_region)
# '''


''' Flipping Images
# a flag to specify how to flip the array; 0 means flipping around the x-axis and positive value
# (for example, 1) means flipping around y-axis. Negative value (for example, -1) means flipping around both axes.
img_flip = cv2.imread(r"D:\\ML\\OpenCV\\bugatti.jpg")[:, :, ::-1]
img_rgb_flipped_horz = cv2.flip(img_flip, 1)
img_rgb_flipped_vert = cv2.flip(img_flip, 0)
img_rgb_flipped_both = cv2.flip(img_flip, -1)
# Show the images
plt.figure(figsize=(18, 5))
plt.subplot(141)
plt.imshow(img_rgb_flipped_horz)
plt.title("Horizontal Flip")

plt.subplot(142)
plt.imshow(img_rgb_flipped_vert)
plt.title("Vertical Flip")

plt.subplot(143)
plt.imshow(img_rgb_flipped_both)
plt.title("Both Flipped")

plt.subplot(144)
plt.imshow(img_flip)
plt.title("Original")
plt.show()
# '''
