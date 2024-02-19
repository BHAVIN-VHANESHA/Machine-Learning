import cv2
import numpy as np
import matplotlib.pyplot as plt


# We will learn how to peform the following annotations to images.
# 1. Draw lines
# 2. Draw circles
# 3. Draw rectangles
# 4. Add text
image = cv2.imread(r"D:\\ML\\OpenCV\\Dataset\\apollo11.png", cv2.IMREAD_COLOR)
plt.imshow(image[:, :, ::-1])
plt.show()

# Drawing a Line
imageLine = image.copy()

# The line starts from (200,100) and ends at (400,100)
# The color of the line is YELLOW (Recall that OpenCV uses BGR format)
# Thickness of line is 5px
# Linetype is cv2.LINE_AA

cv2.line(imageLine, (100, 200), (300, 200), (0, 255, 255), thickness=5, lineType=cv2.LINE_AA)

# Display the image
plt.imshow(imageLine[:, :, ::-1])
plt.show()

# Drawing a Circle
imageCircle = image.copy()
cv2.circle(imageCircle, (550, 325), 50, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA)

# Display the image
plt.imshow(imageCircle[:, :, ::-1])
plt.show()

# Drawing a Rectangle
# Draw a rectangle (thickness is a positive integer)
imageRectangle = image.copy()
cv2.rectangle(imageRectangle, (300, 30), (440, 400), (255, 0, 255), thickness=5, lineType=cv2.LINE_8)

# Display the image
plt.imshow(imageRectangle[:, :, ::-1])
plt.show()

# Adding Text
imageText = image.copy()
text = "Apollo 11 Saturn V Launch, July 16, 1969"
fontScale = 2
fontFace = cv2.FONT_HERSHEY_PLAIN
fontColor = (0, 255, 0)
fontThickness = 2

cv2.putText(imageText, text, (50, 400), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA)

# Display the image
plt.imshow(imageText[:, :, ::-1])
plt.show()
