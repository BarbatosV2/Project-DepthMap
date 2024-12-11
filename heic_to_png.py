from PIL import Image
import pillow_heif

# Load HEIC image
heif_file = pillow_heif.read_heif("right_image.heic")
image = Image.frombytes(
    heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode
)

# Save as PNG
image.save("right_image.png", "PNG")

# Load the converted image in OpenCV
import cv2
#img = cv2.imread("right_image.png")
#cv2.imshow("Image", img)
cv2.waitKey(0)
