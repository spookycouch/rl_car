import cv2
import numpy as np

board_size = (8,5)
output_resolution = (1000,1600)
img_flat = np.zeros(np.product(board_size), dtype=np.float32)
img_flat[::2] = 1.0
img = img_flat.reshape(board_size)

img = cv2.resize(img, output_resolution, interpolation=cv2.INTER_NEAREST)
img = img * 255
cv2.imwrite("chessboard.png", img)