import cv2

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    num_images = 0
    print("press S to save image. only works when a chessboard is detected. collect 30.")

    while 1:
        
        _, frame = cam.read()
        
        img = frame.copy()
        ret, corners = cv2.findChessboardCorners(img, (4,5), None)
        if ret:
            cv2.drawChessboardCorners(img, (4,5), corners, ret)
        cv2.imshow("img", img)
        key = cv2.waitKey(10)

        if ret and key in (83, 115):
            cv2.imwrite(f"img_{num_images}.jpg", frame)
            print(f"image {num_images} get!")
            num_images += 1
        if num_images >= 30:
            break