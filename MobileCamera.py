import cv2

url = "http://192.168.1.101:8080/video"  # Corrected IP & video feed URL

cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    height, width, _ = frame.shape  # Get original dimensions

    # Define the cropping region (center 640x640)
    crop_size = 800
    start_x = max((width - crop_size) // 2, 0)
    start_y = max((height - crop_size) // 2, 0)
    end_x = start_x + crop_size
    end_y = start_y + crop_size

    # Ensure cropping does not exceed frame boundaries
    cropped_frame = frame[start_y:end_y, start_x:end_x]

    cv2.imshow("IP Webcam Feed (Cropped)", cropped_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
