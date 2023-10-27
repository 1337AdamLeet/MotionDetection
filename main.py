import cv2
import time


'''
this code demonstrates the cv2 libary being used to calculate relative speed, fps and motion detection
with python. its optimized quite well but cant handle large videos that have multiple motion around.
it detects humans verry well and is able to log any errors in the console.
this source is free to use just credit me please :)
- Adam1337
'''

video_capture = cv2.VideoCapture('video.mp4')
fps_start_time = time.time()
fps_frames = 0

fgbg = cv2.createBackgroundSubtractorMOG2()

prev_frame = None
object_detected = False

while True:
    ret, frame = video_capture.read()

    if not ret:
        print("Unable to read frame. Exiting...")
        break

    fgmask = fgbg.apply(frame)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    object_detected = False

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            object_detected = True
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Calculate relative speed based on pixel movement
    if prev_frame is not None and object_detected:
        diff = cv2.absdiff(prev_frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        movement_pixels = cv2.countNonZero(diff)
        relative_speed = movement_pixels  # You can experiment with scaling factor for better results

        cv2.putText(frame, f'Relative Speed: {relative_speed}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fps_frames += 1
    fps_end_time = time.time() - fps_start_time
    fps = fps_frames / fps_end_time
    cv2.putText(frame, f'FPS: {int(fps)}', (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                cv2.LINE_AA)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
