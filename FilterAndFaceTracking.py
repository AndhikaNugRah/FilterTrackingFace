import sys
import cv2
import time

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = "Integrated Face Tracking & Filter Test"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None

#network architecture that contains weight
net = cv2.dnn.readNetFromCaffe(r"C:\Users\dhika\Downloads\opencv_bootcamp_assets_12\deploy.prototxt", r"C:\Users\dhika\Downloads\opencv_bootcamp_assets_12\res10_300x300_ssd_iter_140000_fp16.caffemodel")
# Model parameters
in_width = 312
in_height = 312
mean = [104, 117, 123]
conf_threshold = 0.7

def detect_faces(net, source, in_width, in_height, mean, conf_threshold, win_name):
    has_frame, frame = source.read()
    if not has_frame:
        return False

    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False) 
    # Run a model
    net.setInput(blob)
    detections = net.forward()

    # Annotating the frame 
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 255))
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(
                frame,
                (x_left_bottom, y_left_bottom - label_size[1]),
                (x_left_bottom + label_size[0], y_left_bottom + base_line),
                (0, 0, 0),
                cv2.FILLED,
            )
            cv2.putText(frame, label, (x_left_bottom, y_left_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))

    t, _ = net.getPerfProfile() 
    label = "Face Tracking Mode Active-Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
    cv2.imshow(win_name, frame)
    return True

feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=15, blockSize=9)
feature_flag = detect_faces_flag = blur_flag = canny_flag = False


while True:
    if cv2.waitKey(1) == 27:
        break
    key = cv2.waitKey(1) & 0xFF
    if key == ord('f'):
        feature_flag = True
        detect_faces_flag = blur_flag = canny_flag = False
    elif key == ord('p'):
        detect_faces_flag = True
        feature_flag = blur_flag = canny_flag = False
    elif key == ord('b'):
        blur_flag = True
        detect_faces_flag = feature_flag = canny_flag = False
    elif key == ord('c'):
        canny_flag = True
        detect_faces_flag = feature_flag = blur_flag = False
    elif key == ord('d'):
        feature_flag = detect_faces_flag = blur_flag = canny_flag = False

        cv2.putText(frame, "Press B - for Blur Mode", (410, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
        cv2.putText(frame, "Press F - for Corner Mode", (410, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
        cv2.putText(frame, "Press P - for Face Tracking", (410, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
        cv2.putText(frame, "Press D - for Default Mode", (410, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))

    has_frame, frame = source.read()
    if not has_frame:
        break
    
    if feature_flag:
        frame = cv2.flip(frame,1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        
        mode_text = "Corner Mode Active"
        (text_width, text_height) = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        x = 0
        y = 0
        w = text_width + 5
        h = text_height + 5 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
        cv2.putText(frame, mode_text, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.putText(frame, "Press B - for Blur Mode", (410, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
        cv2.putText(frame, "Press C - for Canny Mode", (410, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120))
        cv2.putText(frame, "Press P - for Face Tracking", (410, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 155))
        cv2.putText(frame, "Press D - for Default Mode", (410, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 155, 255))

        if corners is not None:
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), 1)

    if blur_flag:

        frame = cv2.flip(frame,1)
        frame = cv2.blur(frame, (5, 5))
        mode_text = "Blur Mode Active"
        (text_width, text_height) = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        x = 0
        y = 0
        w = text_width + 5
        h = text_height + 5 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), -1)
        cv2.putText(frame, mode_text, (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255))
        cv2.putText(frame, "Press C - for Canny Mode", (410, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120))
        cv2.putText(frame, "Press F - for Corner Mode", (410, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
        cv2.putText(frame, "Press P - for Face Tracking", (410, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 155))
        cv2.putText(frame, "Press D - for Default Mode", (410, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 155, 255))

    if canny_flag:
        frame = cv2.flip(frame,1)
        frame = cv2.Canny(frame, 100, 200)

    if detect_faces_flag:
        frame = cv2.flip(frame,1)
        detect_faces(net, source, in_width, in_height, mean, conf_threshold, win_name)
    
    else: 
        cv2.imshow(win_name, frame)
