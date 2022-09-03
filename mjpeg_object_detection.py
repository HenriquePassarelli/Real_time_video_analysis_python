import numpy as np
import cv2 as cv

LABELS_FILE = 'yolo/data/coco.names'
CONFIG_FILE = 'yolo/cfg/yolov3-tiny.cfg'
WEIGHTS_FILE = 'yolo/weights/yolov3-tiny.weights'
CONFIDENCE_THRESHOLD = 0.5

H = None
W = None

LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

net = cv.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)


def get_stream(file_path):
    capture = cv.VideoCapture(0)

    if not capture.isOpened():
        print("Cannot open camera")
        return

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    OutLayers = net.getUnconnectedOutLayers()
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in OutLayers]
    count = 0
    while True:
        ret, frame = capture.read()
        count += 1
        print('frame', count)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        H, W = frame.shape[:2]
        blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # initialize our lists
        boxes, confidences, classIDs = [], [], []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > CONFIDENCE_THRESHOLD:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        idxs = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
                               CONFIDENCE_THRESHOLD)

        thickness = 1

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[classIDs[i]]]

                cv.rectangle(frame, (x, y), (x + w, y + h),
                             color, thickness=thickness)
        cv.waitKey(1)
        # return the analyzed frame
        jpeg = cv.imencode('.jpg', frame)[1]
        return jpeg.tobytes()

    # do a bit of cleanup
    cv.destroyAllWindows()

    # release the file pointers
    print("[INFO] cleaning up...")
    capture.release()
