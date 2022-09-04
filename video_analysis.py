import numpy as np
import cv2

LABELS_FILE = 'yolo/data/coco.names'
CONFIG_FILE = 'yolo/cfg/yolov4-tiny.cfg'
WEIGHTS_FILE = 'yolo/weights/yolov4-tiny.weights'
CONFIDENCE_THRESHOLD = 0.3
LABELS = open(LABELS_FILE).read().strip().split("\n")

COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")


class VideoAnalysis:

    ln, h, w, net, ROIArea = None, None, None, None, None
    ROIColor = (255, 0, 0)

    def __init__(self):
        global ROIArea
        self.load_layers()
        ROIArea = []

    @staticmethod
    def load_layers():
        global net, ln
        net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
        ln = net.getLayerNames()
        OutLayers = net.getUnconnectedOutLayers()
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in OutLayers]

    def get_video(self, video_url=0):
        capture = cv2.VideoCapture(video_url)

        if not capture.isOpened():
            print("Cannot open camera")
            return capture.isOpened()

        return capture

    def check_points(self, ROIArea, point_start, point_end):
        points = [point_start, [point_start[0], point_end[1]],
                  point_end, [point_start[1], point_end[0]]]

        for area in ROIArea:
            for point in points:
                check = cv2.pointPolygonTest(
                    area, point, False)
                if check >= 0:
                    return True

    def read_stream(self, video_url):
        video = self.get_video(video_url)
        if not video:
            return

        while True:
            frame = video.read()[1]
            global h, w
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            boxes, confidences, classIDs = [], [], []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > CONFIDENCE_THRESHOLD:
                        box = detection[0:4] * np.array([w, h, w, h])
                        (centerX, centerY, width, height) = box.astype("int")

                        x_start = int(centerX - (width / 2))
                        y_start = int(centerY - (height / 2))

                        # update our list of bounding box coordinates, confidences and IDs
                        boxes.append(
                            [x_start, y_start, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
                                    CONFIDENCE_THRESHOLD)
            thickness = 1

            isClosed = True

            global ROIArea, ROIColor

            for area in ROIArea:
                area = area.reshape((-1, 1, 2))
                cv2.polylines(frame, [area],
                              isClosed, ROIColor, thickness)

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x_start, y_start) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    point_start = [x_start, y_start]
                    point_end = [x_start + w, y_start + h]

                    threshold = self.check_points(
                        ROIArea, point_start, point_end)

                    if threshold:
                        print('result: ', threshold)
                    else:
                        print('result: ', threshold)

                    color = [int(c) for c in COLORS[classIDs[i]]]

                    cv2.rectangle(frame, point_start, point_end,
                                  color, thickness=thickness)

            image = cv2.imencode('.jpg', frame)[1]
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image.tobytes() + b'\r\n\r\n')

    def set_ROIArea(self, area, color=ROIColor):
        global ROIArea, ROIColor

        ROIArea = [np.array(area, np.int32)]
        ROIColor = color


""" 
## Square pattern
# top-left, bottom-left, bottom-right, top-right
            area_1 = np.array([[100, 200], [100, 400], [800, 400], [800, 200]],
                              np.int32)
# line pattern
            area_2 = np.array([[150, 340], [800, 340]],
                              np.int32)
 """
