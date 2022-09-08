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
        self.__load_layers()
        ROIArea = []

    @staticmethod
    def __load_layers():
        global net, ln
        net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
        ln = net.getLayerNames()
        OutLayers = net.getUnconnectedOutLayers()
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in OutLayers]

    def __get_video(self, video_url=0):
        capture = cv2.VideoCapture(video_url)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if not capture.isOpened():
            print("Cannot open camera")
            return capture.isOpened()

        return capture

    # segments must be pass in the format [(x0,y0),(x1,y1)]
    def __intersects(self, segment1, segment2):
        dx0 = segment1[1][0]-segment1[0][0]
        dx1 = segment2[1][0]-segment2[0][0]
        dy0 = segment1[1][1]-segment1[0][1]
        dy1 = segment2[1][1]-segment2[0][1]
        p0 = dy1*(segment2[1][0]-segment1[0][0]) - \
            dx1*(segment2[1][1]-segment1[0][1])
        p1 = dy1*(segment2[1][0]-segment1[1][0]) - \
            dx1*(segment2[1][1]-segment1[1][1])
        p2 = dy0*(segment1[1][0]-segment2[0][0]) - \
            dx0*(segment1[1][1]-segment2[0][1])
        p3 = dy0*(segment1[1][0]-segment2[1][0]) - \
            dx0*(segment1[1][1]-segment2[1][1])

        return (p0*p1 <= 0) & (p2*p3 <= 0)

    def __check_points(self, ROIArea, point_start, point_end):
        points = [point_start, [point_start[0], point_end[1]],
                  point_end, [point_start[1], point_end[0]]]
        check_distance = False

        for area in ROIArea:
            if len(area) <= 2:
                lines = [[points[0], points[1]], [points[2], points[3]]]
                for line in lines:
                    return self.__intersects(area, line)
            for point in points:
                check = cv2.pointPolygonTest(
                    area, point, check_distance)
                if check >= 0:
                    return True

        return False

    # start the stream and analysis
    def gen_read_stream(self, video_url):
        video = self.__get_video(video_url)
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

            # loop over each of the layer outputs
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter out weak predictions
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

            # filter overlapping boxes
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

                    threshold = self.__check_points(
                        ROIArea, point_start, point_end)

                    if threshold:
                        print('warn: ', threshold)
                    else:
                        print('warn: ', threshold)

                    # print(LABELS[classIDs[i]])
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    # draw a box
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
