from flask import Flask, render_template, Response, request
from mjpeg_object_detection import get_stream
import numpy as np
import base64
import sys
import cv2

app = Flask(__name__)

LABELS_FILE = 'yolo/data/coco.names'
CONFIG_FILE = 'yolo/cfg/yolov4-tiny.cfg'
WEIGHTS_FILE = 'yolo/weights/yolov4-tiny.weights'
CONFIDENCE_THRESHOLD = 0.3
LABELS = open(LABELS_FILE).read().strip().split("\n")

COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

ln, h, w, net = None, None, None, None


def get_video(video_url=0):
    capture = cv2.VideoCapture(video_url)

    if not capture.isOpened():
        print("Cannot open camera")
        return capture.isOpened()

    return capture


def load_layers():
    global net, ln
    net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
    ln = net.getLayerNames()
    OutLayers = net.getUnconnectedOutLayers()
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in OutLayers]


def read_stream(video_url):
    video = get_video(video_url)
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

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
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

                cv2.rectangle(frame, (x, y), (x + w, y + h),
                              color, thickness=thickness)
        # show the output image
        # cv2.imshow("frame", frame)
        image = cv2.imencode('.jpg', frame)[1]
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image.tobytes() + b'\r\n\r\n')
         
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

#######################################################

# micro API

@app.route('/')
def index():
    return render_template('index.html')


def gen(url):
    frame = get_stream(url)
    while True:
        print(frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# http://127.0.0.1:9000/stream?url=<base64 url>
@app.route('/stream')
def http_stream():
    userInput = request.args.get('url')
    url = base64.b64decode(userInput).decode('utf-8')
    print(url)

    return Response(read_stream(url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def test():

    return read_stream()


@app.route('/test')
def test_stream():
    """ userInput = request.args.get('url')
    url = base64.b64decode(userInput).decode('utf-8')
    print(url) """

    return Response(test(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    load_layers()
    port = 5000 if len(sys.argv) <= 1 else sys.argv[1]
    app.run(host='0.0.0.0', debug=True, port=port)
