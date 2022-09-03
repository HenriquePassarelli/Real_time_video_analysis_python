from flask import Flask, render_template, Response, request
from mjpeg_object_detection import get_stream
import base64
import sys

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(url):
    while True:
        frame = get_stream(url)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# http://127.0.0.1:9000/stream?url=<base64 url>
@app.route('/stream')
def http_stream():
    userInput = request.args.get('url')
    url = base64.b64decode(userInput).decode('utf-8')
    print(url)

    return Response(gen(url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    port = 5000 if len(sys.argv) <= 1 else sys.argv[1]
    app.run(host='0.0.0.0', debug=True, port=port)
