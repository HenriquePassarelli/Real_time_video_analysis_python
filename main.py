from flask import Flask, render_template, Response, request
import base64
import sys

from video_analysis import VideoAnalysis

app = Flask(__name__)

video_analysis = VideoAnalysis()

@app.route('/')
def index():
    return render_template('index.html')


# http://127.0.0.1:9000/stream?url=<base64 url>
@app.route('/stream')
def http_stream():
    userInput = request.args.get('url')
    url = base64.b64decode(userInput).decode('utf-8')
    print(url)

    return Response(video_analysis.read_stream(url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/ROI', methods=['POST'])
def set_ROI():
    content = request.json
    print('content', content)
    video_analysis.set_ROIArea(content)
    return Response()


if __name__ == '__main__':
    port = 5000 if len(sys.argv) <= 1 else sys.argv[1]
    app.run(host='0.0.0.0', debug=True, port=port)
