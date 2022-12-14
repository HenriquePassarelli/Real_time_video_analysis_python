from flask import Flask, render_template, Response, request
from waitress import serve
import argparse
import base64

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
    if not url.strip():
        print('hi')
        return Response('Missing parameters, eg. /stream?url=base64 url', status=400)
    print(url)
    return Response(video_analysis.gen_read_stream(url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/ROI', methods=['POST'])
def set_ROI():
    content = request.json
    if not content:
        return Response(status=400)
    print('content', content)
    video_analysis.set_ROIArea(content)
    return Response(status=200)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Camera Server")
    parser.add_argument("--port",
                        type=int,
                        default=8000,
                        help="Run on the given port")
    args: argparse.Namespace = parser.parse_args()
    serve(app, host="0.0.0.0", port=args.port)
    # app.run(host='0.0.0.0', debug=True, port=args.port)
