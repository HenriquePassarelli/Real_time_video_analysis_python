# Real_time_video_analysis_python

Using Python along with opencv to read a Mjpeg stream, and do analysis.

### Install python

```
sudo apt install python3

sudo apt install python3-pip
```
### Install dependencies

```
pip install numpy opencv-python
pip install Flask
```

### adding models files

inside the yolo directory add two folders, one for the weights, and another for config. Reference right below.

```
CONFIG_FILE = 'yolo/cfg/yolov4-tiny.cfg'
WEIGHTS_FILE = 'yolo/weights/yolov4-tiny.weights'
```

### run script

```
python3 main.py <optional port>   
```
### run analytics

```
http://127.0.0.1:<port>/stream?url=<base64 video url>
```
