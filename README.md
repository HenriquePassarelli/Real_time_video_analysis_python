# Real_time_video_analysis_python

Using Python along with opencv to read a Mjpeg stream, do analysis and check if the detected object is inside a predefined zone .

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

### set predefined area
POST request
```
http://127.0.0.1:<port>/ROI
```

resquest body 

##### Square pattern
- top-left, bottom-left, bottom-right, top-right - ```[[100, 200], [100, 400], [800, 400], [800, 200]]```
##### Line pattern 
- ```[[150, 340], [800, 340]]```
