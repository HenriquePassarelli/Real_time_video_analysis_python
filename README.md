# Real_time_video_analysis_python

Using Python along with opencv to read a Mjpeg stream 

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

### run script

```
python3 main.py <optional port>   
```
### run analytics

```
http://127.0.0.1:<port>/stream?url=<base64 video url>
```
