# Real Time Face Detection, Gender Classification, Age Classification

* install requirements.txt

<pre>
pip install -r requirements.txt
</pre>

* Run on ubuntu using webcam

<pre>
python3 detect.py --source 0
</pre>
---
## Using for Customer analysis

### Model:

- Face Detection: YOLOv5n-0.5 [YOLOv5-face](https://github.com/deepcam-cn/yolov5-face)
- Gender Classification: YOLOv5n-cls [YOLOv5](https://github.com/ultralytics/yolov5)
- Age Classification: YOLOv5n-cls [YOLOv5](https://github.com/ultralytics/yolov5)

### Dataset:

- Gender: GenderOcclusionDta (with mask and non-mask)
- Age: Megaage asian

### Class
- Gender: 0 - female, 1 - male
- Age: 0 - [0-10), 1 - [10-20), 2 - [20-30), 3 - [30-40), 4 - [40-50),
5 - [50-60), 6 - [60-70), 7 - [70-80), 8 - [80-90)
