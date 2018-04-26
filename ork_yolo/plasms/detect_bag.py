#!/usr/bin/env python
import ecto
import ecto_ros

from ecto_opencv import imgproc
from ecto_opencv.highgui import imshow, ImageReader
from ecto_image_pipeline.io.source import create_source
from ecto_image_pipeline.conversion import MatToPointCloudXYZRGB
from ecto_pcl import PointCloudT2PointCloud, CloudViewer, XYZRGB
from ecto.opts import run_plasm, cell_options, scheduler_options

import ecto_ros.ecto_sensor_msgs as ecto_sensor_msgs

import argparse
import sys
import os

import object_recognition_yolo.ecto_cells.ecto_yolo as ecto_yolo

parser = argparse.ArgumentParser(description='Test detection.')

bag = "/home/sam/rosbags/sigverse/stroll.bag"
ImageBagger = ecto_sensor_msgs.Bagger_Image
CameraInfoBagger = ecto_sensor_msgs.Bagger_CameraInfo
baggers = dict(
    image=ImageBagger(topic_name='/hsrb/head_rgbd_sensor/rgb/image_raw'),
    image_ci=CameraInfoBagger(topic_name='/hsrb/head_rgbd_sensor/rgb/camera_info'),
)

source = ecto_ros.BagReader(
    'Bag Reader',
    baggers=baggers,
    bag=bag,
    random_access=False,
)

image = ecto_ros.Image2Mat()
rgb = imgproc.cvtColor('bgr -> rgb', flag=imgproc.Conversion.BGR2RGB)
display = imshow(name='RGB', triggers=dict(save=ord('s')))

detector = ecto_yolo.Detector(
    data_config="/home/sam/OrkData/cfg/ork.dataset",
    network_config="/home/sam/OrkData/cfg/yolov3-ork.cfg",
    weights="/home/sam/OrkData/weights/yolov3-ork_34000.weights",
    detection_threshold=0.2,
)

# ecto options
scheduler_options(parser)
args = parser.parse_args()

plasm = ecto.Plasm()
plasm.connect(
    source['image'] >> image['image'],
    image['image'] >> rgb['image'],
    rgb['image'] >> detector['image']
)

ecto.view_plasm(plasm)

run_plasm(args, plasm, locals=vars())
