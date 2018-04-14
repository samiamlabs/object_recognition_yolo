#!/usr/bin/env python
import ecto
import ecto_ros

from ecto_opencv import imgproc
from ecto_opencv.highgui import ImageReader
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

# this will read all images in the path
path = '/home/sam/Code/vision/VOCdevkit/VOC2012/JPEGImages'
source = ImageReader(path=os.path.expanduser(path))

detector = ecto_yolo.Detector(
    data_config="/home/sam/OrkData/cfg/voc.dataset",
    network_config="/home/sam/OrkData/cfg/tiny-yolo-voc.cfg",
    weights="/home/sam/OrkData/weights/tiny-yolo-voc.weights",
    detection_threshold=0.5,
)

# ecto options
scheduler_options(parser)
args = parser.parse_args()

plasm = ecto.Plasm()
plasm.connect(
    source['image'] >> detector['image']
)

# ecto.view_plasm(plasm)

run_plasm(args, plasm, locals=vars())
