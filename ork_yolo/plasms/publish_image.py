#!/usr/bin/env python
import ecto
from ecto_opencv.highgui import imshow, ImageReader
import ecto_ros, ecto_ros.ecto_sensor_msgs as ecto_sensor_msgs

import os
import sys

from ecto.opts import run_plasm, cell_options, scheduler_options
import argparse

ImagePub = ecto_sensor_msgs.Publisher_Image

def do_ecto():
    # ecto options
    parser = argparse.ArgumentParser(description='Publish images from directory.')
    scheduler_options(parser)
    args = parser.parse_args()

    #this will read all images in the path
    path = '/home/sam/Code/vision/VOCdevkit/VOC2012/JPEGImages'
    images = ImageReader(path=os.path.expanduser(path))
    mat2image = ecto_ros.Mat2Image(encoding='rgb8')

    pub_rgb = ImagePub("image_pub", topic_name='/camera/rgb/image_raw')
    display = imshow(name='image', waitKey=5000)

    plasm = ecto.Plasm()
    plasm.connect(
        images['image'] >>  mat2image['image'],
        images['image'] >> display['image'],
        mat2image['image'] >> pub_rgb['input'],
    )

    ecto.view_plasm(plasm)
    run_plasm(args, plasm, locals=vars())

if __name__ == '__main__':
    ecto_ros.init(sys.argv, "image_pub")
    do_ecto()
