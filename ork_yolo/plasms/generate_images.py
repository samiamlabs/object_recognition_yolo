#!/usr/bin/env python
import ecto
import ecto_ros
import ecto_ros.ecto_sensor_msgs as ecto_sensor_msgs

from ecto_opencv import imgproc, highgui
from ecto_opencv.highgui import ImageReader
import object_recognition_yolo.ecto_cells.ecto_yolo as ecto_yolo
from object_recognition_core.db.tools import interpret_object_ids

import argparse
from ecto.opts import scheduler_options, run_plasm

import os

from ecto_opencv.calib import PoseDrawer
from ecto_opencv.highgui import MatPrinter

class TestCell(ecto.cell.Cell):
    @staticmethod
    def declare_params(params):
        params.declare("N", "number of outputs", 3)

    @staticmethod
    def declare_io(params, inputs, outputs):
        outputs.declare("object_id", "The object id, to associate this model with", "123")
        outputs.declare("object_id_index", "The object id index, to associate this model with", 0)

    def configure(self, params, inputs, outputs):
        json_db_str = '{"type": "CouchDB", "root": "http://localhost:5984", "collection": "object_recognition"}'

        self.object_ids = list(interpret_object_ids(json_db_str, 'all', []))
        self.object_id_index = 0

    def process(self, inputs=None, outputs=None):
        print("Printing obj id:")
        if self.object_id_index == len(self.object_ids):
            self.object_id_index = 0

        print self.object_ids[self.object_id_index]
        print self.object_id_index
        outputs.object_id = self.object_ids[self.object_id_index]
        outputs.object_id_index = self.object_id_index

        self.object_id_index += 1
        return 0


parser = argparse.ArgumentParser(description='Generate training data by 3d rendering.')

test_cell = TestCell()

trainer = ecto_yolo.Trainer()
training_image_saver = ecto_yolo.TrainingImageSaver()

observation_renderer  = ecto_yolo.ObservationRenderer()

json_db_str = '{"type": "CouchDB", "root": "http://localhost:5984", "collection": "object_recognition"}'
json_db = ecto.Constant(value=json_db_str)

object_id_str = '4680aac58c1d263b9449d57bd2000f27'
object_id = ecto.Constant(value=object_id_str)

frame_id_str = 'camera_optical_frame'
frame_id = ecto.Constant(value=frame_id_str)

ImageBagger = ecto_sensor_msgs.Bagger_Image
CameraInfoBagger = ecto_sensor_msgs.Bagger_CameraInfo

image_ci = ecto_ros.CameraInfo2Cv('camera_info -> cv::Mat')
image = ecto_ros.Image2Mat()
depth = ecto_ros.Image2Mat()

bag = "/home/sam/rosbags/sigverse/no_objects.bag"

baggers = dict(
    image=ImageBagger(topic_name='/hsrb/head_rgbd_sensor/rgb/image_raw'),
    image_ci=CameraInfoBagger(topic_name='/hsrb/head_rgbd_sensor/rgb/camera_info'),
    depth=ImageBagger(topic_name='/hsrb/head_rgbd_sensor/depth/image_raw'),
)

# this will read all images in the path
path = '/home/sam/Code/vision/VOCdevkit/VOC2012/JPEGImages'
file_source = ImageReader(path=os.path.expanduser(path))

bag_reader = ecto_ros.BagReader('Bag Reader',
                          baggers=baggers,
                          bag=bag,
                          random_access=True,
                          )

rgb = imgproc.cvtColor('bgr -> rgb', flag=imgproc.Conversion.BGR2RGB)

display = highgui.imshow(name='Training Data', waitKey=10000)

image_mux = ecto_yolo.ImageMux()

pose_drawer = PoseDrawer()

rt_2_pose = ecto_ros.RT2PoseStamped(frame_id=frame_id_str)

graph = []

graph += [
            bag_reader['image'] >> image['image'],
            bag_reader['depth'] >> depth['image'],
            image[:] >> rgb[:],
            bag_reader['image_ci'] >> image_ci['camera_info'],
            image_ci['K'] >> image_mux['K1'],
            image_mux['K'] >> observation_renderer['K'],
            rgb['image'] >> image_mux['image1'],
            depth['image'] >> image_mux['depth1'],
            file_source['image'] >> image_mux['image2'],
            image_mux['image'] >> observation_renderer['image'],
            image_mux['depth'] >> observation_renderer['depth'],
            json_db['out'] >> observation_renderer['json_db'],
        ]

# graph += [
#             bag_reader['image'] >> image['image'],
#             image[:] >> rgb[:],
#             rgb['image'] >> observation_renderer['image'],
#             json_db['out'] >> observation_renderer['json_db'],
#         ]

# graph += [
#             object_id['out'] >> observation_renderer['object_id'],
#          ]

graph += [
            test_cell['object_id'] >> observation_renderer['object_id'],
         ]


graph += [
            observation_renderer['R'] >> MatPrinter(name='R')['mat'],
            observation_renderer['T'] >> MatPrinter(name='T')['mat'],
            observation_renderer['K'] >> MatPrinter(name='K')['mat'],
            observation_renderer['R', 'T', 'K'] >> pose_drawer['R', 'T', 'K'],
            observation_renderer['debug_image'] >> pose_drawer['image'],
            pose_drawer['output'] >> display['image'],
        ]

graph += [
            observation_renderer['R', 'T'] >> rt_2_pose['R', 'T'],
         ]
            # frame_id['out'] >> rt_2_pose['frame_id'],

plasm = ecto.Plasm()
plasm.connect(graph)

# sched = ecto.Scheduler(plasm)
# sched.execute(niter=1)

ecto.view_plasm(plasm)

# add ecto scheduler args.
scheduler_options(parser)
options = parser.parse_args()
run_plasm(options, plasm, locals=vars())
