#!/usr/bin/env python
import ecto
import ecto_ros
import ecto_ros.ecto_sensor_msgs as ecto_sensor_msgs

from ecto_opencv import imgproc, highgui, calib
from ecto_opencv.highgui import ImageReader
import object_recognition_yolo.ecto_cells.ecto_yolo as ecto_yolo
from object_recognition_core.db.tools import interpret_object_ids

import argparse
from ecto.opts import scheduler_options, run_plasm

import os
import sys

from ecto_opencv.calib import PoseDrawer
from ecto_opencv.highgui import MatPrinter

from ecto_ros.ecto_geometry_msgs import Publisher_PoseStamped
from ecto_ros.ecto_sensor_msgs import Publisher_Image, Publisher_PointCloud

from object_recognition_reconstruction import MatToPointCloudXYZRGB, PointCloudTransform
from ecto_image_pipeline.base import RescaledRegisteredDepth

import ecto_pcl
import ecto_pcl_ros

from object_recognition_core.db import tools, models
from object_recognition_core.db.cells import ObservationInserter

import couchdb
import object_recognition_core.db.tools as dbtools
from object_recognition_core.db.tools import args_to_db_params


class IdDispatcher(ecto.cell.Cell):
    @staticmethod
    def declare_params(params):
        params.declare("N", "number of outputs", 3)

    @staticmethod
    def declare_io(params, inputs, outputs):
        outputs.declare(
            "object_id", "The object id, to associate this model with", "123")
        outputs.declare("object_id_index",
                        "The object id index, to associate this model with", 0)

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


class ObservationRenderer():
    def __init__(self):
        self.graph = []

        self.parse_args()
        self.connect_background_source()
        self.connect_renderer()

    def connect_background_source(self):
        json_db_str = '{"type": "CouchDB", "root": "http://localhost:5984", "collection": "object_recognition"}'
        self.json_db = ecto.Constant(value=json_db_str)

        object_id_str = '4680aac58c1d263b9449d57bd2000f27'
        self.object_id = ecto.Constant(value=object_id_str)

        self.image_ci = ecto_ros.CameraInfo2Cv('camera_info -> cv::Mat')
        self.image = ecto_ros.Image2Mat()
        self.depth = ecto_ros.Image2Mat()

        bag = "/home/sam/rosbags/sigverse/no_objects.bag"

        ImageBagger = ecto_sensor_msgs.Bagger_Image
        CameraInfoBagger = ecto_sensor_msgs.Bagger_CameraInfo

        baggers = dict(
            image=ImageBagger(
                topic_name='/hsrb/head_rgbd_sensor/rgb/image_raw'),
            image_ci=CameraInfoBagger(
                topic_name='/hsrb/head_rgbd_sensor/rgb/camera_info'),
            depth=ImageBagger(
                topic_name='/hsrb/head_rgbd_sensor/depth/image_raw'),
        )

        # this will read all images in the path
        path = '/home/sam/Code/vision/VOCdevkit/VOC2012/JPEGImages'
        self.file_source = ImageReader(path=os.path.expanduser(path))

        self.bag_reader = ecto_ros.BagReader('Bag Reader',
                                             baggers=baggers,
                                             bag=bag,
                                             random_access=True,
                                             )

        self.rgb = imgproc.cvtColor(
            'bgr -> rgb', flag=imgproc.Conversion.BGR2RGB)
        self.display = highgui.imshow(name='Training Data', waitKey=10000)
        self.image_mux = ecto_yolo.ImageMux()

        self.graph += [
            self.bag_reader['image'] >> self.image['image'],
            self.bag_reader['depth'] >> self.depth['image'],
            self.image[:] >> self.rgb[:],
            self.bag_reader['image_ci'] >> self.image_ci['camera_info'],
            self.image_ci['K'] >> self.image_mux['K1'],
            self.rgb['image'] >> self.image_mux['image1'],
            self.depth['image'] >> self.image_mux['depth1'],
            self.file_source['image'] >> self.image_mux['image2'],
        ]

    def connect_renderer(self):
        self.id_dispatcher = IdDispatcher()
        self.observation_renderer = ecto_yolo.ObservationRenderer()

        self.graph += [
            self.image_mux['K'] >> self.observation_renderer['K'],
            self.image_mux['image'] >> self.observation_renderer['image'],
            self.image_mux['depth'] >> self.observation_renderer['depth'],
            self.json_db['out'] >> self.observation_renderer['json_db'],
        ]

        self.graph += [
            self.id_dispatcher['object_id'] >> self.observation_renderer['object_id'],
            # self.object_id['out'] >> self.observation_renderer['object_id'],
        ]

    def connect_debug(self):
        self.pose_drawer = PoseDrawer()

        self.graph += [
            self.observation_renderer['R'] >> MatPrinter(name='R')['mat'],
            self.observation_renderer['T'] >> MatPrinter(name='T')['mat'],
            self.observation_renderer['K'] >> MatPrinter(name='K')['mat'],
            self.observation_renderer['R', 'T',
                                      'K'] >> self.pose_drawer['R', 'T', 'K'],
            self.observation_renderer['debug_image'] >> self.pose_drawer['image'],
        ]

        self.graph += [
            self.pose_drawer['output'] >> self.display['image'],
            # self.observation_renderer['depth'] >> self.display['image'],
            # self.observation_renderer['mask'] >> self.display['image'],
        ]

    def connect_ros_pose(self):
        frame_id_str = 'camera_optical_frame'
        self.rt2pose = ecto_ros.RT2PoseStamped(frame_id=frame_id_str)
        self.pose_publisher = Publisher_PoseStamped(topic_name='/object_pose')

        self.graph += [
            self.observation_renderer['R', 'T'] >> self.rt2pose['R', 'T'],
            self.rt2pose['pose'] >> self.pose_publisher['input'],
        ]

    def connect_ros_image(self):
        frame_id_str = 'camera_optical_frame'

        self.image2ros = ecto_ros.Mat2Image(
            frame_id=frame_id_str, encoding='bgr8')
        self.depth2ros = ecto_ros.Mat2Image(
            frame_id=frame_id_str, encoding='16UC1')

        self.image_publisher = Publisher_Image(topic_name='/image')
        self.depth_publisher = Publisher_Image(topic_name='/depth')

        self.graph += [
            self.observation_renderer['image'] >> self.image2ros['image'],
            self.observation_renderer['depth'] >> self.depth2ros['image'],
            self.observation_renderer['depth'] >> self.depth2point_cloud_ros['image'],
            self.image2ros['image'] >> self.image_publisher['input'],
            self.depth2ros['image'] >> self.depth_publisher['input'],
        ]

    def connect_ros_point_cloud(self):
        self.depthTo3d = calib.DepthTo3d('Depth ~> 3D')
        self.erode = imgproc.Erode('Mask Erosion', kernel=3)  # -> 7x7
        # this is for SXGA mode scale handling.
        self.rescale_depth = RescaledRegisteredDepth('Depth scaling')

        self.point_cloud_converter = MatToPointCloudXYZRGB('To Point Cloud')
        self.to_ecto_pcl = ecto_pcl.PointCloudT2PointCloud(
            'converter', format=ecto_pcl.XYZRGB)

        self.cloud2msg = ecto_pcl_ros.PointCloud2Message("cloud2msg")
        self.cloud_pub = ecto_sensor_msgs.Publisher_PointCloud2(
            "cloud_pub", topic_name='/ecto_pcl/sample_output')

        self.graph += [
            self.observation_renderer['depth', 'image',
                                      'mask'] >> self.rescale_depth['depth', 'image', 'mask'],
            self.observation_renderer['K'] >> self.rescale_depth['K'],
            self.rescale_depth['K'] >> self.depthTo3d['K'],
            self.rescale_depth['depth'] >> self.depthTo3d['depth'],
            self.depthTo3d['points3d'] >> self.point_cloud_converter['points'],
            self.observation_renderer['image'] >> self.point_cloud_converter['image'],
            self.rescale_depth['mask'] >> self.erode['image'],
            self.erode['image'] >> self.point_cloud_converter['mask'],
            self.point_cloud_converter['point_cloud'] >> self.to_ecto_pcl['input'],
            self.to_ecto_pcl['output'] >> self.cloud2msg['input'],
            self.cloud2msg['output'] >> self.cloud_pub['input'],
        ]

    def connect_db_inserter(self):
        self.couch = couchdb.Server(self.options.db_root)

        self.db = dbtools.init_object_databases(self.couch)
        sessions = self.db

        session = models.Session()
        session.object_id = "various"
        session.bag_id = "none"
        session.store(self.db)

        session_id = session.id
        print("Session id: " + session_id)
        object_id = "test_object_id"
        db_params = args_to_db_params(self.options)

        db_inserter = ObservationInserter(
            "db_inserter", object_id=object_id,
            session_id=session_id, db_params=db_params)

        self.graph += [
            self.observation_renderer['depth'] >> db_inserter['depth'],
            self.observation_renderer['R', 'T'] >> db_inserter['R', 'T'],
            self.observation_renderer['mask'] >> db_inserter['mask'],
            self.observation_renderer['image'] >> db_inserter['image'],
            self.id_dispatcher['object_id'] >> db_inserter['object_id'],
            self.image_ci['K'] >> db_inserter['K'],
        ]

    def parse_args(self):
        self.parser = argparse.ArgumentParser(
            description='Generate observations by 3d rendering.')
        dbtools.add_db_arguments(self.parser)
        scheduler_options(self.parser)
        self.options = self.parser.parse_args()

    def do_ecto(self):

        plasm = ecto.Plasm()
        plasm.connect(self.graph)

        # sched = ecto.Scheduler(plasm)
        # sched.execute(niter=1)

        ecto.view_plasm(plasm)

        # add ecto scheduler args.
        run_plasm(self.options, plasm, locals=vars())


if __name__ == '__main__':
    ecto_ros.init(sys.argv, "observation_renderer")
    observation_renderer = ObservationRenderer()
    # observation_renderer.connect_debug()
    observation_renderer.connect_ros_pose()

    # observation_renderer.connect_ros_image()
    observation_renderer.connect_ros_point_cloud()
    observation_renderer.connect_db_inserter()

    observation_renderer.do_ecto()
