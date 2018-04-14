#!/usr/bin/env python
import ecto
import ecto_ros
import ecto_ros.ecto_sensor_msgs as ecto_sensor_msgs

from ecto_opencv import imgproc, highgui
import object_recognition_yolo.ecto_cells.ecto_yolo as ecto_yolo
from object_recognition_core.db.tools import interpret_object_ids

import argparse
from ecto.opts import scheduler_options, run_plasm

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

json_db_str = '{"type": "CouchDB", "root": "http://localhost:5984", "collection": "object_recognition"}'
json_db = ecto.Constant(value=json_db_str)

object_id_str = '3fa57d1921eae374aca7334728014a8e'
object_id = ecto.Constant(value=object_id_str)

ImageBagger = ecto_sensor_msgs.Bagger_Image
CameraInfoBagger = ecto_sensor_msgs.Bagger_CameraInfo

image_ci = ecto_ros.CameraInfo2Cv('camera_info -> cv::Mat')
image = ecto_ros.Image2Mat()

bag = "/home/sam/rosbags/sigverse/no_objects.bag"

baggers = dict(image=ImageBagger(topic_name='/hsrb/head_rgbd_sensor/rgb/image_raw'),
               image_ci=CameraInfoBagger(topic_name='/hsrb/head_rgbd_sensor/rgb/camera_info'),
               )

sync = ecto_ros.BagReader('Bag Reader',
                          baggers=baggers,
                          bag=bag,
                          random_access=True,
                          )

rgb = imgproc.cvtColor('bgr -> rgb', flag=imgproc.Conversion.BGR2RGB)

display = highgui.imshow(name='Training Data')

image_mux = ecto_yolo.ImageMux()

graph = []

graph += [
            sync['image'] >> image['image'],
            image[:] >> rgb[:],
            test_cell['object_id'] >> trainer['object_id'],
            json_db['out'] >> trainer['json_db'],
            rgb[:] >> image_mux['image'],
            image_mux['image_out'] >> trainer['image'],
            trainer['box_labels', 'color_images'] >> training_image_saver['box_labels', 'color_images'],
            test_cell['object_id_index'] >> training_image_saver['object_id_index'],
         ]

            # trainer['debug_image'] >> display['image'],
            # object_id['out'] >> trainer['object_id'],

plasm = ecto.Plasm()
plasm.connect(graph)

# sched = ecto.Scheduler(plasm)
# sched.execute(niter=1)


# add ecto scheduler args.
scheduler_options(parser)
options = parser.parse_args()
run_plasm(options, plasm, locals=vars())
