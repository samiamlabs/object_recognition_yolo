#!/usr/bin/env python
"""
Module defining the Yolo trainer to train the Yolo models
"""

from ecto import BlackBoxCellInfo as CellInfo, BlackBoxForward as Forward
from object_recognition_core.ecto_cells.db import ModelWriter
from object_recognition_core.pipelines.training import TrainerBase
import ecto
import ecto_cells.ecto_yolo as ecto_yolo

##############################################################################


class YoloTrainer(ecto.BlackBox, TrainerBase):
    '''Implements the training pipeline functions'''

    def __init__(self, *args, **kwargs):
        ecto.BlackBox.__init__(self, *args, **kwargs)
        TrainerBase.__init__(self)

    @classmethod
    def declare_cells(cls, _p):
        # passthrough cells
        cells = {'json_db': CellInfo(ecto.Constant),
                 'object_id': CellInfo(ecto.Constant)
                 }

        # cells = {}

        # 'real cells'
        cells.update({'model_filler': CellInfo(ecto_yolo.ModelFiller),
                      'model_writer': CellInfo(ModelWriter, params={'method': 'YOLO'}),
                      'trainer': CellInfo(ecto_yolo.Trainer)})

        return cells

    @classmethod
    def declare_forwards(cls, _p):
        p = {'json_db': [Forward('value', 'json_db')],
             'object_id': [Forward('value', 'object_id')]}
        # p = {'object_id': [Forward('value', 'object_id')]}
        # p = {}
        i = {}
        o = {}

        return (p, i, o)

    def connections(self, _p):
        connections = []

        # connect the constants
        # connections += [self.json_db['out'] >> self.trainer['json_db'],
        #                 self.object_id['out'] >> self.trainer['object_id']]
        # connections += [self.object_id['out'] >> self.trainer['object_id']]

        # connect the output to the post-processor
        # connections += [self.trainer['detector','Rs','Ts','distances','Ks','renderer_n_points','renderer_angle_step','renderer_radius_min','renderer_radius_max','renderer_radius_step','renderer_width','renderer_height','renderer_focal_length_x','renderer_focal_length_y','renderer_near','renderer_far'] >> self.model_filler['detector','Rs','Ts','distances','Ks','renderer_n_points','renderer_angle_step','renderer_radius_min','renderer_radius_max','renderer_radius_step','renderer_width','renderer_height','renderer_focal_length_x','renderer_focal_length_y','renderer_near','renderer_far']]

        # and write everything to the DB
        connections += [ self.object_id['out'] >> self.model_writer['object_id'],
                         self.model_filler['db_document'] >> self.model_writer['db_document']]

        return connections
