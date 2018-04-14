#!/usr/bin/env python
"""
Module defining the YOLO detector to find objects in a scene
"""

from ecto import BlackBoxCellInfo as CellInfo, BlackBoxForward as Forward
from object_recognition_core.db import ObjectDb, ObjectDbParameters
from object_recognition_core.pipelines.detection import DetectorBase

from ecto_opencv.rgbd import OnPlaneClusterer
from object_recognition_tabletop.ecto_cells.tabletop_table import TableDetector

import ecto
import ecto_cells.ecto_yolo as ecto_yolo

########################################################################################################################

class YoloDetector(ecto.BlackBox, DetectorBase):

    def __init__(self, *args, **kwargs):
        ecto.BlackBox.__init__(self, *args, **kwargs)
        DetectorBase.__init__(self)

    @staticmethod
    def declare_cells(_p):
        from object_recognition_tabletop.ecto_cells import tabletop_object

        return {'main': CellInfo(ecto_yolo.Detector)}

    @staticmethod
    def declare_forwards(_p):
        return ({'main':'all'}, {'main':'all'}, {'main':'all'})

    def connections(self, _p):
        return [self.main]
