#!/usr/bin/env python3

import tensorflow.keras as K
import numpy as np

class Yolo:

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        
        # Loading class names
        class_names = []
        file = open(classes_path, "r")
        for line in file:
            class_names.append(line.strip())
        file.close()

        self.model = K.models.load_model(model_path)
        self.class_names = class_names
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
