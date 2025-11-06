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
    
    @staticmethod
    def sigmoid(array):
        return 1 / (1 + np.exp(-1 * array))

    def process_outputs(self, outputs, image_size):

        boxes, box_confidences, box_class_probs = [], [], []

        i = 0
        for output in outputs:
            
            ## Grid dimensions
            grid_w, grid_h = output.shape[0 : 2]

            ## Getting raw(unprocessed) object 
            ## center and bounding box dimensions
            t_x_y = output[..., 0 : 2]
            t_w_h = output[..., 2 : 4]

            ## Getting anchor boxes
            anchors = self.anchors[i]

            ## Defining a grid for offsetting
            grid = np.tile(np.indices((grid_w, grid_h)).T, anchors.shape[0])
            grid = np.reshape(grid, (grid_w, grid_h, anchors.shape[0], 2))

            ## Offsetting
            b_x_y = self.sigmoid(t_x_y) + grid
            ## Scaling anchor boxes
            b_w_h = anchors * np.exp(t_w_h)

            ## Normalizing
            b_x_y /= [grid_w, grid_h]
            b_w_h /= self.model.inputs[0].shape.as_list()[1 : 3]

            ## Changing to rectangular coordinates
            top_left = b_x_y - (b_w_h / 2)
            bottom_right = b_x_y + (b_w_h / 2)
            
            ## Popuating boxes
            box = np.concatenate((top_left, bottom_right), axis = -1)
            ## Rescaling to original image size
            box = box * np.tile(np.flip(image_size, axis = 0), 2)
            boxes.append(box)

            ## Popuating box confidences
            confidence = np.expand_dims(self.sigmoid(output[..., 4]), axis = -1)
            box_confidences.append(confidence)

            ## Popuating box proabilities
            box_class_probs.append(self.sigmoid(output[..., 5:]))

            i += 1

        return boxes, box_confidences, box_class_probs
