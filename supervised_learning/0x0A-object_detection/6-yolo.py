#!/usr/bin/env python3

import cv2
import os
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
        
        ## actual model
        self.model = K.models.load_model(model_path)
        ## names of object classes
        self.class_names = class_names
        ## score threshold
        self.class_t = class_t
        ## non-max suppresion threshold
        self.nms_t = nms_t
        ## anchor boxes for outputs
        self.anchors = anchors
    
    @staticmethod
    def sigmoid(array):
        ## sigmoid activation
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

        return (boxes, box_confidences, box_class_probs)
    
    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        filtered_boxes, box_classes, box_scores = [], [], []

        i = 0
        for box in boxes:
            ## Get confidence and class_probabiities of each box
            confidence = box_confidences[i]
            class_prob = box_class_probs[i]
            
            ## Compute score
            box_score = confidence * class_prob

            ## Get the best scoring class's index and actual score
            best_class = np.argmax(box_score, axis = -1)
            best_class_score = np.max(box_score, axis = -1)

            ## Generating a filter
            mask = best_class_score >= self.class_t

            # Filtering
            filtered_boxes.append(box[mask])
            box_classes.append(best_class[mask])
            box_scores.append(best_class_score[mask])

            i += 1
        
        ## Converting to ndarray
        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)


        return (filtered_boxes, box_classes, box_scores)

    @staticmethod
    def IoU(box_1, box_2):
        ## Coordinates of possibble intersection
        x_1 = max(box_1[0], box_2[0])
        y_1 = max(box_1[1], box_2[1])
        x_2 = min(box_1[2], box_2[2])
        y_2 = min(box_1[3], box_2[3])

        ## Check actual intersection
        if (x_1 < x_2) and (y_1 < y_2):
            ## Area of intersection
            area_I = (x_2 -x_1) * (y_2 - y_1)
            ## Area of box_1
            area_1 = (box_1[2] - box_1[0]) * (box_1[3] - box_1[1])
            ## Area of box_2
            area_2 = (box_2[2] - box_2[0]) * (box_2[3] - box_2[1])

            return area_I / (area_1 + area_2 - area_I)
        else:
            return 0

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):

        box_predictions, predicted_box_classes, predicted_box_scores = [], [], []

        # Sort by score (descending) and class (descending), but keep track of all indices
        indices = np.lexsort((box_scores, -box_classes))[::-1]

        while len(indices) > 0:
            # Pick the box with the highest score and add it to the prediction
            box_predictions.append(filtered_boxes[indices[0]])
            predicted_box_classes.append(box_classes[indices[0]])
            predicted_box_scores.append(box_scores[indices[0]])

            # Remove the selected box from further NMS
            indices = indices[1:]

            suppressed = []
            i = 0
            for idx in indices:
                if box_classes[idx] == predicted_box_classes[-1]:
                    # Suppress if overlapping and same class
                    if self.IoU(box_predictions[-1], filtered_boxes[idx]) > self.nms_t:
                        suppressed.append(i)
                i += 1

            # Continue with the non-suppressed
            indices = np.delete(indices, suppressed)

        # Convert to NumPy arrays
        return (
            np.array(box_predictions),
            np.array(predicted_box_classes),
            np.array(predicted_box_scores)
            )

    @staticmethod
    def load_images(folder_path):
        images, image_paths = [], []
        img_path = folder_path + "/"
        image_path = os.listdir(folder_path)

        for path in image_path:
            
            images.append(cv2.imread(img_path + path))
            image_paths.append(img_path + path)

        return images, image_paths

    def preprocess_images(self, images):
        pimages, image_shapes = [], []
        
        ## Image dimensions
        i_width = self.model.input.shape[1]
        i_height = self.model.input.shape[2]

        for image in images:
            image_shapes.append([image.shape[0], image.shape[1]])
            pimages.append(cv2.resize(src = image, 
                                      dsize = (i_width, i_height), 
                                      interpolation = cv2.INTER_CUBIC
                                     ))
        ## Normalize pixel value
        pimages = [x / 255 for x in pimages]

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        
        for i in range(boxes.shape[0]):
            ## Convert the float coordinates to integers
            x_1, y_1, x_2, y_2 = map(int, boxes[i])
            ## Draw bounding box
            image = cv2.rectangle(img = image,
                                 pt1 = (x_1, y_1),
                                 pt2 = (x_2, y_2),
                                 color = (255, 0, 0),
                                 thickness = 2)

            ## Classname and score
            class_name = self.class_names[box_classes[i]]
            score = np.around(box_scores, 2)[i]
            ## Output text
            text = str(class_name) + " " + str(score)
            ## Draw the text on the image
            image = cv2.putText(img = image,
                                text = text,
                                org = (x_1, y_1 - 5),
                                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale = 0.5,
                                color = (0, 0, 255),
                                thickness = 1,
                                lineType = cv2.LINE_AA)
        
        cv2.namedWindow(winname = file_name,
                        flags = cv2.WINDOW_NORMAL)

        # Set a fixed size for the window
        cv2.resizeWindow(winname = file_name,
                        width = 600,
                        height = 600)
        
        ## The window
        cv2.imshow(winname = file_name, 
                   mat = image)

        ## Waiting to close
        key = cv2.waitKey(delay = 0) & 0xFF
        if key == ord('s'):
            try:
                os.mkdir("detections")
            except FileExistsError:
                pass
            ## Save image
            cv2.imwrite(filename = "detections/" + file_name,
                       image = image)
        cv2.destroyAllWindows()
