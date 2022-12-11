import tensorflow as tf

from PIL import Image
from PIL import ImageOps
import numpy as np

import copy

class Interpreter:

    def __init__(self, model_path):

        # Load the model.
        self.model = tf.lite.Interpreter(model_path=model_path)

        # Set model input.
        self.input_details = self.model.get_input_details()
        self.model.allocate_tensors()

        # Get image size - converting from BHWC to WH
        self.input_size = self.input_details[0]['shape'][2], self.input_details[0]['shape'][1]
        print("Model : ", model_path)
        print("Model Input Size: ", self.input_size)
        
    def interprete(self, img):

        resized_image = img.convert('RGB').resize(self.input_size, Image.BILINEAR)

        # Convert to a NumPy array, add a batch dimension, and normalize the image.
        image_for_prediction = np.asarray(resized_image).astype(np.float32)
        image_for_prediction = np.expand_dims(image_for_prediction, 0)
        image_for_prediction = image_for_prediction / 127.5 - 1

        # Invoke the interpreter to run inference.
        self.model.set_tensor(self.input_details[0]['index'], image_for_prediction)
        self.model.invoke()

        # Retrieve the raw output map.
        raw_prediction = self.model.tensor(self.model.get_output_details()[0]['index'])()
        
        return copy.deepcopy(raw_prediction)
