import numpy as np
import cv2
import os
import display_seg_map

if __name__ == '__main__':

    arr = np.fromfile(open('/media/Data/lib/snpe-1.66.0.3729/bin/x86_64-linux-clang/output/Result_1/ResizeBilinear_2:0.raw', 'r'), dtype="float32")
    print(arr.size)


    arr = np.reshape(arr, (513,513,21))
    arr = np.argmax(arr, axis=2)
    
    segment = arr[342:, 342:]

    seg_image = display_seg_map.label_to_color_image(segment).astype(np.uint8)

    cv2.imwrite("dlcout.jpg", seg_image)
