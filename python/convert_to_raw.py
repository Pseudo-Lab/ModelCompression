import numpy as np
import cv2
import os

def convert2raw(db_path, width, height):
    
    gt_path = os.path.join(db_path, "SegmentationClass")
    img_path = os.path.join(db_path, "JPEGImages")
    
    iou_array =np.array([0])
    proc_time_array =np.array([0])

    f = open(os.path.join(db_path, "list.txt"),'w')

    
    for files in os.listdir(gt_path):

        filename = files.rsplit('.')[0]

        path = os.path.join(img_path, filename +".jpg")
        frame = cv2.imread(path)

        print(frame.shape)

        frame_resized = cv2.resize(frame,(width,height))
        # Pad smaller dimensions to Mean value & Multiply with 0.007843
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (width,height), (127.5, 127.5, 127.5), swapRB=True)
        # Making numpy array of required shape
        blob = np.reshape(blob, (1,width,height,3))
        # Storing to a raw file
        save_path = os.path.join(img_path, filename +".raw")
        np.ndarray.tofile(blob, open(save_path,'w') )

        f.write(save_path+"\n")

        print(save_path)

    f.close()            


if __name__ == '__main__':

    db_path = '/media/Data/Git/ModelCompression/eval/datasets/VOCdevkit/VOC2012/'
    convert2raw(db_path, 513, 513)