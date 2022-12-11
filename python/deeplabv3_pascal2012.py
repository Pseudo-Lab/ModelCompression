import argparse
import os
import time
import numpy as np
import tensorflow as tf

from PIL import Image

from TfLite_Interpreter import Interpreter
import display_seg_map

import cv2

def parsing_argument():

    parser = argparse.ArgumentParser(description="DeepLabv3 Evaluator",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter    )

    parser.add_argument('--model_path', metavar='str', type=str, 
                        default ='../../models/tflite/deeplabv3_mnv2_dm05_pascal_trainval_int.tflite', 
                        help='tflite model path')
    parser.add_argument('--db_path', metavar='str', type=str, 
                        default ='../datasets/VOCdevkit/VOC2012/', 
                        help='tflite model path ({path}/VOCdevkit/VOC2012/)')

    return parser.parse_args()

def compute_miou(gt, pred):
    # source : https://github.com/CYBORG-NIT-ROURKELA/Improving_Semantic_segmentation/blob/master/miou_calculation.py
    # converting into 1d array and then finding the frequency of each class
    pred = pred.reshape((pred.size,))
    pred_count = np.bincount(pred, weights = None, minlength = 21) # A
    gt = gt.reshape((gt.size,))
    actual_count = np.bincount(gt, weights = None, minlength = 21) # B

    temp = gt * 21 + pred

    cm = np.bincount(temp, weights = None, minlength = 441)
    cm = cm.reshape((21, 21))

    # A ⋂ B
    Nr = np.diag(cm) 
    # A ⋃ B
    Dr = pred_count + actual_count - Nr 

    # (A ⋂ B)/(A ⋃ B)
    individual_iou = Nr / Dr 
    # nanmean is used to neglect 0/0 case which arise due to absence of any class
    miou = np.nanmean(individual_iou) 
    return miou

def get_segment_map(output_tensor, width, height):

    seg_map = tf.argmax(tf.image.resize(output_tensor, (height, width)), axis=3)
    seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)
    
    return seg_map

def simple_test(model, img_path, output_path):
    
    input_img = Image.open(img_path)
    output_tensor = model.interprete(input_img)
    
    width, height = input_img.size
    seg_map = get_segment_map(output_tensor, width, height)
    
    seg_image = display_seg_map.label_to_color_image(seg_map).astype(np.uint8)

    cv2.imwrite(output_path, seg_image)

def evaluation(models, db_path):
    
    gt_path = os.path.join(db_path, "SegmentationClass")
    img_path = os.path.join(db_path, "JPEGImages")
    
    iou_array =np.array([0])
    proc_time_array =np.array([0])
    
    for files in os.listdir(gt_path):

        filename = files.rsplit('.')[0]

        # Get GT        
        gt_map = display_seg_map._remove_colormap_deeplab(os.path.join(gt_path, files))
        gt_map[gt_map == 255] = 0
        
        # predict
        input_img = Image.open(os.path.join(img_path, filename +".jpg"))

        start_time = time.time()
        output_tensor = models.interprete(input_img)
        end_time = time.time()
        proc_time = end_time - start_time
        proc_time_array = np.append(proc_time_array, proc_time)
        
        width, height = input_img.size
        seg_map = get_segment_map(output_tensor, width, height)

        # evaluate
        iou = compute_miou(gt_map, seg_map)
        iou_array = np.append(iou_array, iou)

        
        print(files, ": iou =", iou, ", Proc. Time=", proc_time)
    
    avr_iou = np.mean(iou_array)
    avr_proc_time = np.mean(proc_time_array)

    print("average iou =", avr_iou, "avr proc. time=", avr_proc_time)
    return avr_iou

def main(args):

    deeplab = Interpreter(args.model_path)

    # simple test
    simple_test(deeplab, "image3.jpg", "result.jpg")

    # PASCAL VOC 2021 validation
    evaluation(deeplab, args.db_path)

if __name__ == '__main__':

    args = parsing_argument()
    main(args)