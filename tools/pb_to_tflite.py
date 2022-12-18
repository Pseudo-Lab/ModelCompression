# Reference: https://colab.research.google.com/drive/1qUbQy8yJWPBp3ReLnE8otMgzS9Qw686C#scrollTo=VCBfZcj2eV_Y

import argparse
import os
import numpy as np
import tensorflow as tf

import pb_opt

import copy

# from "https://eehoeskrap.tistory.com/521"
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parsing_argument():

    parser = argparse.ArgumentParser(description="DeepLabv3 Evaluator",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter    )

    parser.add_argument('--model_path', metavar='str', type=str, 
                        default ='deeplabv3/tf/deeplabv3_mnv2_dm05_pascal_trainval/frozen_inference_graph.pb', 
                        help='tflite model path')
    parser.add_argument('--input_layer_name', metavar='str', type=str, 
                        default ='sub_7', 
                        help='input layername')
    parser.add_argument('--output_layer_name', metavar='str', type=str, 
                        default ='ResizeBilinear_2', 
                        help='output layername')
    parser.add_argument('--input_shape', nargs=4, metavar=('batch', 'height', 'width', 'ch'), type=int, 
                        default =[1 , 513, 513, 3], 
                        help='input shape (e.g. 1 513 513 3')
    parser.add_argument('-b', '--opt_graph', help='optimize Frozen Graph', default=True, type=str2bool)    

    return parser.parse_args()

def representative_dataset_gen():
    
    for _ in range(10):
        dummy_image = tf.random.uniform(g_input_shape, 0., 255., dtype=tf.float32)
        dummy_image = dummy_image / 127.5 - 1
        yield [dummy_image]

def convert_tflite(args):

    global g_input_shape
    g_input_shape = copy.deepcopy(args.input_shape)

    input_file_name = os.path.basename(args.model_path)

    # Load the TensorFlow model
    # The preprocessing and the post-processing steps should not be included in the TF Lite model graph 
    # because some operations (ArgMax) might not support the delegates. 
    # Insepct the graph using Netron https://lutzroeder.github.io/netron/
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file = args.model_path, 
        input_arrays = [args.input_layer_name],
        output_arrays = [args.output_layer_name]
    )

    tflite_model = converter.convert()
    tflite_path = input_file_name + '_fp32.tflite'
    tflite_model_size = open(tflite_path, 'wb').write(tflite_model)


    # Optional: Perform the simplest optimization known as post-training dynamic range quantization.
    # https://www.tensorflow.org/lite/performance/post_training_quantization#dynamic_range_quantization
    # You can refer to the same document for other types of optimizations.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert to TFLite Model
    tflite_model = converter.convert()

    #_, dynamic_tflite_path = tempfile.mkstemp('.tflite')
    dynamic_tflite_path = input_file_name + '_dynamic_quant.tflite'
    tflite_model_size = open(dynamic_tflite_path, 'wb').write(tflite_model)
    tf_model_size = os.path.getsize(args.model_path)
    print('TensorFlow Model is  {} bytes'.format(tf_model_size))
    print('TFLite Model is      {} bytes'.format(tflite_model_size))
    print('Post training dynamic range quantization saves {} bytes'.format(tf_model_size-tflite_model_size))


    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file = args.model_path, 
        input_arrays = [args.input_layer_name],
        output_arrays = [args.output_layer_name]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    #_, f16_tflite_path = tempfile.mkstemp('.tflite')
    f16_tflite_path = input_file_name + '_fp16.tflite'
    tflite_model_size = open(f16_tflite_path, 'wb').write(tflite_model)
    tf_model_size = os.path.getsize(args.model_path)
    print('TensorFlow Model is  {} bytes'.format(tf_model_size))
    print('TFLite Model is      {} bytes'.format(tflite_model_size))
    print('Post training float16 quantization saves {} bytes'.format(tf_model_size-tflite_model_size))


    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file = args.model_path, 
        input_arrays = [args.input_layer_name],
        output_arrays = [args.output_layer_name]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen

    tflite_model = converter.convert()

    int_tflite_path = input_file_name + '_int_quant.tflite'
    tflite_model_size = open(int_tflite_path, 'wb').write(tflite_model)
    tf_model_size = os.path.getsize(args.model_path)
    print('TensorFlow Model is  {} bytes'.format(tf_model_size))
    print('TFLite Model is      {} bytes'.format(tflite_model_size))
    print('Post training int8 quantization saves {} bytes'.format(tf_model_size-tflite_model_size))


if __name__ == '__main__':

    args = parsing_argument()

    """
    if args.opt_graph :
        
        opt_output_path = os.path.basename(args.model_path) + "_opt.pb"
        pb_opt.optimize_graph(args.model_path, opt_output_path, args.input_layer_name, args.output_layer_name)
        args.model_path = opt_output_path
    """
    convert_tflite(args)