# Reference: https://gist.github.com/huningxin/060fe2ac60b61781d6ea36f83969ed2a
# Tested on TF1.15
# In TF2.0 above, The graph optimizer use Grappler
# https://www.tensorflow.org/guide/graph_optimization
# http://web.stanford.edu/class/cs245/slides/TFGraphOptimizationsStanford.pdf

from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import freeze_graph
from tensorflow.python import ops
#from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.tools import graph_transforms
import tensorflow as tf


opt_option =['remove_nodes(op=Identity)', 
             'merge_duplicate_nodes', 
             'strip_unused_nodes', 
             'fold_constants(ignore_errors=true)', 
             'fold_batch_norms', 
             'fold_old_batch_norms']

def get_graph_def_from_file(graph_filepath):
    tf.compat.v1.reset_default_graph
    with ops.Graph().as_default():
        with tf.io.gfile.GFile(graph_filepath, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def

def optimize_graph(in_model_file, out_model_file, input_layer_names, output_layer_names):
    graph_def = get_graph_def_from_file(in_model_file)
    optimized_graph_def = graph_transforms.TransformGraph(
        graph_def,
        input_layer_names,
        output_layer_names,
        opt_option
    )
    tf.io.write_graph(optimized_graph_def,
        logdir='./',
        as_text=False,
        name=out_model_file)