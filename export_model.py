import os
import sys

import oss2

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from tensorflow.examples.tutorials.mnist import input_data as mnist_input_data

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the exported model.')
tf.app.flags.DEFINE_string('checkpoint_path', "/var/logs", 'Checkpoints path.')
# tf.app.flags.DEFINE_string('checkpoint_path', "/Users/houyadong/Documents/pythonProject/mnist_example/logs", 'Checkpoints path.')
tf.app.flags.DEFINE_string('base_path', '/var/serving_dir', '')
# tf.app.flags.DEFINE_string('base_path', '/Users/houyadong/Documents/pythonProject/mnist_example/serving_dir', '')
FLAGS = tf.app.flags.FLAGS

def export_model():
  checkpoint_basename="model.ckpt"
  default_meta_graph_suffix='.meta'
  ckpt_path=os.path.join(FLAGS.checkpoint_path, checkpoint_basename + '-0')
  meta_graph_file=ckpt_path + default_meta_graph_suffix
  with tf.Session() as new_sess:
    new_saver = tf.train.import_meta_graph(meta_graph_file, clear_devices=True) #'/test/mnistoutput/ckpt.meta')
    new_saver.restore(new_sess, ckpt_path) #'/test/mnistoutput/ckpt')
    new_graph = tf.get_default_graph()
    new_x = new_graph.get_tensor_by_name('input/x-input:0')
    print(new_x)
    new_y = new_graph.get_tensor_by_name('layer2/activation:0')
    print(new_y)

    export_path_base = FLAGS.base_path
    export_path = os.path.join(
      compat.as_bytes(export_path_base),
      compat.as_bytes(str(FLAGS.model_version)))
    print('Exporting trained model to', export_path)
    builder = saved_model_builder.SavedModelBuilder(export_path)

  # Build the signature_def_map.
    tensor_info_x = utils.build_tensor_info(new_x)
    tensor_info_y = utils.build_tensor_info(new_y)

    prediction_signature = signature_def_utils.build_signature_def(
      inputs={'images': tensor_info_x},
      outputs={'scores': tensor_info_y},
      method_name=signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(tf.initialize_all_tables(), name='legacy_init_op')

    builder.add_meta_graph_and_variables(
      new_sess, [tag_constants.SERVING],
      signature_def_map={
          'predict_images':
              prediction_signature,
      },
      legacy_init_op=legacy_init_op,
      clear_devices=True)
    builder.save()

  print('Done exporting!')


def request_oss():
    try:
      accessid, accesskey, ossbucket, saved_dir = sys.argv[0], sys.argv[1], sys.argv[2], sys.argv[3]
      auth = oss2.Auth(str(accessid), str(accesskey))
      bucket = oss2.Bucket(auth, "http://oss-cn-beijing.aliyuncs.com", str(ossbucket))
      result = bucket.put_object_from_file(str(saved_dir) + "saved_model.pb", FLAGS.base_path + "/1/saved_model.pb")
      print("Done requesting oss")
    except Exception as e:
      print("error on requesting oss {}".format(e))


def main(_):
    export_model()
    request_oss()


if __name__ == '__main__':
  tf.app.run()