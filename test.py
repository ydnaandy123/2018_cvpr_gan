import tensorflow as tf
from dataset_parser import SemanticParser
import numpy as np
from tensorflow.contrib.slim.nets import resnet_v1
slim = tf.contrib.slim

flags = tf.app.flags.FLAGS
tf.flags.DEFINE_string('mode', "train", "Mode train/ test-dev/ test")
tf.flags.DEFINE_boolean('debug', True, "Is debug mode or not")
tf.flags.DEFINE_string('dataset_dir', "./dataset/msra4000_semantic", "directory of the dataset")

tf.flags.DEFINE_integer("image_height", 224, "image target height")
tf.flags.DEFINE_integer("image_width", 224, "image target width")
tf.flags.DEFINE_integer("c_in_dim", 3, "input image color channel")
tf.flags.DEFINE_integer("c_out_dim", 1, "output image color channel")

tf.flags.DEFINE_integer("gf_dim", 64, "# of gen filters in first conv layer")
tf.flags.DEFINE_integer("df_dim", 64, "# of dis filters in first conv layer")
tf.flags.DEFINE_integer("batch_size", 1, "batch size for training")
tf.flags.DEFINE_integer("pool_size", 50, "max size for image pooling")

tf.flags.DEFINE_integer("training_iter", 160000, "data size for training")
tf.flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("beta1", 0.0, "Momentum term of adam [0.5]")
tf.flags.DEFINE_float("beta2", 0.9, "Momentum term of adam [0.9999]")
tf.flags.DEFINE_float("lambda_gp", 10.0, "Gradient penalty lambda hyper parameter [10.0]")

tf.flags.DEFINE_integer('save_freq', 2000, "save a model every save_freq iterations")
tf.flags.DEFINE_integer('log_freq', 10, "log a model every log_freq iterations")
tf.flags.DEFINE_integer('observe_freq', 400, "observe training image every observe_freq iterations")
tf.flags.DEFINE_integer('valid_freq', 200, "valid a model every valid_freq iterations")
tf.flags.DEFINE_integer('valid_num', 20, "num of images for each validation")


def main(args=None):
    print(args)
    tf.reset_default_graph()
    """
    Read dataset parser     
    """
    flags.network_name = args[0].split('/')[-1].split('.')[0].split('main_')[-1]
    flags.logs_dir = './logs_' + flags.network_name
    dataset_parser = SemanticParser(flags=flags)
    """
    Transform data to TFRecord format (Only do once.)     
    """
    if False:
        dataset_parser.load_paths(is_jpg=True, load_val=True)
        dataset_parser.data2record(name='{}_train.tfrecords'.format(dataset_parser.dataset_name),
                                   set_type='train', test_num=None)
        dataset_parser.data2record(name='{}_val.tfrecords'.format(dataset_parser.dataset_name),
                                   set_type='val', test_num=None)
        # coco_parser.data2record_test(name='coco_stuff2017_test-dev_all_label.tfrecords', is_dev=True, test_num=None)
        # coco_parser.data2record_test(name='coco_stuff2017_test_all_label.tfrecords', is_dev=False, test_num=None)
        return
    """
    Build Graph
    """
    with tf.Graph().as_default():
        """
        Input (TFRecord)
        """
        with tf.name_scope('TFRecord'):
            # DatasetA
            training_a_dataset = dataset_parser.tfrecord_get_dataset(
                name='{}_trainA.tfrecords'.format(dataset_parser.dataset_name), batch_size=flags.batch_size,
                shuffle_size=None)
            val_a_dataset = dataset_parser.tfrecord_get_dataset(
                name='{}_valA.tfrecords'.format(dataset_parser.dataset_name), batch_size=flags.batch_size)
            # DatasetB
            training_b_dataset = dataset_parser.tfrecord_get_dataset(
                name='{}_trainB.tfrecords'.format(dataset_parser.dataset_name), batch_size=flags.batch_size,
                shuffle_size=None)
            val_b_dataset = dataset_parser.tfrecord_get_dataset(
                name='{}_valB.tfrecords'.format(dataset_parser.dataset_name), batch_size=flags.batch_size)
            # A feed-able iterator
            with tf.name_scope('RealA'):
                handle_a = tf.placeholder(tf.string, shape=[])
                iterator_a = tf.contrib.data.Iterator.from_string_handle(
                    handle_a, training_a_dataset.output_types, training_a_dataset.output_shapes)
                real_a = iterator_a.get_next()
            with tf.name_scope('RealB'):
                handle_b = tf.placeholder(tf.string, shape=[])
                iterator_b = tf.contrib.data.Iterator.from_string_handle(
                    handle_b, training_b_dataset.output_types, training_b_dataset.output_shapes)
                real_b = iterator_b.get_next()
            with tf.name_scope('InitialA_op'):
                training_a_iterator = training_a_dataset.make_initializable_iterator()
                validation_a_iterator = val_a_dataset.make_initializable_iterator()
            with tf.name_scope('InitialB_op'):
                training_b_iterator = training_b_dataset.make_initializable_iterator()
                validation_b_iterator = val_b_dataset.make_initializable_iterator()
        """
        Network (Computes predictions from the inference model)
        """
        with tf.name_scope('Network'):
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                logits, end_points = resnet_v1.resnet_v1_50(real_a, num_classes=1000, is_training=False)
            init_fn = slim.assign_from_checkpoint_fn(
                './pretrained/resnet_v1_50.ckpt', slim.get_model_variables('resnet_v1_50'))

            saver = tf.train.Saver(max_to_keep=2)
            # Graph Logs
            """
            Session
            """
            tfconfig = tf.ConfigProto(allow_soft_placement=True)
            tfconfig.gpu_options.allow_growth = True
            with tf.Session(config=tfconfig) as sess:
                with tf.name_scope('Initial'):
                    ckpt = tf.train.get_checkpoint_state(dataset_parser.checkpoint_dir)
                    if ckpt and ckpt.model_checkpoint_path:
                        print("Model restored: {}".format(ckpt.model_checkpoint_path))
                        saver.restore(sess, ckpt.model_checkpoint_path)
                    else:
                        print("No Model found.")
                        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                        sess.run(init_op)
                        init_fn(sess)
                    summary_writer = tf.summary.FileWriter(dataset_parser.logs_dir, sess.graph)

                logits_sess = sess.run(logits, feed_dict={real_a: np.zeros([1, 224, 224, 3], dtype=np.float32)})

if __name__ == "__main__":
    tf.app.run()
