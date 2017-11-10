from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception_resnet_v2
from dataset_parser import SemanticParser
from ops import train_op
import time

flags = tf.app.flags.FLAGS
tf.flags.DEFINE_string('mode', "test", "Mode train/ test-dev/ test")
tf.flags.DEFINE_boolean('debug', True, "Is debug mode or not")
tf.flags.DEFINE_string('dataset_dir', "./dataset/msra_gt", "directory of the dataset")

tf.flags.DEFINE_integer("image_height", 299, "image target height")
tf.flags.DEFINE_integer("image_width", 299, "image target width")
tf.flags.DEFINE_integer("c_in_dim", 3, "input image color channel")
tf.flags.DEFINE_integer("c_out_dim", 1, "output image color channel")

tf.flags.DEFINE_integer("gf_dim", 64, "# of gen filters in first conv layer")
tf.flags.DEFINE_integer("df_dim", 64, "# of dis filters in first conv layer")
tf.flags.DEFINE_integer("batch_size", 1, "batch size for training")
tf.flags.DEFINE_integer("pool_size", 50, "max size for image pooling")

tf.flags.DEFINE_integer("training_iter", 60000, "data size for training")
tf.flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("beta1", 0.9, "Momentum term of adam [0.5]")
tf.flags.DEFINE_float("beta2", 0.999, "Momentum term of adam [0.9999]")
tf.flags.DEFINE_float("lambda_gp", 10.0, "Gradient penalty lambda hyper parameter [10.0]")

tf.flags.DEFINE_integer('save_freq', 8000, "save a model every save_freq iterations")
tf.flags.DEFINE_integer('log_freq', 80, "log a model every log_freq iterations")
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
        dataset_parser.load_paths(is_jpg=False, load_val=True)
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
                shuffle_size=None, need_flip=False)
            val_a_dataset = dataset_parser.tfrecord_get_dataset(
                name='{}_valA.tfrecords'.format(dataset_parser.dataset_name), batch_size=flags.batch_size,
                need_flip=False)
            # DatasetB
            training_b_dataset = dataset_parser.tfrecord_get_dataset(
                name='{}_trainB.tfrecords'.format(dataset_parser.dataset_name), batch_size=flags.batch_size,
                is_label=True, shuffle_size=None, need_flip=False)
            val_b_dataset = dataset_parser.tfrecord_get_dataset(
                name='{}_valB.tfrecords'.format(dataset_parser.dataset_name), batch_size=flags.batch_size,
                is_label=False, need_flip=False)
            # A feed-able iterator
            with tf.name_scope('RealA'):
                handle_a = tf.placeholder(tf.string, shape=[])
                iterator_a = tf.contrib.data.Iterator.from_string_handle(
                    handle_a, training_a_dataset.output_types, training_a_dataset.output_shapes)
                real_a, real_a_name, real_a_shape = iterator_a.get_next()
            with tf.name_scope('RealB'):
                handle_b = tf.placeholder(tf.string, shape=[])
                iterator_b = tf.contrib.data.Iterator.from_string_handle(
                    handle_b, training_b_dataset.output_types, training_b_dataset.output_shapes)
                real_b, real_b_name, real_b_shape = iterator_b.get_next()
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
            # Input
            global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)
            global_step_update_op = tf.assign_add(global_step, 1, name='global_step_update_op')
            mean_rgb = tf.constant((123.68, 116.78, 103.94), dtype=tf.float32)
            '''
            fake_b_pool = tf.placeholder(tf.float32,
                                         shape=[None, flags.image_height, flags.image_width, flags.c_in_dim],
                                         name='fake_B_pool')
            image_linear_shape = tf.constant(flags.image_height * flags.image_width * flags.c_in_dim,
                                             dtype=tf.int32, name='image_linear_shape')
            '''

            # A -> B
            with tf.name_scope('Generator'):
                with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
                    net, end_points = inception_resnet_v2.inception_resnet_v2(
                        real_a - mean_rgb, num_classes=None, is_training=True)

                with tf.variable_scope('Generator_A2B'):
                    fuse_1 = tf.layers.conv2d_transpose(net, 2080, [3, 3], strides=8, padding='valid') + end_points['Mixed_7a']
                    fuse_2 = tf.layers.conv2d_transpose(fuse_1, 1088, [3, 3], strides=2, padding='valid') + end_points['Mixed_6a']
                    fuse_3 = tf.layers.conv2d_transpose(fuse_2, 320, [3, 3], strides=2, padding='valid') + end_points['Mixed_5b']
                    pred = tf.layers.conv2d_transpose(fuse_3, 1, [8, 8], strides=8, padding='valid')

                    # pred = tf.layers.conv2d(tf.nn.relu(net), 1, 1, 1)
                    pred_upscale = tf.image.resize_bilinear(
                        pred, (flags.image_height, flags.image_width), name='up_scale')
                    segment_a = tf.nn.sigmoid(pred_upscale, name='segment_a')

                    logits_a_ori = tf.image.resize_bilinear(
                        pred, (real_a_shape[0][0], real_b_shape[0][1]), name='logits_a_ori')
                    segment_a_ori = tf.nn.sigmoid(logits_a_ori, name='segment_a_ori')

            # sigmoid cross entropy Loss
            with tf.name_scope('loss_gen_a2b'):
                loss_gen_a2b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=pred_upscale, labels=real_b/255.0, name='sigmoid'), name='mean')

            # Optimizer
            trainable_var_resnet = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='inception_resnet_v2')
            trainable_var_gen_a2b = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator_A2B') + trainable_var_resnet
            slim.model_analyzer.analyze_vars(trainable_var_gen_a2b, print_info=True)
            '''
            trainable_var_gen_a2b = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator_A2B')
            trainable_var_dis_b = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator_B')
            '''
            with tf.name_scope('learning_rate_decay'):
                decay = tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / flags.training_iter), name='decay')
                learning_rate = tf.multiply(flags.learning_rate, decay, name='learning_rate')
            train_op_gen_a2b = train_op(loss_gen_a2b, learning_rate, flags, trainable_var_gen_a2b, name='gen_a2b')
            # train_op_dis_b = train_op(loss_dis_b, learning_rate, flags, trainable_var_dis_b, name='dis_b')

        saver = tf.train.Saver(max_to_keep=2)
        # Graph Logs
        with tf.name_scope('GEN_a2b'):
            tf.summary.scalar("loss/gen_a2b/all", loss_gen_a2b)
        '''
        with tf.name_scope('DIS_b'):
            tf.summary.scalar("loss/dis_b/all", loss_dis_b)
            tf.summary.scalar("loss/dis_b/adv_real", loss_dis_b_adv_real)
            tf.summary.scalar("loss/dis_b/adv_fake", loss_dis_b_adv_fake)
        '''
        summary_op = tf.summary.merge_all()
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

                    init_fn = slim.assign_from_checkpoint_fn('./pretrained/inception_resnet_v2.ckpt',
                                                             slim.get_model_variables('inception_resnet_v2'))
                    init_fn(sess)
                summary_writer = tf.summary.FileWriter(dataset_parser.logs_dir, sess.graph)
            """
            Training Mode
            """
            if flags.mode == 'train':
                print('Training mode! Batch size:{:d}'.format(flags.batch_size))
                with tf.variable_scope('Input_port'):
                    training_a_handle = sess.run(training_a_iterator.string_handle())
                    training_b_handle = sess.run(training_b_iterator.string_handle())
                    # val_a_handle = sess.run(validation_a_iterator.string_handle())
                    # val_b_handle = sess.run(validation_b_iterator.string_handle())

                print('Start Training!')
                start_time = time.time()
                sess.run([training_a_iterator.initializer, training_b_iterator.initializer])
                feed_dict_train = {handle_a: training_a_handle, handle_b: training_b_handle}
                # feed_dict_valid = {is_training: False}
                global_step_sess = sess.run(global_step)
                while global_step_sess < flags.training_iter:
                    try:
                        # Update gen_A2B, gen_B2A
                        _, loss_gen_a2b_sess = sess.run([train_op_gen_a2b, loss_gen_a2b], feed_dict=feed_dict_train)

                        sess.run(global_step_update_op)
                        global_step_sess, learning_rate_sess = sess.run([global_step, learning_rate])
                        print('global step:[{:d}/{:d}], learning rate:{:f}, time:{:4.4f}'.format(
                            global_step_sess, flags.training_iter, learning_rate_sess, time.time() - start_time))

                        # Logging the events
                        if global_step_sess % flags.log_freq == 1:
                            print('Logging the events')
                            summary_op_sess = sess.run(summary_op, feed_dict={
                                handle_a: training_a_handle, handle_b: training_b_handle})
                            summary_writer.add_summary(summary_op_sess, global_step_sess)
                            # summary_writer.flush()

                        # Observe training situation (For debugging.)
                        if flags.debug and global_step_sess % flags.observe_freq == 1:
                            real_a_sess, real_b_sess, segment_a_sess,\
                                real_a_name_sess = \
                                sess.run([real_a, real_b, segment_a, real_a_name],
                                         feed_dict={handle_a: training_a_handle, handle_b: training_b_handle})
                            print('Logging training images.')
                            dataset_parser.visualize_data(
                                real_a=real_a_sess, real_b=real_b_sess,
                                segment_a=segment_a_sess, shape=(1, 1),
                                global_step=global_step_sess, logs_dir=dataset_parser.logs_image_train_dir,
                                real_a_name=real_a_name_sess[0].decode())
                        """
                        Saving the checkpoint
                        """
                        if global_step_sess % flags.save_freq == 0:
                            print('Saving model...')
                            saver.save(sess, dataset_parser.checkpoint_dir + '/model.ckpt',
                                       global_step=global_step_sess)

                    except tf.errors.OutOfRangeError:
                        print('----------------One epochs finished!----------------')
                        sess.run([training_a_iterator.initializer, training_b_iterator.initializer])
            elif flags.mode == 'test':
                from PIL import Image
                import scipy.io as sio
                import numpy as np
                print('Start Testing!')
                '''
                with tf.variable_scope('Input_port'):
                    val_a_handle = sess.run(validation_a_iterator.string_handle())
                    val_b_handle = sess.run(validation_b_iterator.string_handle())
                sess.run([validation_a_iterator.initializer, validation_b_iterator.initializer])
                '''
                with tf.variable_scope('Input_port'):
                    val_a_handle = sess.run(validation_a_iterator.string_handle())
                    val_b_handle = sess.run(validation_b_iterator.string_handle())
                sess.run([validation_a_iterator.initializer, validation_b_iterator.initializer])
                feed_dict_test = {handle_a: val_a_handle, handle_b: val_b_handle}
                image_idx = 0
                while True:
                    try:
                        segment_a_ori_sess, real_a_name_sess, real_b_sess = \
                            sess.run([segment_a_ori, real_a_name, real_b], feed_dict=feed_dict_test)
                        segment_a_ori_sess = (np.squeeze(segment_a_ori_sess)) * 255.0
                        x_png = Image.fromarray(segment_a_ori_sess.astype(np.uint8))
                        x_png.save('{}/{}_pred.png'.format(dataset_parser.logs_image_val_dir,
                                                           real_a_name_sess[0].decode()), format='PNG')
                        real_b_sess = np.squeeze(real_b_sess)
                        x_png = Image.fromarray(real_b_sess.astype(np.uint8))
                        x_png.save('{}/{}.png'.format(dataset_parser.logs_image_val_dir,
                                                      real_a_name_sess[0].decode()), format='PNG')

                        sio.savemat('{}/{}.mat'.format(
                            dataset_parser.logs_mat_output_dir, real_a_name_sess[0].decode()),
                                    {'pred': np.squeeze(segment_a_ori_sess)})
                        print(image_idx)
                        image_idx += 1
                    except tf.errors.OutOfRangeError:
                        print('----------------One epochs finished!----------------')
                        break
if __name__ == "__main__":
    tf.app.run()
