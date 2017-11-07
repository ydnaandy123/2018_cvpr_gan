from __future__ import print_function
import tensorflow as tf
from dataset_parser import GANParser, ImagePool
from ops import train_op
from module import generator_resnet_sigmoid, discriminator_se_wgangp, high_light
import time

flags = tf.app.flags.FLAGS
tf.flags.DEFINE_string('mode', "train", "Mode train/ test-dev/ test")
tf.flags.DEFINE_boolean('debug', True, "Is debug mode or not")
tf.flags.DEFINE_string('dataset_dir', "./dataset/exp_msra4000_hlt", "directory of the dataset")

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
    dataset_parser = GANParser(flags=flags)
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
                name='{}_valB.tfrecords'.format(dataset_parser.dataset_name), batch_size=flags.batch_size,)
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
            # Input
            global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)
            global_step_update_op = tf.assign_add(global_step, 1, name='global_step_update_op')
            mean_rgb = tf.constant((123.68, 116.78, 103.94), dtype=tf.float32)
            fake_b_pool = tf.placeholder(tf.float32,
                                         shape=[None, flags.image_height, flags.image_width, flags.c_in_dim],
                                         name='fake_B_pool')
            image_linear_shape = tf.constant(flags.image_height * flags.image_width * flags.c_in_dim,
                                             dtype=tf.int32, name='image_linear_shape')
            real_a_test = tf.placeholder(tf.float32,
                                         shape=[None, flags.image_height, flags.image_width, flags.c_in_dim],
                                         name='real_a_test')

            # A -> B
            '''
            with tf.name_scope('Generator'):
                with slim.arg_scope(vgg.vgg_arg_scope()):
                    net, end_points = vgg.vgg_16(real_a - mean_rgb, num_classes=1, is_training=True, spatial_squeeze=False)
                    print(net)
                    return

                with tf.variable_scope('Generator_A2B'):
                    pred = tf.layers.conv2d(tf.nn.relu(net), 1, 1, 1)
                    pred_upscale = tf.image.resize_bilinear(pred, (flags.image_height, flags.image_width), name='up_scale')
                    segment_a = tf.nn.sigmoid(pred_upscale, name='segment_a')

            # sigmoid cross entropy Loss
            with tf.name_scope('loss_gen_a2b'):
                loss_gen_a2b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=pred_upscale, labels=real_b/255.0, name='sigmoid'), name='mean')
            '''

            # A -> B
            # adjusted_a = tf.zeros_like(real_a, tf.float32, name='mask', optimize=True)
            adjusted_a = high_light(real_a, name='high_light')
            segment_a = generator_resnet_sigmoid(real_a, flags, False, name="Generator_A2B")
            segment_a_test = generator_resnet_sigmoid(real_a_test, flags, True, name="Generator_A2B")
            with tf.variable_scope('Fake_B'):
                foreground = tf.multiply(real_a, segment_a, name='foreground')
                background = tf.multiply(adjusted_a, (1 - segment_a), name='background')
                fake_b = tf.add(foreground, background, name='fake_b')

            #
            fake_b_f = tf.reshape(fake_b, [-1, image_linear_shape], name='fake_b_f')
            fake_b_pool_f = tf.reshape(fake_b_pool, [-1, image_linear_shape], name='fake_b_pool_f')
            real_b_f = tf.reshape(real_b, [-1, image_linear_shape], name='real_b_f')
            dis_fake_b = discriminator_se_wgangp(fake_b_f, flags, reuse=False, name="Discriminator_B")
            dis_fake_b_pool = discriminator_se_wgangp(fake_b_pool_f, flags, reuse=True, name="Discriminator_B")
            dis_real_b = discriminator_se_wgangp(real_b_f, flags, reuse=True, name="Discriminator_B")

            # WGAN Loss
            with tf.name_scope('loss_gen_a2b'):
                loss_gen_a2b = -tf.reduce_mean(dis_fake_b)

            with tf.name_scope('loss_dis_b'):
                loss_dis_b_adv_real = -tf.reduce_mean(dis_real_b)
                loss_dis_b_adv_fake = tf.reduce_mean(dis_fake_b_pool)
                loss_dis_b = tf.reduce_mean(dis_fake_b_pool) - tf.reduce_mean(dis_real_b)
                with tf.name_scope('wgan-gp'):
                    alpha = tf.random_uniform(shape=[flags.batch_size, 1], minval=0., maxval=1.)
                    differences = fake_b_pool_f - real_b_f
                    interpolates = real_b_f + (alpha * differences)
                    gradients = tf.gradients(
                        discriminator_se_wgangp(interpolates, flags, reuse=True, name="Discriminator_B"),
                        [interpolates])[0]
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                    loss_dis_b += flags.lambda_gp * gradient_penalty

            # Optimizer
            '''
            trainable_var_resnet = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_16')
            trainable_var_gen_a2b = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator_A2B') + trainable_var_resnet
            slim.model_analyzer.analyze_vars(trainable_var_gen_a2b, print_info=True)
            '''
            trainable_var_gen_a2b = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator_A2B')
            trainable_var_dis_b = tf.get_collection(
                key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator_B')
            with tf.name_scope('learning_rate_decay'):
                decay = tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / flags.training_iter), name='decay')
                learning_rate = tf.multiply(flags.learning_rate, decay, name='learning_rate')
            train_op_gen_a2b = train_op(loss_gen_a2b, learning_rate, flags, trainable_var_gen_a2b, name='gen_a2b')
            train_op_dis_b = train_op(loss_dis_b, learning_rate, flags, trainable_var_dis_b, name='dis_b')

        saver = tf.train.Saver(max_to_keep=2)
        # Graph Logs
        with tf.name_scope('GEN_a2b'):
            tf.summary.scalar("loss/gen_a2b/all", loss_gen_a2b)
        with tf.name_scope('DIS_b'):
            tf.summary.scalar("loss/dis_b/all", loss_dis_b)
            tf.summary.scalar("loss/dis_b/adv_real", loss_dis_b_adv_real)
            tf.summary.scalar("loss/dis_b/adv_fake", loss_dis_b_adv_fake)
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

                    # init_fn = slim.assign_from_checkpoint_fn('./pretrained/vgg_16.ckpt',
                    #                                          slim.get_model_variables('vgg_16'))
                    # init_fn(sess)
                summary_writer = tf.summary.FileWriter(dataset_parser.logs_dir, sess.graph)
            """
            Training Mode
            """
            if flags.mode == 'train':
                print('Training mode! Batch size:{:d}'.format(flags.batch_size))
                with tf.variable_scope('Input_port'):
                    training_a_handle = sess.run(training_a_iterator.string_handle())
                    val_a_handle = sess.run(validation_a_iterator.string_handle())
                    training_b_handle = sess.run(training_b_iterator.string_handle())
                    val_b_handle = sess.run(validation_b_iterator.string_handle())
                    image_pool_a, image_pool_b = ImagePool(flags.pool_size), ImagePool(flags.pool_size)

                print('Start Training!')
                start_time = time.time()
                sess.run([training_a_iterator.initializer, training_b_iterator.initializer])
                feed_dict_train = {handle_a: training_a_handle, handle_b: training_b_handle}
                # feed_dict_valid = {is_training: False}
                global_step_sess = sess.run(global_step)
                while global_step_sess < flags.training_iter:
                    try:
                        # Update gen_A2B, gen_B2A
                        _, fake_b_sess, = sess.run([train_op_gen_a2b, fake_b], feed_dict=feed_dict_train)
                        # _, loss_gen_a2b_sess = sess.run([train_op_gen_a2b, loss_gen_a2b], feed_dict=feed_dict_train)

                        # Update dis_B, dis_A
                        fake_b_pool_query = image_pool_b.query(fake_b_sess)
                        _ = sess.run(train_op_dis_b, feed_dict={
                            fake_b_pool: fake_b_pool_query, handle_b: training_b_handle})

                        sess.run(global_step_update_op)
                        global_step_sess, learning_rate_sess = sess.run([global_step, learning_rate])
                        print('global step:[{:d}/{:d}], learning rate:{:f}, time:{:4.4f}'.format(
                            global_step_sess, flags.training_iter, learning_rate_sess, time.time() - start_time))

                        # Logging the events
                        if global_step_sess % flags.log_freq == 1:
                            print('Logging the events')
                            summary_op_sess = sess.run(summary_op, feed_dict={
                                handle_a: training_a_handle, handle_b: training_b_handle,
                                fake_b_pool: fake_b_pool_query})
                            summary_writer.add_summary(summary_op_sess, global_step_sess)
                            # summary_writer.flush()

                        # Observe training situation (For debugging.)
                        if flags.debug and global_step_sess % flags.observe_freq == 1:
                            real_a_sess, real_b_sess, adjusted_a_sess, segment_a_sess, fake_b_sess = \
                                sess.run([real_a, real_b, adjusted_a, segment_a, fake_b],
                                         feed_dict={handle_a: training_a_handle, handle_b: training_b_handle})
                            print('Logging training images.')
                            dataset_parser.visualize_data(
                                real_a=real_a_sess, real_b=real_b_sess, adjusted_a=adjusted_a_sess,
                                segment_a=segment_a_sess, fake_b=fake_b_sess, shape=(1, 1),
                                global_step=global_step_sess, logs_dir=dataset_parser.logs_image_train_dir)
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
                from glob import glob
                from PIL import Image
                import os
                dataset_dir = './dataset/msra4000'
                data = []
                for folder in os.listdir(dataset_dir):
                    path = os.path.join(dataset_dir, folder, "*.jpg")
                    data.extend(glob(path))
                data_len = len(data)
                for img_idx, img_path in enumerate(data):
                    print('[{:d}/{:d}]'.format(img_idx, data_len))
                    img_name = img_path.split('/')[-1].split('.jpg')[0]
                    test_img = Image.open(img_path)
                    test_img_size = test_img.size
                    test_img = test_img.resize((flags.image_height, flags.image_width), Image.BILINEAR)
                    x = np.array(test_img)
                    if len(x.shape) < 3:
                        x = np.dstack((x, x, x))
                    x = np.expand_dims(x, axis=0)
                    segment_a_test_sess = sess.run(
                        segment_a_test, feed_dict={real_a_test: x})
                    segment_a_test_sess = np.array(segment_a_test_sess) * 255
                    x_png = Image.fromarray(np.squeeze(segment_a_test_sess).astype(np.uint8))
                    x_png = x_png.resize(test_img_size, Image.BILINEAR)
                    x_png.save('{}/{}.png'.format(dataset_parser.logs_image_val_dir, img_name), format='PNG')

if __name__ == "__main__":
    tf.app.run()
