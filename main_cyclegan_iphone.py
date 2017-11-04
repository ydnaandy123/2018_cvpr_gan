from __future__ import print_function
from dataset_parser import CycleParser, ImagePool
from module import *
import time

flags = tf.flags.FLAGS
tf.flags.DEFINE_integer("image_height", 256, "image target height")
tf.flags.DEFINE_integer("image_width", 256, "image target width")
tf.flags.DEFINE_integer("c_in_dim", 3, "input image color channel")
tf.flags.DEFINE_integer("c_out_dim", 3, "output image color channel")
tf.flags.DEFINE_integer("gf_dim", 64, "# of gen filters in first conv layer")
tf.flags.DEFINE_integer("df_dim", 64, "# of dis filters in first conv layer")

tf.flags.DEFINE_integer("batch_size", 1, "batch size for training")
tf.flags.DEFINE_integer("pool_size", 50, "max size for image pooling")
tf.flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
tf.flags.DEFINE_float("beta2", 0.999, "Momentum term of adam [0.9999]")
tf.flags.DEFINE_float("lambda_gp", 10.0, "Gradient penalty lambda hyper parameter [10.0]")
tf.flags.DEFINE_float("lambda_rec", 10.0, "Cycle lambda hyper parameter [10.0]")

tf.flags.DEFINE_integer("num_epochs", 200, "number of epochs for training")
tf.flags.DEFINE_integer("num_epochs_decay", 100, "number of epochs to decay learning rate")
tf.flags.DEFINE_boolean('debug', True, "Is debug mode or not")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test-dev/ test")
tf.flags.DEFINE_string('dataset_dir', "./dataset/iphone2dslr", "directory of the dataset")
tf.flags.DEFINE_integer('save_freq', 1000, "save a model every save_freq iterations")
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
    dataset_parser = CycleParser(dataset_dir=flags.dataset_dir, flags=flags)
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
    Input (TFRecord)
    """
    with tf.variable_scope('TFRecord'):
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
        with tf.variable_scope('RealA'):
            handle_a = tf.placeholder(tf.string, shape=[])
            iterator_a = tf.contrib.data.Iterator.from_string_handle(
                handle_a, training_a_dataset.output_types, training_a_dataset.output_shapes)
            real_a = iterator_a.get_next()
        with tf.variable_scope('RealB'):
            handle_b = tf.placeholder(tf.string, shape=[])
            iterator_b = tf.contrib.data.Iterator.from_string_handle(
                handle_b, training_b_dataset.output_types, training_b_dataset.output_shapes)
            real_b = iterator_b.get_next()
        with tf.variable_scope('InitialA_op'):
            training_a_iterator = training_a_dataset.make_initializable_iterator()
            validation_a_iterator = val_a_dataset.make_initializable_iterator()
        with tf.variable_scope('InitialB_op'):
            training_b_iterator = training_b_dataset.make_initializable_iterator()
            validation_b_iterator = val_b_dataset.make_initializable_iterator()
    """
    Network (Computes predictions from the inference model)
    """
    with tf.variable_scope('Network'):
        # Input
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        # is_training = tf.placeholder(tf.bool, name="is_training")
        # drop_probability = tf.placeholder(tf.float32, name="drop_probability")
        fake_a_pool = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='fake_A_pool')
        fake_b_pool = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='fake_B_pool')

        # A -> B
        fake_b = generator_resnet(real_a, flags, False, name="Generator_A2B")
        dis_fake_b = discriminator(fake_b, flags, reuse=False, name="Discriminator_B")
        dis_fake_b_pool = discriminator(fake_b_pool, flags, reuse=True, name="Discriminator_B")
        dis_real_b = discriminator(real_b, flags, reuse=True, name="Discriminator_B")

        # B -> A
        fake_a = generator_resnet(real_b, flags, False, name="Generator_B2A")
        dis_fake_a = discriminator(fake_a, flags, reuse=False, name="Discriminator_A")
        dis_fake_a_pool = discriminator(fake_a_pool, flags, reuse=True, name="Discriminator_A")
        dis_real_a = discriminator(real_a, flags, reuse=True, name="Discriminator_A")

        # Cycle
        recon_a = generator_resnet(fake_b, flags, True, name="Generator_B2A")
        recon_b = generator_resnet(fake_a, flags, True, name="Generator_A2B")

        with tf.variable_scope('loss_gen_cycle'):
            loss_gen_recon_a = abs_criterion(real_a, recon_a, name='recon_a')
            loss_gen_recon_b = abs_criterion(real_b, recon_b, name='recon_b')
            loss_gen_cycle = loss_gen_recon_a + loss_gen_recon_b
        with tf.variable_scope('loss_gen_a2b'):
            loss_gen_a2b_adv = mae_criterion(dis_fake_b, tf.ones_like(dis_fake_b), name='adv')
            loss_gen_a2b = flags.lambda_rec * loss_gen_cycle + loss_gen_a2b_adv
        with tf.variable_scope('loss_gen_b2a'):
            loss_gen_b2a_adv = mae_criterion(dis_fake_a, tf.ones_like(dis_fake_a), name='adv')
            loss_gen_b2a = flags.lambda_rec * loss_gen_cycle + loss_gen_b2a_adv

        with tf.variable_scope('loss_dis_b'):
            loss_dis_b_adv_real = mae_criterion(dis_real_b, tf.ones_like(dis_real_b), name='adv_real')
            loss_dis_b_adv_fake = mae_criterion(dis_fake_b_pool, tf.zeros_like(dis_fake_b_pool), name='adv_fake')
            loss_dis_b = loss_dis_b_adv_real + loss_dis_b_adv_fake
        with tf.variable_scope('loss_dis_a'):
            loss_dis_a_adv_real = mae_criterion(dis_real_a, tf.ones_like(dis_real_a), name='adv_real')
            loss_dis_a_adv_fake = mae_criterion(dis_fake_a_pool, tf.zeros_like(dis_fake_a_pool), name='adv_fake')
            loss_dis_a = loss_dis_a_adv_real + loss_dis_a_adv_fake

        trainable_var_gen_a2b = tf.get_collection(
            key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Network/Generator_A2B')
        trainable_var_dis_b = tf.get_collection(
            key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Network/Discriminator_B')
        trainable_var_gen_b2a = tf.get_collection(
            key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Network/Generator_B2A')
        trainable_var_dis_a = tf.get_collection(
            key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Network/Discriminator_A')
        # TODO: adam need learning rate?
        train_op_gen_a2b = train_op(loss_gen_a2b, learning_rate, flags, trainable_var_gen_a2b, name='gen_a2b')
        train_op_dis_b = train_op(loss_dis_b, learning_rate, flags, trainable_var_dis_b, name='dis_b')
        train_op_gen_b2a = train_op(loss_gen_b2a, learning_rate, flags, trainable_var_gen_b2a, name='gen_b2a')
        train_op_dis_a = train_op(loss_dis_a, learning_rate, flags, trainable_var_dis_a, name='dis_a')
        '''
        train_op_gen_a2b = tf.train.AdamOptimizer(learning_rate, flags.beta1, flags.beta2)\
            .minimize(loss_gen_a2b, var_list=trainable_var_gen_a2b, name='gen_a2b')
        train_op_dis_b = tf.train.AdamOptimizer(learning_rate, flags.beta1, flags.beta2)\
            .minimize(loss_dis_b, var_list=trainable_var_dis_b, name='dis_b')
        train_op_gen_b2a = tf.train.AdamOptimizer(learning_rate, flags.beta1, flags.beta2)\
            .minimize(loss_gen_b2a, var_list=trainable_var_gen_b2a, name='gen_b2a')
        train_op_dis_a = tf.train.AdamOptimizer(learning_rate, flags.beta1, flags.beta2)\
            .minimize(loss_dis_a, var_list=trainable_var_dis_a, name='dis_a')
        '''

    saver = tf.train.Saver(max_to_keep=2)
    # Graph Logs
    with tf.variable_scope('GEN_a2b'):
        tf.summary.scalar("loss/gen_a2b/all", loss_gen_a2b)
        tf.summary.scalar("loss/gen_a2b/adv", loss_gen_a2b_adv)
        tf.summary.scalar("loss/gen/cycle/all", loss_gen_cycle)
        tf.summary.scalar("loss/gen/cycle/recon_a", loss_gen_recon_a)
        tf.summary.scalar("loss/gen/cycle/recon_b", loss_gen_recon_b)
    with tf.variable_scope('DIS_b'):
        tf.summary.scalar("loss/dis_b/all", loss_dis_b)
        tf.summary.scalar("loss/dis_b/adv_real", loss_dis_b_adv_real)
        tf.summary.scalar("loss/dis_b/adv_fake", loss_dis_b_adv_fake)
    with tf.variable_scope('GEN_b2a'):
        tf.summary.scalar("loss/gen_b2a/all", loss_gen_b2a)
        tf.summary.scalar("loss/gen_b2a/adv", loss_gen_b2a_adv)
        tf.summary.scalar("loss/gen/cycle/all", loss_gen_cycle)
        tf.summary.scalar("loss/gen/cycle/recon_a", loss_gen_recon_a)
        tf.summary.scalar("loss/gen/cycle/recon_b", loss_gen_recon_b)
    with tf.variable_scope('DIS_a'):
        tf.summary.scalar("loss/dis_a/all", loss_dis_a)
        tf.summary.scalar("loss/dis_a/adv_real", loss_dis_a_adv_real)
        tf.summary.scalar("loss/dis_a/adv_fake", loss_dis_a_adv_fake)
    summary_op = tf.summary.merge_all()
    """
    Session
    """
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        with tf.variable_scope('Initial'):
            ckpt = tf.train.get_checkpoint_state(dataset_parser.checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("Model restored: {}".format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print("No Model found.")
                init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
                sess.run(init_op)
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
            global_step_sess = sess.run(global_step)
            for epoch in range(flags.num_epochs):
                sess.run([training_a_iterator.initializer, training_b_iterator.initializer])
                learning_rate_sess = flags.learning_rate if epoch < flags.num_epochs_decay\
                    else flags.learning_rate*(flags.num_epochs-epoch)/(flags.num_epochs-flags.num_epochs_decay)
                feed_dict_train = {learning_rate: learning_rate_sess,
                                   handle_a: training_a_handle, handle_b: training_b_handle}
                # feed_dict_valid = {is_training: False}
                while True:
                    try:
                        print('epoch:[{:d}/{:d}], global step:{:d}, learning rate:{:f}, time:{:4.4f}'.format(
                            epoch, flags.num_epochs, global_step_sess, learning_rate_sess, time.time() - start_time))

                        # Update gen_A2B, gen_B2A
                        _, fake_b_sess, _, fake_a_sess = sess.run([train_op_gen_a2b, fake_b, train_op_gen_b2a, fake_a],
                                                                  feed_dict=feed_dict_train)
                        # Update dis_B, dis_A
                        fake_b_pool_query = image_pool_b.query(fake_b_sess)
                        fake_a_pool_query = image_pool_a.query(fake_a_sess)
                        _, _ = sess.run([train_op_dis_b, train_op_dis_a],
                                        feed_dict={learning_rate: learning_rate_sess,
                                                   fake_b_pool: fake_b_pool_query, handle_b: training_b_handle,
                                                   fake_a_pool: fake_a_pool_query, handle_a: training_a_handle})

                        global_step_sess += 1
                        sess.run(tf.assign(global_step, global_step_sess))

                        # Logging the events
                        if global_step_sess % flags.log_freq == 1:
                            print('Logging the events')
                            summary_op_sess = sess.run(summary_op, feed_dict={
                                handle_a: training_a_handle, handle_b: training_b_handle,
                                fake_a_pool: fake_a_pool_query, fake_b_pool: fake_b_pool_query})
                            summary_writer.add_summary(summary_op_sess, global_step_sess)
                            # summary_writer.flush()

                        # Observe training situation (For debugging.)
                        if flags.debug and global_step_sess % flags.observe_freq == 1:
                            real_a_sess, real_b_sess, fake_a_sess, fake_b_sess, recon_a_sess, recon_b_sess = \
                                sess.run([real_a, real_b, fake_a, fake_b, recon_a, recon_b],
                                         feed_dict={handle_a: training_a_handle, handle_b: training_b_handle})
                            print('Logging training images.')
                            dataset_parser.visualize_data(
                                real_a=real_a_sess, real_b=real_b_sess, fake_a=fake_a_sess, fake_b=fake_b_sess,
                                recon_a=recon_a_sess, recon_b=recon_b_sess, shape=(1, 1),
                                global_step=global_step_sess, logs_dir=dataset_parser.logs_image_train_dir)

                        """
                        Saving the checkpoint
                        """
                        if global_step_sess % flags.save_freq == 0:
                            print('Saving model...')
                            saver.save(sess, dataset_parser.checkpoint_dir + '/model.ckpt', global_step=global_step)

                    except tf.errors.OutOfRangeError:
                        print('----------------One epochs finished!----------------')
                        break

if __name__ == "__main__":
    tf.app.run()
