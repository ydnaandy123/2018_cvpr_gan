from module import *

flags = tf.flags.FLAGS
tf.flags.DEFINE_integer("image_height", "256", "image target height")
tf.flags.DEFINE_integer("image_width", "256", "image target width")
tf.flags.DEFINE_integer("c_dim", "3", "image color channel")
tf.flags.DEFINE_integer("gf_dim", "64", "# of gen filters in first conv layer")
tf.flags.DEFINE_integer("df_dim", "64", "# of dis filters in first conv layer")
tf.flags.DEFINE_integer("batch_size", "4", "batch size for training")
tf.flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for Adam Optimizer")
tf.flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
tf.flags.DEFINE_float("beta2", 0.999, "Momentum term of adam [0.9999]")
tf.flags.DEFINE_float("lambda_gp", 10.0, "Gradient penalty lambda hyper parameter [10.0]")
tf.flags.DEFINE_float("lambda_rec", 10.0, "L1 lambda hyper parameter [10.0]")

tf.flags.DEFINE_integer("num_epochs", "50", "number of epochs for training")
tf.flags.DEFINE_boolean('debug', True, "Is debug mode or not")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test-dev/ test")
tf.flags.DEFINE_string('dataset_dir', "./dataset/horse2zebra", "directory of the dataset")
tf.flags.DEFINE_integer('save_freq', 1000, "save a model every save_freq iterations")
tf.flags.DEFINE_integer('log_freq', 100, "log a model every log_freq iterations")
tf.flags.DEFINE_integer('observe_freq', 600, "observe training image every observe_freq iterations")


def main(args=None):
    print(args)
    tf.reset_default_graph()
    """
    Read dataset parser     
    """
    flags.network_name = args[0].split('/')[-1].split('.')[0].split('main_')[-1]
    flags.logs_dir = './logs_' + flags.network_name

    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    is_training = tf.placeholder(tf.bool, name="is_training")
    drop_probability = tf.placeholder(tf.float32, name="drop_probability")
    real_a = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="real_a")
    real_b = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="real_b")
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
    recon_b = generator_resnet(fake_a, flags, True,  name="Generator_A2B")

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
    # d_loss = da_loss + db_loss

    trainable_var_gen_a2b = tf.get_collection(
        key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator_A2B')
    trainable_var_dis_b = tf.get_collection(
        key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator_B')
    trainable_var_gen_b2a = tf.get_collection(
        key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator_B2A')
    trainable_var_dis_a = tf.get_collection(
        key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator_A')
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op_gen_a2b = train_op(loss_gen_a2b, learning_rate, flags, trainable_var_gen_a2b, name='gen_a2b')
        train_op_dis_b = train_op(loss_dis_b, learning_rate, flags, trainable_var_dis_b, name='dis_b')
        train_op_gen_b2a = train_op(loss_gen_b2a, learning_rate, flags, trainable_var_gen_b2a, name='gen_b2a')
        train_op_dis_a = train_op(loss_dis_a, learning_rate, flags, trainable_var_dis_a, name='dis_a')

    # Graph Logs
    # summary_loss = tf.summary.scalar("loss_train", loss)
    # summary_loss_valid = tf.summary.scalar("loss_valid", valid_average_loss)

    summary_op = tf.summary.merge([
        tf.summary.scalar("loss/gen_cycle", loss_gen_cycle),
        tf.summary.scalar("loss/gen_recon_a", loss_gen_recon_a),
        tf.summary.scalar("loss/gen_recon_b", loss_gen_recon_b)
        ], name='GEN')

    tf.summary.scalar("loss/loss_gen_a2b", loss_gen_a2b)

    saver = tf.train.Saver(max_to_keep=2)

    """
    Session
    """
    with tf.Session() as sess:
        """
        Model restore and initialize
        """
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(flags.logs_dir, sess.graph)
        print(flags.logs_dir)
        sess.run(init_op)


if __name__ == "__main__":
    tf.app.run()
