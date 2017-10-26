import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator

dataset_trainA = Dataset.range(10)
dataset_valA = Dataset.range(10, 20)

dataset_trainB = Dataset.range(100, 110)
dataset_valB = Dataset.range(110, 120)

iter_trainA_handle = dataset_trainA.make_one_shot_iterator().string_handle()
iter_valA_handle = dataset_valA.make_one_shot_iterator().string_handle()
iter_trainB_handle = dataset_trainB.make_one_shot_iterator().string_handle()
iter_valB_handle = dataset_valB.make_one_shot_iterator().string_handle()

handleA = tf.placeholder(tf.string, shape=[])
iteratorA = Iterator.from_string_handle(
    handleA, dataset_trainA.output_types, dataset_trainA.output_shapes)
next_batchA = iteratorA.get_next()

handleB = tf.placeholder(tf.string, shape=[])
iteratorB = Iterator.from_string_handle(
    handleB, dataset_trainB.output_types, dataset_trainB.output_shapes)
next_batchB = iteratorB.get_next()

with tf.train.MonitoredTrainingSession() as sess:
    handle_trainA, handle_valA, handle_trainB, handle_valB = sess.run(
        [iter_trainA_handle, iter_valA_handle, iter_trainB_handle, iter_valB_handle])

    for step in range(10):
        print('trainA', sess.run(next_batchA, feed_dict={handleA: handle_trainA}))
        print('trainB', sess.run(next_batchB, feed_dict={handleB: handle_trainB}))

        if step % 3 == 0:
            print('valA', sess.run(next_batchA, feed_dict={handleA: handle_valA}))
            print('valB', sess.run(next_batchB, feed_dict={handleB: handle_valB}))

