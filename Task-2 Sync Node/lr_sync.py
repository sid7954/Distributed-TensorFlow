import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import os

start_time = time.time()

tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
tf.app.flags.DEFINE_string("job_name", "worker", "either worker or ps")
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)
config=tf.ConfigProto(log_device_placement=False)

clusterSpec_single = tf.train.ClusterSpec({
    "worker" : [
        "localhost:2222"
    ]
})

clusterSpec_cluster = tf.train.ClusterSpec({
    "ps" : [
        "128.104.222.193:2221"
    ],
    "worker" : [
        "128.104.222.193:2223",
        "128.104.222.195:2223"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps" : [
        "128.104.222.193:2221"
    ],
    "worker" : [
        "128.104.222.193:2223",
        "128.104.222.195:2223",
        "128.104.222.196:2223",
        "128.104.222.194:2223"
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}

clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index,config=config)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x=tf.placeholder(tf.float32, [None, 784]) 
y=tf.placeholder(tf.float32, [None, 10])
batch_size = 60
display_step = 1

if FLAGS.job_name == "ps":
    server.join()

elif FLAGS.job_name == "worker":
    is_chief = (FLAGS.task_index == 0)
    with tf.device(tf.train.replica_device_setter( worker_device="/job:worker/task:%d" % FLAGS.task_index, ps_device="/job:ps/cpu:0", cluster=clusterinfo)):
        W=tf.Variable(tf.zeros([784, 10]))
        b=tf.Variable(tf.zeros([10]))
        pred = tf.nn.softmax(tf.matmul(x, W) + b)
        loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
        global_step = tf.Variable(0, name="global_step", trainable=False)
        op=tf.train.GradientDescentOptimizer(0.01)
        opt=tf.train.SyncReplicasOptimizer(op, replicas_to_aggregate=4, total_num_replicas=4)
        optimizer = opt.minimize(loss, global_step=global_step)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    local_init_op = opt.local_step_init_op
    if is_chief:
        local_init_op = opt.chief_init_op

    ready_for_local_init_op = opt.ready_for_local_init_op
    chief_queue_runner = opt.get_chief_queue_runner()
    sync_init_op = opt.get_init_tokens_op()

    init = tf.global_variables_initializer()

    sv = tf.train.Supervisor(is_chief=is_chief, init_op=init, local_init_op=local_init_op, 
        ready_for_local_init_op=ready_for_local_init_op, recovery_wait_secs=1, global_step=global_step)
    sess = sv.prepare_or_wait_for_session(server.target)

    if is_chief:
        sess.run(sync_init_op)
        sv.start_queue_runners(sess, [chief_queue_runner])

    time_begin = time.time()
    local_step = 0
    train_steps=6000
    while True:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        train_feed = {x: batch_xs, y: batch_ys}
        _, step = sess.run([optimizer, global_step], feed_dict=train_feed)
        local_step += 1
        with sess.as_default():
        	if (step % 250)==0 :
	        	print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        if step >= train_steps:
            break
time_end = time.time()
training_time = time_end - time_begin
print("Training elapsed time: %f s" % training_time)