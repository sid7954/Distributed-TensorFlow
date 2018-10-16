import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

# define the command line flags that can be sent
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
batch_size = 10
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

    if is_chief:
        print("Worker %d: Initializing session..." % FLAGS.task_index)
    else:
        print("Worker %d: Waiting for session to be initialized..." %  FLAGS.task_index)

    sess = sv.prepare_or_wait_for_session(server.target)
    #summary_writer = tf.train.SummaryWriter('logdir', sess.graph_def)

    print("Worker %d: Session initialization complete." % FLAGS.task_index)

    if is_chief:
        sess.run(sync_init_op)
        sv.start_queue_runners(sess, [chief_queue_runner])

    time_begin = time.time()
    print("Training begins @ %f" % time_begin)

    # with tf.train.MonitoredTrainingSession(master=server.target, is_chief=is_chief, config=config, hooks=hooks, stop_grace_period_secs=10) as sess:
    # #with tf.Session(server.target) as sess:
    #     print('Starting training on worker %d'%FLAGS.task_index)
    #     sess.run(init)
    #     while not sess.should_stop():
    #         #data0, data1, data2, data3 = tf.split(mnist, [mnist.train.num_examples/4,mnist.train.num_examples/4,mnist.train.num_examples/4,mnist.train.num_examples/4],0)
    #         total_batch = int(mnist.train.num_examples/(4*batch_size))
    #         for i in range(total_batch):
    #             print("Device:", '%04d' % FLAGS.task_index)
    #             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    #             _, cost = sess.run([optimizer, global_step], feed_dict={x: batch_xs, y: batch_ys})
    #         if is_chief:
    #             time.sleep(1)

    local_step = 0
    train_steps=6000
    while True:
        
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        train_feed = {x: batch_xs, y: batch_ys}

        _, step = sess.run([optimizer, global_step], feed_dict=train_feed)
        local_step += 1

        now = time.time()
        #print("%f: Worker %d: training step %d done (global step: %d)" % (now, FLAGS.task_index, local_step, step))

        if step >= train_steps:
            break

    with sess.as_default():
        print("Device:", '%04d' % FLAGS.task_index, "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

time_end = time.time()
print("Training ends @ %f" % time_end)
training_time = time_end - time_begin
print("Training elapsed time: %f s" % training_time)
