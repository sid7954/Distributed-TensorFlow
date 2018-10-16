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
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x=tf.placeholder(tf.float32, [None, 784]) 
y=tf.placeholder(tf.float32, [None, 10])
batch_size = 25
display_step = 1

if FLAGS.job_name == "ps":
    server.join()

elif FLAGS.job_name == "worker":
    
    with tf.device(tf.train.replica_device_setter( worker_device="/job:worker/task:%d" % FLAGS.task_index, cluster=clusterinfo)):
        W=tf.Variable(tf.zeros([784, 10]))
        b=tf.Variable(tf.zeros([10]))
        pred = tf.nn.softmax(tf.matmul(x, W) + b)
        loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
        global_step = tf.contrib.framework.get_or_create_global_step()
        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)

    hooks=[tf.train.StopAtStepHook(last_step=50)]
    init = tf.global_variables_initializer()

    #with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0), checkpoint_dir="/tmp/train_logs", hooks=hooks) as sess:
    with tf.Session(server.target) as sess:
        sess.run(init)
        for q in range(30):
            avg_cost = 0.0
            total_batch = int(mnist.train.num_examples/batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, cost = sess.run([optimizer, loss], feed_dict={x: batch_xs, y: batch_ys})
                avg_cost += cost / total_batch
            print("Device:", '%04d' % FLAGS.task_index,"Epoch:", '%04d' % (q+1), "cost=", "{:.9f}".format(avg_cost))

        print("Trained %d",FLAGS.task_index)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Device:", '%04d' % FLAGS.task_index, "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

print("--- %s seconds ---" % (time.time() - start_time))