import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# define the command line flags that can be sent
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
        "sgarg33@c220g2-011009.wisc.cloudlab.us:2222"
    ],
    "worker" : [
        "sgarg33@c220g2-011009.wisc.cloudlab.us:2222",
        "sgarg33@c220g2-011011.wisc.cloudlab.us:2222"
    ]
})

clusterSpec_cluster2 = tf.train.ClusterSpec({
    "ps" : [
        "sgarg33@c220g2-011009.wisc.cloudlab.us:2222"
    ],
    "worker" : [
        "sgarg33@c220g2-011009.wisc.cloudlab.us:2222",
        "sgarg33@c220g2-011011.wisc.cloudlab.us:2222",
        "sgarg33@c220g2-011012.wisc.cloudlab.us:2222",
        "sgarg33@c220g2-011010.wisc.cloudlab.us:2222"
    ]
})

clusterSpec = {
    "single": clusterSpec_single,
    "cluster": clusterSpec_cluster,
    "cluster2": clusterSpec_cluster2
}

clusterinfo = clusterSpec[FLAGS.deploy_mode]
server = tf.train.Server(clusterinfo, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    #put your code here
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    batch_size = 10
    display_step = 1

    x=tf.placeholder(tf.float32, [None, 784]) 
    y=tf.placeholder(tf.float32, [None, 10])
    W=tf.Variable(tf.zeros([784, 10]))
    b=tf.Variable(tf.zeros([10]))

    pred = tf.nn.softmax(tf.matmul(x, W) + b)
    loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(40):
            avg_cost = 0.0
            total_batch = int(mnist.train.num_examples/batch_size)

            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, cost = sess.run([optimizer, loss], feed_dict={x: batch_xs, y: batch_ys})
                avg_cost += cost / total_batch
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        print("Trained")

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
