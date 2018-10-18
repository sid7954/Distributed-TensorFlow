from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .common import ModelBuilder
from .alexnetcommon import alexnet_inference, alexnet_part_conv, alexnet_loss, alexnet_eval
from ..optimizers.momentumhybrid import HybridMomentumOptimizer


# def original(images, labels, num_classes, total_num_examples, devices=None, is_train=True):
#     """Build inference"""
#     if devices is None:
#         devices = [None]
#
#     def configure_optimizer(global_step, total_num_steps):
#         """Return a configured optimizer"""
#         def exp_decay(start, tgtFactor, num_stairs):
#             decay_step = total_num_steps / (num_stairs - 1)
#             decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
#             return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
#                                               staircase=True)
#
#         def lparam(learning_rate, momentum):
#             return {
#                 'learning_rate': learning_rate,
#                 'momentum': momentum
#             }
#
#         return HybridMomentumOptimizer({
#             'weights': lparam(exp_decay(0.001, 250, 4), 0.9),
#             'biases': lparam(exp_decay(0.002, 10, 2), 0.9),
#         })
#
#     def train(total_loss, global_step, total_num_steps):
#         """Build train operations"""
#         # Compute gradients
#         with tf.control_dependencies([total_loss]):
#             opt = configure_optimizer(global_step, total_num_steps)
#             grads = opt.compute_gradients(total_loss)
#
#         # Apply gradients.
#         apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
#
#         with tf.control_dependencies([apply_gradient_op]):
#             return tf.no_op(name='train')
#
#     with tf.device(devices[0]):
#         builder = ModelBuilder()
# 	print('num_classes: ' + str(num_classes))
#         net, logits, total_loss = alexnet_inference(builder, images, labels, num_classes)
#
#         if not is_train:
#             return alexnet_eval(net, labels)
#
#         global_step = builder.ensure_global_step()
# 	print('total_num_examples: ' + str(total_num_examples))
#         train_op = train(total_loss, global_step, total_num_examples)
#     return net, logits, total_loss, train_op, global_step


def distribute(images, labels, num_classes, total_num_examples, devices, is_train=True):
    def configure_optimizer(global_step, total_num_steps):
        """Return a configured optimizer"""

        def exp_decay(start, tgtFactor, num_stairs):
            decay_step = total_num_steps / (num_stairs - 1)
            decay_rate = (1 / tgtFactor) ** (1 / (num_stairs - 1))
            return tf.train.exponential_decay(start, global_step, decay_step, decay_rate,
                                              staircase=True)

        def lparam(learning_rate, momentum):
            return {
                'learning_rate': learning_rate,
                'momentum': momentum
            }

        return HybridMomentumOptimizer({
            'weights': lparam(exp_decay(0.001, 250, 4), 0.9),
            'biases': lparam(exp_decay(0.002, 10, 2), 0.9),
        })
    my_devices = devices
    builder = ModelBuilder(my_devices[-1])

    my_devices = my_devices[0:-1]
    if not is_train:
        net, logits, total_loss = alexnet_inference(builder, images, labels, num_classes)
        return alexnet_eval(net, labels)
    #with tf.device(builder.variable_device()):
    global_step = builder.ensure_global_step()
    gradients = list()
    with tf.device(builder.variable_device()):
        image_list = tf.split(images, len(my_devices), 0)
        label_list = tf.split(labels, len(my_devices), 0)
    optimizer = configure_optimizer(global_step, total_num_examples)
    # with tf.variable_scope('model') as var_scope:
    #
    #     for i in range(len(my_devices)):
    #         curr_device = my_devices[i]
    #         with tf.name_scope('device-{}'.format(i)) as name_scope:
    #             with tf.device(curr_device):
    #                 net, logits, total_loss = alexnet_inference(builder, image_list[i], label_list[i], num_classes,
    #                                                             name_scope)
    #                 test_gradients = opt.compute_gradients(total_loss)
    #     gradients.append(test_gradients)
    #     var_scope.reuse_variables()

    with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as var_scope:
    	with tf.name_scope('test') :
        	for i in range(len(my_devices)):
            		curr_device = my_devices[i]
            		with tf.name_scope('device-{}'.format(i)) as name_scope:
                		with tf.device(curr_device):
                    			net, logits, total_loss = alexnet_inference(builder, image_list[i], label_list[i], num_classes,
                                                                name_scope)
                    			test_gradients = optimizer.compute_gradients(total_loss)
        	gradients.append(test_gradients)
    with tf.device(builder.variable_device()):
        final_gradients = builder.average_gradients(gradients)
        apply_gradient_op = optimizer.apply_gradients(final_gradients, global_step=global_step)
        train_op = tf.group(apply_gradient_op, name='train')
    return net, logits, total_loss, train_op, global_step



