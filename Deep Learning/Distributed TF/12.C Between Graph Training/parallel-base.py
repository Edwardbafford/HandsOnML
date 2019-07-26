# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 15:41:33 2019

Framework for parallel execution, super barren but useful as a starting point
for expanding into a more complex system or even a framework overtop of TF

@author: Louie Bafford
"""

import os
import sys
import time
import numpy as np
from random import randint
from multiprocessing import Process
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import dropout

print('Starting script..')
sys.stdout.flush()
time.sleep(2)

# Cluster configuration
tasks = ["localhost:2222", "localhost:2223","localhost:2224", "localhost:2225"]
jobs = {"local": tasks}
cluster = tf.train.ClusterSpec(jobs)

model = 'C:\\Users\\Louie Bafford\\Repositories\\HandsOnML\\tensorflow\\models\\12_C_init_model.ckpt'
num_models = 2

# Feed data into input queues
def input_queues(n):
    # Load the data
    print('Loading data...')
    sys.stdout.flush()
    time.sleep(2)
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train.reshape(-1,28*28), x_test.reshape(-1,28*28)

    # Create queues    
    print('Creating queues..')
    sys.stdout.flush()
    time.sleep(2)
    with tf.device("/job:local/task:2"):
        q1 = tf.FIFOQueue(capacity=100000, dtypes=[tf.int32, tf.float32], 
                  shapes=[[5,],[5,28*28]], name='input_q1', 
                  shared_name='input_q1')
        x1 = tf.placeholder(tf.float32, shape=(5,28*28))
        y1 = tf.placeholder(tf.int32, shape=(5,))
        e1 = q1.enqueue((y1,x1))
        y_out_1, x_out_1 = q1.dequeue()

        q2 = tf.FIFOQueue(capacity=100000, dtypes=[tf.int32, tf.float32], 
                  shapes=[[5,],[5,28*28]], name='input_q2', 
                  shared_name='input_q2')
        x2 = tf.placeholder(tf.float32, shape=(5,28*28))
        y2 = tf.placeholder(tf.int32, shape=(5,))
        e2 = q2.enqueue((y2,x2))

    # TO DO -- Remove
    #server1 = tf.train.Server(cluster, job_name="local", task_index=0)
    server2 = tf.train.Server(cluster, job_name="local", task_index=1)
    server4 = tf.train.Server(cluster, job_name="local", task_index=3)
    
    #Push data to queues
    print('server 3 initiated..')
    sys.stdout.flush()
    time.sleep(2)
    server3 = tf.train.Server(cluster, job_name="local", task_index=2)
    sess3 = tf.Session(target=server3.target)
    
    #Wait for all servers to be initialized
    print('Push queues waiting...')
    sys.stdout.flush()
    time.sleep(20)
    
    print('Push to queues..')
    sys.stdout.flush()
    time.sleep(2)
    for i in range(n):
        r1 = randint(0,59995)
        r2 = randint(0,59995)
        sess3.run(e1, feed_dict={y1:y_train[r1:r1+5], x1:x_train[r1:r1+5]})
        sess3.run(e2, feed_dict={y2:y_train[r2:r2+5], x2:x_train[r2:r2+5]})

        print('Input queue -- {0} | y1 -- {1} | y2 -- {2}'
              .format(i + 1, y_train[r1:r1+5], y_train[r2:r2+5]))
        sys.stdout.flush()
        time.sleep(.5)
    
    print('Input queues returning')
    sys.stdout.flush()
    time.sleep(40)
    return



#Make NN predictions
def NN(task, model, n):
    #Create NN
    print('Creating NN{0}..'.format(task))
    sys.stdout.flush()
    time.sleep(2)
    with tf.device("/job:local/task:{0}".format(task)):
        is_training = tf.placeholder(tf.bool, shape=())
        X = tf.placeholder(tf.float32, shape=(None,28*28))
        y = tf.placeholder(tf.int32, shape=(None))
        X_drop = dropout(X,.5, is_training=is_training)
        he_init = tf.contrib.layers.variance_scaling_initializer()
        bn_params = {'is_training':is_training, 'decay':0.99, 
                       'updates_collections':None}
        with tf.contrib.framework.arg_scope([fully_connected], 
                                        weights_initializer=he_init, 
                                        activation_fn=tf.nn.elu, 
                                        normalizer_fn=batch_norm, 
                                        normalizer_params=bn_params):
            h1 = dropout(fully_connected(X_drop, 40, scope='h1_{0}'.format(task)))
            h2 = dropout(fully_connected(h1, 30, scope='h2_{0}'.format(task)))
            h3 = dropout(fully_connected(h2, 20, scope='h3_{0}'.format(task)))
            output = fully_connected(h3, 10, scope='output_{0}'.format(task), activation_fn=None)
        
        x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=output)
        loss = tf.reduce_mean(x_entropy)
        optimizer = tf.train.AdamOptimizer()
        grads = optimizer.compute_gradients(loss)
        train = optimizer.apply_gradients(grads, name='train_{0}'.format(task))
    
    print('server {0} initiated..'.format(task))
    sys.stdout.flush()
    time.sleep(2)
    server = tf.train.Server(cluster, job_name="local", task_index=task)
    sess = tf.Session(target=server.target)
    
    print('NN{0} waiting..'.format(task))
    sys.stdout.flush()
    time.sleep(20)
    
    if task == 0:
        #Save model weights
        print('NN{0} saving weights..'.format(task))
        sys.stdout.flush()
        time.sleep(2)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, os.path.join(model))

    print('NN{0} weights sample -- {1}'.format(
          task,
          sess.run([v for v in tf.global_variables() if v.name == 'h1_{0}/weights:0'.format(task)][0])[0][:5]
        )
    )
    sys.stdout.flush() 
    time.sleep(2)
    
    print('Dequeue from queue {0}...'.format(task + 1))
    sys.stdout.flush()
    time.sleep(5)    
    
    with tf.device("/job:local/task:2"):
        q1 = tf.FIFOQueue(capacity=100000, dtypes=[tf.int32, tf.float32], 
                          shapes=[[5,],[5,28*28]], name='input_q{0}'.format(task + 1), 
                          shared_name='input_q{0}'.format(task + 1))
        y1, x1 = q1.dequeue()
 
    for i in range(n):
        y_in, x_in = sess.run([y1, x1])
        sess.run(train, feed_dict={X:x_in, 
                                   y:y_in, 
                                   is_training:True})
        print('NN{0} push updates {1}'.format(task, i + 1))
        sys.stdout.flush()
        time.sleep(.5)
        
    print('NN{0} returning..'.format(task))
    sys.stdout.flush()
    time.sleep(2)
    return


# Only for parent process
if __name__ == '__main__':
    print('starting new processes!')
    input_queue_process = Process(target=input_queues, args=(10,))
    input_queue_process.start()
    nn1_process = Process(target=NN, args=(0,model,8))
    nn1_process.start()
    
    #Verify parallel behavior
    time.sleep(25)
    print('waiting for processes to be killed')
    time.sleep(75)
    
    #Kill the cluster
    print('Terminating')
    input_queue_process.terminate()
    nn1_process.terminate()


