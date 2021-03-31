from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import datetime
from tensorflow.python.platform import tf_logging as logging
import os
import json

flags = tf.app.flags


tf.app.flags.DEFINE_string("ps_hosts", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_string("worker_hosts", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Worker or server index")
flags.DEFINE_string("job_name", "ps", "worker/ps")
flags.DEFINE_string("checkpointDir", None, "oss checkpointDir")
flags.DEFINE_string("files", None, "oss checkpointDir")
FLAGS = tf.app.flags.FLAGS

def model_fn(features, labels, mode):
    W = tf.Variable(tf.zeros([3, 2]))
    b = tf.Variable(tf.zeros([2]))
    pred = tf.matmul(features, W) + b

    loss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred,labels=labels))
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        opt = tf.train.GradientDescentOptimizer(0.05)
        train_op = opt.minimize(loss, global_step=global_step, name='train_op')
        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  loss=loss,
                  train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                  mode=mode,
                  loss=loss)
    else:
        raise ValueError(
            "Only TRAIN and EVAL modes are supported: %s" % (mode))

def decode_line(line):
    """Convert a csv line into a (features_dict,label) pair."""
    v1, v2, v3, v4 = tf.decode_csv(line, record_defaults=[[1.0]] * 4, field_delim=',')
    labels = tf.cast(v4, tf.int32)
    features = tf.stack([v1, v2, v3])
    return features, labels

def train_input_fn():


    dataset = tf.data.TextLineDataset(FLAGS.files.split(","))
    
    d = dataset.map(decode_line).shuffle(True).batch(128).repeat()
    return d

def eval_input_fn():
    dataset = tf.data.TextLineDataset(FLAGS.files.split(","))
    d = dataset.map(decode_line).batch(128)
    return d


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def main(unused_argv):
    
    tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
    logging.info("tf_config: %s", tf_config)
    task_config = tf_config.get('task', {})
    FLAGS.job_name = task_config.get('type')
    FLAGS.task_index = task_config.get('index')

    cluster_config = tf_config.get('cluster', {})
    ps_hosts = cluster_config.get('ps')
    worker_hosts = cluster_config.get('worker')

    ps_hosts_str = ','.join(ps_hosts)
    worker_hosts_str = ','.join(worker_hosts)
    FLAGS.ps_hosts = ps_hosts_str
    FLAGS.worker_hosts = worker_hosts_str
    
    FLAGS.files = os.environ.get('FILES')
    FLAGS.checkpointDir = os.environ.get('CHECKPOINTDIR')
    logging.info("files: %s", FLAGS.files)
    logging.info("checkpoint: %s", FLAGS.checkpointDir)
    '''
    if FLAGS.job_name == 'chief' and FLAGS.task_index == 0:
        new_tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
        new_tf_config['cluster']['ps'] = ["dist-mnist-aimaster-ps-0.fctest.svc:2222"]
        new_tf_config['cluster']['worker'] = []
        new_tf_config['cluster']['worker'].append("dist-mnist-aimaster-worker-0.fctest.svc:2222")
        new_tf_config['cluster']['worker'].append("dist-mnist-aimaster-worker-1.fctest.svc:2222")
        new_tf_config['cluster']['worker'].append("dist-mnist-aimaster-worker-2.fctest.svc:2222")
        new_tf_config['cluster']['worker'].append("dist-mnist-aimaster-worker-3.fctest.svc:2222")
        os.environ['TF_CONFIG_NEW'] = json.dumps(new_tf_config)
        os.environ['AIMASTER_ANALYZE'] = 'TRUE'
    '''
    strategy = tf.contrib.distribute.ParameterServerStrategy()
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    config = tf.estimator.RunConfig(train_distribute=strategy,
                                    session_config=sess_config,
                                    save_checkpoints_steps=1000,
                                    save_summary_steps=100)
    import aimaster.python.tf_estimator.estimator as am_estimator
    estimator = am_estimator.EstimatorPlus(
                    model_fn=model_fn,
                    config=config,
                    model_dir=FLAGS.checkpointDir)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, start_delay_secs=6, throttle_secs=1)
    import aimaster.python.tf_estimator.training as aimaster_estimator
    aimaster_estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print("done")

if __name__=="__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    env_dist = os.environ
    print(env_dist.get('TF_CONFIG'))
    tf.app.run()
