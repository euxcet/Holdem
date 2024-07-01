import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import time
from .model import CnnPolicy


class Predictor:
    def __init__(self):
        self.action_dim = 9
        self.state_dim = 2193
        self.g = tf.Graph()
        with self.g.as_default():
            with tf.variable_scope("ppo"):
                self.model = CnnPolicy("ppo", self.state_dim, self.action_dim)
        self._init_sess()

    def _init_sess(self):
        cpu_num = 1
        config = tf.ConfigProto(device_count={"CPU": cpu_num},
                                inter_op_parallelism_threads=cpu_num,
                                intra_op_parallelism_threads=cpu_num,
                                log_device_placement=False)
        with self.g.as_default():
            self.init_saver = tf.train.Saver(tf.global_variables())
            self.sess = tf.Session(graph=self.g, config=config)
            self.sess.run(tf.global_variables_initializer())

    def init_model(self, model_path):
        with self.g.as_default():
            try:
                self.init_saver.restore(self.sess, save_path=model_path)
            except Exception as e:
                print("init_model failed %s" % e)
                return False
            return True

    def get_value(self, state):
        obs = state
        return self.sess.run(self.model.vpred,
                             feed_dict={self.model.ph_ob: [obs]})

    def get_prob(self, state):
        obs = state
        #obs = cal_feature(state, SHANTEN=False, GLOBAL_VALUE=True)
        return self.sess.run([self.model.prob, self.model.pi_logits, self.model.card_concat, self.model.player_bet],
                             feed_dict={self.model.ph_ob: [obs]})

    def get_sample_action(self, state):
        obs = state
        prob=self.sess.run(self.model.prob, feed_dict={self.model.ph_ob: [obs]})
        #print("prob actions :fold check call 0.5 0.75 1 1.5 2 allin",prob)
        return self.sess.run(self.model.action,
                             feed_dict={self.model.ph_ob: [obs]})

    def get_actions(self, state):
        prob, logits, card_concat, player_bet = self.get_prob(state)
        max_action = np.argmax(prob, axis=-1)
        return max_action[0]
