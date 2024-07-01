import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from functools import reduce

USE_BN = False
DECAY = 0.99
INPUT_NORM = [0.024, 6.498]


class CnnPolicy():
    def __init__(self, scope, ob_space, ac_space):

        self.ob_space = ob_space
        self.ac_space = ac_space

        self.ph_ob = tf.placeholder(dtype=tf.float32, shape=[None, self.ob_space], name='ph_ob')
        self.ph_ac = tf.placeholder(dtype=tf.int32, shape=[None], name='ph_ac')

        self.is_training = True
        self.data_format = 'NHWC'

        self.initializer = tf.variance_scaling_initializer(scale=1.0, mode='fan_in', distribution='truncated_normal')
        self.MACs = []
        self.MEMs = []
        self.W = []
        self.ACTIVATION = 'leaky'
        self.BOTTLENECK = True
        self.REPEAT = 1
        self.GLOBAL_VALUE = False

        self.apply_policy()

    def get_shape(self, x):
        return x.get_shape().as_list()

    def mul_all(self, mul_list):
        return reduce(lambda x, y: x * y, mul_list)

    def activation(self, x):
        if self.ACTIVATION == 'prelu':
            if len(self.get_shape(x)) > 2:
                shared_axes = [1, 2]
            else:
                shared_axes = None
            return tf.keras.layers.PReLU(alpha_initializer=tf.initializers.constant(0.25), shared_axes=shared_axes)(x)
        elif self.ACTIVATION == 'leaky':
            return tf.nn.leaky_relu(x)
        elif self.ACTIVATION == 'relu':
            return tf.nn.relu(x)
        else:
            assert -1

    def get_variable(self, shape, name, initializer=None, trainable=True):
        if initializer is None:
            initializer = self.initializer
        return tf.get_variable(name=name, shape=shape, initializer=initializer, trainable=trainable)

    def window(self, stride_or_ksize):
        if isinstance(stride_or_ksize, int):
            stride_or_ksize = [stride_or_ksize] * 2
        return [1, 1] + stride_or_ksize if self.data_format == 'NCHW' \
            else [1] + stride_or_ksize + [1]

    def reshape(self, x, shape=None):
        if shape is None:
            shape = [self.mul_all(self.get_shape(x)[1:])]
        shape = [-1] + shape
        x = tf.reshape(x, shape)
        return x

    def fc(self, x, c_out, bias=False, append=False, name='fc'):
        c_in = self.get_shape(x)[-1]
        W = self.get_variable([c_in, c_out], name)
        if append: self.W.append(W)
        x = tf.matmul(x, W)
        if isinstance(bias, bool):
            if bias:
                b = self.get_variable([c_out], name + '_bias', initializer=tf.constant_initializer(0.0))
                x = x + b
        elif isinstance(bias, float):
            b = self.get_variable([c_out], name + '_bias', initializer=tf.constant_initializer(bias))
            x = x + b

        MACs = c_in * c_out
        MEMs = c_out
        self.MACs.append([name, MACs])
        self.MEMs.append([name, MEMs])

        return x

    def conv(self, x, ksize, c_out=None, stride=1, padding='SAME', bias=False, append=False, name='conv'):
        data_format = self.data_format
        shape_in = self.get_shape(x)
        c_in = shape_in[1] if data_format == 'NCHW' else shape_in[-1]
        if c_out is None: c_out = c_in
        if isinstance(ksize, int):
            ksize = [ksize, ksize]
        W = self.get_variable(ksize + [c_in, c_out], name)
        if append: self.W.append(W)
        x = tf.nn.conv2d(x, W, self.window(stride), padding=padding, data_format=data_format, name=name)
        if bias:
            b = self.get_variable([c_out], name + '_b', initializer=tf.initializers.zeros)
            x = tf.nn.bias_add(x, b, data_format=data_format)

        shape_out = self.get_shape(x)
        MEMs = self.mul_all(shape_out[1:])
        MACs = c_in * ksize[0] * ksize[1] * MEMs
        self.MACs.append([name, MACs])
        self.MEMs.append([name, MEMs])

        return x

    def depthwise_conv(self, x, ksize, channel_multiplier=1, stride=1, padding='SAME', name='depthwise_conv'):
        data_format = self.data_format
        shape_in = self.get_shape(x)
        c_in = shape_in[1] if data_format == 'NCHW' else shape_in[-1]

        W = self.get_variable([ksize, ksize, c_in, channel_multiplier], name)
        x = tf.nn.depthwise_conv2d(x, W, self.window(stride), padding=padding, data_format=data_format, name=name)

        shape_out = self.get_shape(x)
        MEMs = self.mul_all(shape_out[1:])
        MACs = ksize * ksize * MEMs
        self.MACs.append([name, MACs])
        self.MEMs.append([name, MEMs])

        return x

    def conv_2x3(self, x, ksize, c_out=None, stride=1, padding='SAME', bias=False, append=False, name='conv_2x3'):
        assert ksize == [2, 3]
        x0, x1 = tf.split(x, 2, axis=-1 if self.data_format == 'NHWC' else 1)
        if self.data_format == 'NHWC':
            pad0 = tf.constant([[0, 0], [1, 0], [1, 1], [0, 0]])
            pad1 = tf.constant([[0, 0], [0, 1], [1, 1], [0, 0]])
        else:
            pad0 = tf.constant([[0, 0], [0, 0], [1, 0], [1, 1]])
            pad1 = tf.constant([[0, 0], [0, 0], [0, 1], [1, 1]])
        x0 = tf.pad(x0, paddings=pad0)
        x1 = tf.pad(x1, paddings=pad1)
        x = tf.concat([x0, x1], axis=-1 if self.data_format == 'NHWC' else 1)
        x = self.conv(x, [2, 3], c_out=c_out, stride=stride, padding='VALID', bias=bias, append=append, name=name)
        return x

    def batch_norm(self, x, center=True, scale=True, decay=DECAY, epsilon=1e-3):
        if not USE_BN:
            return x
        x = tf.layers.batch_normalization(
            x,
            axis=-1 if self.data_format == 'NHWC' else 1,
            momentum=decay,
            epsilon=epsilon,
            center=center,
            scale=scale,
            training=self.is_training
        )

        shape_out = self.get_shape(x)
        MEMs = self.mul_all(shape_out[1:])
        self.MEMs.append(['batch_norm', MEMs])
        return x

    def count_parameters(self):
        dict_parameters = {}

        def dict_add(key, num):
            if key not in dict_parameters.keys():
                dict_parameters[key] = 0
            dict_parameters[key] += num

        key_list = ['batch_norm', 'conv', 'fc', 'emb', 'p_re_lu']

        for var in tf.trainable_variables():
            print(var.device, var.op.name, var.shape.as_list())
            name_lowcase = var.op.name.lower()
            num = reduce(lambda x, y: x * y, var.get_shape().as_list())

            has_key = False
            for key in key_list:
                if key in name_lowcase:
                    dict_add(key, num)
                    has_key = True
                    break
            if not has_key:
                dict_add(key, num)

        total = 0
        for _, value in dict_parameters.items():
            total += value
        print('Parameters:', total, dict_parameters)

        return dict_parameters

    def count_MACs(self):
        dict_MACs = {}

        def dict_add(key, num):
            if key not in dict_MACs.keys():
                dict_MACs[key] = 0
            dict_MACs[key] += num

        key_list = ['conv', 'fc', 'emb']
        total = 0
        for MAC in self.MACs:
            for key in key_list:
                if key in MAC[0]:
                    dict_add(key, MAC[1])
                    break
            total += MAC[1]
        print('MACs:', total, dict_MACs)
        return total

    def count_MEMs(self):
        total = 0
        for MEM in self.MEMs:
            total += MEM[1]
        total = total * self.batch_size * 4 // (1024 * 1024)
        print('MEMs:', total)
        return total

    # pre activation
    def resnet_v2_card(self, x, stage, repeat, channel=None, bottleneck=False):
        H = self.get_shape(x)[1 if self.data_format == 'NHWC' else 2]
        if H == 1:
            kernel = [[1, 3]] * stage
            if channel is not None:
                pass
            elif bottleneck:
                channel = [768] * stage + [64]
            else:
                channel = [256] * stage + [64]
        elif H == 4:
            kernel = [[3, 3]] * stage
            if channel is not None:
                pass
            elif bottleneck:
                channel = [512] * stage + [64]
            else:
                channel = [150] * stage + [64]
        else:
            assert -1

        # pre-activation residual block
        def residual(x, c_out, kernel, bottleneck=False):
            shortcut = x
            x = self.batch_norm(x)
            x = self.activation(x)

            if bottleneck:
                with tf.variable_scope('C0'):
                    x = self.conv(x, 1, c_out // 4)
                    x = self.batch_norm(x)
                    x = self.activation(x)
                with tf.variable_scope('C1'):
                    x = self.conv(x, kernel, c_out // 4)
                    x = self.batch_norm(x)
                    x = self.activation(x)
                with tf.variable_scope('C2'):
                    x = self.conv(x, 1, c_out)
            else:
                with tf.variable_scope('C0'):
                    x = self.conv(x, kernel, c_out)
                    x = self.batch_norm(x)
                    x = self.activation(x)
                with tf.variable_scope('C1'):
                    x = self.conv(x, kernel, c_out)
            return x + shortcut

        with tf.variable_scope('init'):
            # x = INPUT_NORM[1] * (x - INPUT_NORM[0])
            x = self.conv(x, 1, channel[0])

        for i in range(stage):
            for j in range(repeat):
                with tf.variable_scope('S%dR%d' % (i, j)):
                    x = residual(x, channel[i], kernel[i], bottleneck=bottleneck)

        x = self.batch_norm(x)
        x = self.activation(x)

        with tf.variable_scope('final'):
            x = self.conv(x, 1, channel[-1])
            x = self.batch_norm(x)
            x = self.activation(x)

        x = tf.layers.flatten(x)
        return x

    def resnet_v2_bet(self, x, stage, repeat, channel=None, bottleneck=False):
        H = self.get_shape(x)[1 if self.data_format == 'NHWC' else 2]
        if H == 1:
            kernel = [[1, 3]] * stage
            if channel is not None:
                pass
            elif bottleneck:
                channel = [768] * stage + [64]
            else:
                channel = [256] * stage + [64]
        elif H == 4:
            kernel = [[3, 3]] * stage
            if channel is not None:
                pass
            elif bottleneck:
                channel = [512] * stage + [64]
            else:
                channel = [150] * stage + [64]
        else:
            assert -1

        # pre-activation residual block
        def residual(x, c_out, kernel, bottleneck=False):
            shortcut = x
            x = self.batch_norm(x)
            x = self.activation(x)

            if bottleneck:
                with tf.variable_scope('C0'):
                    x = self.conv(x, 1, c_out // 4)
                    x = self.batch_norm(x)
                    x = self.activation(x)
                with tf.variable_scope('C1'):
                    x = self.conv(x, kernel, c_out // 4)
                    x = self.batch_norm(x)
                    x = self.activation(x)
                with tf.variable_scope('C2'):
                    x = self.conv(x, 1, c_out)
            else:
                with tf.variable_scope('C0'):
                    x = self.conv(x, kernel, c_out)
                    x = self.batch_norm(x)
                    x = self.activation(x)
                with tf.variable_scope('C1'):
                    x = self.conv(x, kernel, c_out)
            return x + shortcut

        with tf.variable_scope('init'):
            # x = INPUT_NORM[1] * (x - INPUT_NORM[0])
            x = self.conv(x, 1, channel[0])

        for i in range(stage):
            for j in range(repeat):
                with tf.variable_scope('S%dR%d' % (i, j)):
                    x = residual(x, channel[i], kernel[i], bottleneck=bottleneck)

        x = self.batch_norm(x)
        x = self.activation(x)
        with tf.variable_scope('final'):
            x = self.conv(x, 1, channel[-1])
            x = self.batch_norm(x)
            x = self.activation(x)
        x = tf.layers.flatten(x)
        return x

    def mask_logits(self, logits):
        legal_action_flag_list_max_mask = (1.0 - self.legal_action) * tf.pow(10.0, 20.0)
        p_logits_after_mask = logits - legal_action_flag_list_max_mask
        p_logits_after_mask = tf.identity(p_logits_after_mask, "policy_result_after_mask")
        return p_logits_after_mask

    def _prob(self, pi_logits):
        with tf.variable_scope('prob'):
            return tf.nn.softmax(pi_logits, axis=-1)

    def _sample(self, prob):
        prob = tf.reshape(prob, [-1, self.ac_space])
        return tf.reshape(tf.multinomial(tf.log(prob), 1), [-1])

    def neg_log_prob(self, action, name):
        action = tf.cast(action, tf.int32)
        one_hot_actions = tf.one_hot(action, self.ac_space)
        return tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.pi_logits,
            # logits=self.logits_no_mask,
            labels=one_hot_actions,
            dim=-1,
            name=name)

    def _calc_entropy_loss(self):
        pi_logits = self.pi_logits
        logits = pi_logits - tf.reduce_max(pi_logits, axis=-1, keepdims=True)
        exp_logits = tf.exp(logits)
        exp_logits_sum = tf.reduce_sum(exp_logits, axis=-1, keepdims=True)
        p = exp_logits / exp_logits_sum
        temp_entropy_loss = tf.reduce_sum(p * (tf.log(exp_logits_sum) - logits), axis=-1)
        self.entropy_loss = tf.reduce_mean(temp_entropy_loss, name="entropy_loss")

    # @staticmethod
    def apply_policy(self):
        feature = self.ph_ob

        cards, player_bet, legal_action = \
            tf.split(feature, [312, 1872, 9],
                     axis=-1)        
        self.cards = cards
        card_concat = tf.reshape(cards, [-1, 4, 13, 6])
        self.card_concat = tf.identity(card_concat, "card_concat")

        player_bet=tf.reshape(player_bet,[-1, 4, 9, 52])
        self.player_bet = tf.identity(player_bet, "player_bet")

        legal_action = tf.reshape(legal_action, [-1, 9])
        self.legal_action = tf.identity(legal_action, "legal_action")

        with tf.variable_scope('card'):
            self.card_concat = self.resnet_v2_card(self.card_concat,
                                              stage=3,
                                              repeat=self.REPEAT,
                                              bottleneck=self.BOTTLENECK)
        with tf.variable_scope('bet'):
            self.player_bet = self.resnet_v2_bet(self.player_bet,
                                              stage=3,
                                              repeat=self.REPEAT,
                                              bottleneck=self.BOTTLENECK)

        x_concat = tf.concat([self.card_concat, self.player_bet], axis=1, name='x_concat')


        with tf.variable_scope('concat'):
            x_concat = self.fc(x_concat, 1024, name="fc1")
            #x_concat = self.batch_norm(x_concat)
            x_concat = self.activation(x_concat)
            x_concat = self.fc(x_concat, 1024, name="fc2")
            #x_concat = self.batch_norm(x_concat)
            x_concat = self.activation(x_concat)
            self.x_concat = x_concat

        with tf.variable_scope("policy"):
            x = self.fc(self.x_concat, self.ac_space, bias=True, name="fc_last")
            self.logits_no_mask = x
            self.pi_logits = self.mask_logits(x)

        with tf.variable_scope("value"):
            x_value = self.x_concat

            self.vpred = self.fc(x_value, 1, bias=True, append=True, name="fc_last")

        self.pi_logits = tf.reshape(self.pi_logits, [-1, 9])
        self.vpred = tf.reshape(self.vpred, [-1])

        self.prob = self._prob(self.pi_logits)

        self.action = self._sample(self.prob)
        self.action = tf.reshape(self.action, [-1])

        self.nlp = self.neg_log_prob(self.action, "neg_log_pi_old")
        self.nlp = tf.reshape(self.nlp, [-1])

        self.nlpac = self.neg_log_prob(self.ph_ac, "neg_log_pi")
        self.nlpac = tf.reshape(self.nlpac, [-1])

        self._calc_entropy_loss()

    def call(self, dict_obs):
        feed = {self.ph_ob: dict_obs}
        ac, vpred, nlp = tf.get_default_session().run(
            [self.action, self.vpred, self.nlp],
            feed_dict={**feed})
        return ac, vpred, nlp
