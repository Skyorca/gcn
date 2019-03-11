from gcn.inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    '''
    以上这个的意思是：假设keep_prob=0.5，原来0.3的<0.5应该是0，这里是0.3+0.5然后下取整变成0，等效的
    所以也许可以写成：dropout_mask=tf.random_uniform(noise_shape)<keep_prob
    '''
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob) #没有被丢弃的神经元要缩放为1/p


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y) # x is sparse and y is dense
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        '''
        确定每层的名字，以及是否记录
        '''
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower() #类的名字，对于Dense来说就是dense，对于GraphConvolution就是graphconvolution
            name = layer + '_' + str(get_layer_uid(layer))   #第几个XX（密度，图卷积）层
        self.name = name
        self.vars = {} # 因为这里只是一个接口，所以都是空的，之后继承的实类会有vars并且通过_log_vars()记录下来。
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False #默认非稀疏表示

    def _call(self, inputs): #子类要重写这个方法，所以这里随便返回一个值就行。Input是上一层传进来的结果，包括第一层输入层。
        return inputs

    def __call__(self, inputs): #这个类是可调用的，参数是Input(就是输入层，隐层1，...)
        with tf.name_scope(self.name):#每一层都是一个namescope?
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self): #是存变量，变量就是每层的weight和bias
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs) #python2的写法，调用父类的同名方法。
        #in python3: super().__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout'] #placeholder还能当成字典一样传进来？
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'): #提供共享可能
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support'] #???
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless: #featureless是没有特征矩阵嘛？难道是blog里说的用identity代表特征所以乘积等价于权重矩阵本身？
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)
        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


'''
Q:对dense和graphconv的输入x都是什么？
A: 是__call__传进来的输入
Q: 关于support的解释：
A: support 是A+I的对称正则化矩阵。但是由于A是稀疏的，所以正则化矩阵也是稀疏的。这里不用矩阵表示而是改用tuple的list:
[([x1,y1],value,type),([x2,y2],value,type),...]
所以权重也没法用一个整的矩阵来表示，而是拆成小份w1,w2,..,wn，都有相同的维度[in,out]，n就是非零元素的个数。
所以矩阵乘法XW被拆成了一个n-loop:
每次计算稀疏矩阵乘法X*wi，然后和A+I的稀疏表示第i个元素做乘法。
loop结束时把它们加和起来就得到了本层结果。
'''