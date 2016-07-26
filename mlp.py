import numpy as np
import tensorflow as tf

class MLP:

  num_classes = 3
  hidden_size = num_classes
  input_size = 4
  batch_size = 30
  iterations = 10000
  learning_rate = 1e-4
  keep_prob = 1.0
  lam = 0.0
  class_bias = [0.0, 0.0]


  def __init__(self):
    pass

  def set_config(self, config):
    self.num_classes = config.get('num_classes', self.num_classes)
    self.input_size = config.get('input_size', self.input_size)
    self.hidden_size = config.get('hidden_size', self.hidden_size)
    self.batch_size = config.get('batch_size', self.batch_size)
    self.iterations = config.get('iterations', self.iterations)
    self.learning_rate = config.get('learning_rate', self.learning_rate)
    self.keep_prob = config.get('keep_prob', self.keep_prob)
    self.class_bias = config.get('class_bias', self.class_bias)
    self.lam = config.get('lam', self.lam)
    

  @staticmethod
  def _weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  @staticmethod
  def merge_matrices(a, b):
    m = {'TP':0,'FP':0,'TN':0,'FN':0}
    for key in a.keys():
      m[key] = a[key] + b[key]
    return m

  @staticmethod
  def _build_conf_matrix(y, y_):
    '''
    positive => unnatural
    negative => natural
    '''
    matrix = {"TP":0, "FP":0, "TN":0, "FN":0}
    am = np.argmax
    for i in range(y.shape[0]):
      if (am(y[i]) == am(y_[i]) == 1):
        matrix["TP"] += 1
      elif (am(y[i]) == 1 and am(y[i]) != am(y_[i])):
        matrix["FP"] += 1
      elif (am(y[i]) == am(y_[i]) == 0):
        matrix["TN"] += 1
      elif (am(y[i]) == 0 and am(y[i]) != am(y_[i])):
        matrix["FN"] += 1
    
    return matrix

  def run(self, x_data, y_data,
      print_prog=False):
    '''
    x_data and y_data should be a tuple of (train, test)
    '''
    sess = tf.InteractiveSession()
    x  = tf.placeholder(tf.float32, shape=(None, self.input_size))
    y_ = tf.placeholder(tf.float32, shape=(None, self.num_classes))
    keep_prob = tf.placeholder("float")

    
    W_1 = MLP._weight_variable((self.input_size,self.hidden_size))
    b_1 = MLP._bias_variable((self.hidden_size,))
    h_1 = tf.matmul(x,W_1) + b_1

    #h_1 = tf.nn.dropout(tf.matmul(x,W_1) + b_1, keep_prob=self.keep_prob)*(1/keep_prob)

    W_2 = MLP._weight_variable((self.hidden_size, self.num_classes))
    b_2 = MLP._bias_variable((self.num_classes,))
    
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #y = tf.nn.dropout(tf.nn.softmax(tf.matmul(h_1,W_2) + b_2), keep_prob)*(1/keep_prob)
    y = tf.nn.softmax(tf.matmul(h_1,W_2) + b_2)

    all_weights = tf.concat(0, [tf.reshape(W_1, [-1]), tf.reshape(W_2, [-1])])
    weight_decay = self.lam\
        *tf.reduce_sum(all_weights**2.0)/(2.0*tf.to_float(tf.size(all_weights)))

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    #false_positives = tf.slice(tf.matmul(tf.transpose(y),y_), [1,0], [1,1])
    #false_negatives = tf.slice(tf.matmul(tf.transpose(y),y_), [0,1], [1,1])


    cost = cross_entropy + weight_decay
    train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

    sess.run(tf.initialize_all_variables())

    t_class_bias = tf.constant(self.class_bias)
    correct = tf.equal(tf.argmax(y+t_class_bias,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    mean_error = tf.reduce_mean(abs(y-y_))

    for i in range(self.iterations):
      batch = (x_data[0].take(range(i,i+self.batch_size), axis=(0), mode='wrap')
        , y_data[0].take(range(i,i+self.batch_size), axis=(0), mode='wrap'))
      
      train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:self.keep_prob})
      if print_prog and not i % 1000:
        print(str(i) + "/" + str(self.iterations))
        print(accuracy.eval(feed_dict={x: x_data[0][:], y_: y_data[0][:], keep_prob:1.0}))

    confusion = MLP._build_conf_matrix((y+t_class_bias).eval(feed_dict={x: x_data[1][:], y_: y_data[1][:], keep_prob:1.0}), y_data[1][:])
    if print_prog:
      test_acc = accuracy.eval(feed_dict={x: x_data[1][:], y_: y_data[1][:], keep_prob:1.0})
      print("Test acc: " + str(test_acc))
    return confusion

