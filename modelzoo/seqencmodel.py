import tensorflow as tf
import sys
import os
import configparser

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path,"../"))
from utils import convrnn
from tensorflow.contrib.rnn import LSTMStateTuple

from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

MODEL_GRAPH_NAME="graph.meta"
MODEL_CFG_FILENAME="params.ini"
ADVANCED_SUMMARY_COLLECTION_NAME="advanced_summaries"

tf.app.flags.DEFINE_string("modelfolder", None, "target location of graph on disk")

## hyper parameters ##
tf.app.flags.DEFINE_string("kernel", "(1,3,3)", "kernel of convolutions")
tf.app.flags.DEFINE_string("classkernel", "(3,3)", "kernelsize of final classification convolution")
tf.app.flags.DEFINE_string("cnn_activation", "leaky_relu", "activation function for convolutional layers ('relu' or 'leaky_relu' [default])")

tf.app.flags.DEFINE_boolean("bidirectional", True, "Bidirectional Convolutional RNN")
tf.app.flags.DEFINE_integer("convrnn_compression_filters", -1, "number of convrnn compression filters or (default) -1 for no compression")
tf.app.flags.DEFINE_string("convcell", "gru", "Convolutional RNN cell architecture ('gru' (default) or 'lstm')")
tf.app.flags.DEFINE_string("convrnn_kernel", "(3,3)", "kernelsize of recurrent convolution. default (3,3)")
tf.app.flags.DEFINE_integer("convrnn_filters", 24, "number of convolutional filters in ConvLSTM/ConvGRU layer")
tf.app.flags.DEFINE_float("recurrent_dropout_i", 1., "input keep probability for recurrent dropout (default no dropout -> 1.)")
tf.app.flags.DEFINE_float("recurrent_dropout_c", 1., "state keep probability for recurrent dropout (default no dropout -> 1.)")
tf.app.flags.DEFINE_integer("convrnn_layers", 1, "number of convolutional recurrent layers")
tf.app.flags.DEFINE_boolean("peephole", False, "use peephole connections at convrnn layer. only for lstm (default False)")
tf.app.flags.DEFINE_boolean("convrnn_normalize", True, "normalize with batchnorm at convrnn layer (default True)")
tf.app.flags.DEFINE_string("aggr_strat", "state", "aggregation strategie to reduce temporal dimension (either default 'state' or 'sum_output' or 'avg_output')")

tf.app.flags.DEFINE_float("learning_rate", 0.01, "Adam learning rate")
tf.app.flags.DEFINE_float("beta1", 0.9, "Adam beta1")
tf.app.flags.DEFINE_float("beta2", 0.999, "Adam beta2")
tf.app.flags.DEFINE_float("epsilon", 0.9, "Adam epsilon")

## expected data format ##
tf.app.flags.DEFINE_string("expected_datatypes",
                           "(tf.float32, tf.float32, tf.float32, tf.int64)", "expected datatypes")
tf.app.flags.DEFINE_integer("pix10m", 24, "number of 10m pixels")
tf.app.flags.DEFINE_integer("num_bands_10m", 10, "number of bands in 10 meter resolution (4)")
tf.app.flags.DEFINE_integer("num_classes", 4, "number of classes not counting unknown class -> e.g. 0:uk,1:a,2:b,3:c,4:d -> num_classes 4")

## performance ##
tf.app.flags.DEFINE_boolean("swap_memory", True, "Swap memory between GPU and CPU for recurrent layers")


FLAGS = tf.app.flags.FLAGS

class Model():
    def __init__(self):

        # used to feed training/testing or evaluation iterator as string representation
        self.iterator_handle = tf.placeholder(tf.string, shape=[],name="data_iterator_handle")
        # requires info to data dimensions
        #self.output_shapes = tuple(output_shapes)
        #self.output_types = tuple(output_types)

        # changes behavior of batch normalization
        self.is_train = tf.placeholder_with_default(tf.cast(True, tf.bool), shape=(), name="is_train")

        def parse(var,dtype):
            if type(var)==dtype: return var
            else: return eval(var)

        # see FLAGS for documentation
        self.kernelsize = parse(FLAGS.kernel,tuple)
        self.num_classes = parse(FLAGS.num_classes,int)

    def build_graph(self, modelpath=None):

        # input pipeline
        print("building input pipeline...")

        # input pipeline
        with tf.name_scope("input"):
            (x, sequence_lengths), (alllabels,) = self.input()
            self.alllabels = alllabels

        print("building inference...")
        self.logits = self.inference(input=(x, sequence_lengths))
        print('tf logits: ', tf.shape(self.logits))

        # reduce label size to same shape like logits (important for downsampling)
        b,w,h,d = self.logits.get_shape()

        # take first label -> assume labels do not change over timeseries
        first_labelmap = alllabels[:,0]

        # create one-hot labelmap from 0-num_classes
        labels = tf.one_hot(first_labelmap, self.num_classes+1)

        # mask out class 0 -> unknown
        unknown_mask = tf.cast(labels[:,:,:,0], tf.bool)
        not_unknown_mask = tf.logical_not(unknown_mask)

        # keep rest of labels
        self.labels = labels[:,:,:,1:]
        print('tf labels: ', tf.shape(self.labels))


        # mask out all classes labeled unknown (labelid=0)
        #not_unknown_mask = tf.not_equal(first_labelmap, tf.constant(0, dtype=tf.int32))

        #self.labels = self.reshape_labels(labels=first_labelmap, width=w, height=h)
        #self.labels = tf.one_hot(tf.reduce_mean(alllabels, axis=1), self.num_classes)

        print("building loss...")
        self.loss = self.loss(logits=self.logits, labels=self.labels, mask=not_unknown_mask,name="loss")

        print("building optimizer...")
        self.train_op = self.optimize(self.loss, name="train_op")

        print("building metrics and summaries...")
        self.metrics(logits=self.logits, labels=self.labels,mask=not_unknown_mask)

        self.summary_op = self.summary()

        n_trainparams = count_params(collector_fun=tf.trainable_variables)
        print("graph completed with {} trainable parameters".format(n_trainparams))

        if modelpath is not None:
            path = os.path.join(modelpath, MODEL_GRAPH_NAME)
            print("writing meta graph to {}".format(path))
            tf.train.export_meta_graph(filename=path)

            # write FLAGS to file
            cfg = configparser.ConfigParser()

            # TF 1.4
            #cfg["model"] = {"model": os.path.basename(__file__)}
            #cfg["flags"] = FLAGS.__dict__["__flags"]

            # TF 1.7
            flags_dict = dict()
            for name in FLAGS:
                flags_dict[name]=str(FLAGS[name].value)

            cfg["flags"] = flags_dict # FLAGS.__flags #  broke tensorflow=1.5


            path=os.path.join(modelpath, MODEL_CFG_FILENAME)
            print("writing parameters to {}".format(path))
            with open(path, 'w') as configfile:
                cfg.write(configfile)

            # write operation names to file for easier debugging
            path = os.path.join(modelpath, "nodes.txt")
            print("writing nodes list to {}".format(path))
            names = [n.name for n in tf.get_default_graph().as_graph_def().node]
            with open(path, 'w') as f:
                f.write("\n".join(names))

    def input(self):

        #output_shapes = tf.placeholder_with_default(self.output_shapes, [6], name="data_shapes")
        #os = [tf.TensorShape(s) for s in self.output_shapes]

        iterator = tf.data.Iterator.from_string_handle(self.iterator_handle,
                                                               output_types=eval(FLAGS.expected_datatypes))

        with tf.name_scope("raw"):
            x10, doy, year, labels = iterator.get_next()

            self.x10 = tf.cast(x10, tf.float32, name="x10")
            self.doy = tf.cast(doy, tf.float32, name="doy")
            self.year = tf.cast(year, tf.float32,name="year")
            self.y = tf.cast(labels, tf.int32,name="y")

        #self.x10.set_shape([2,46,24,24,5])

        # integer sequence lenths per batch for dynamic_rnn masking
        self.seq = sequence_lengths = tf.reduce_sum(tf.cast(self.x10[:, :, 0, 0, 0] > 0, tf.int32), axis=1, name="sequence_lengths")

        def resize(tensor, new_height, new_width):
            b = tf.shape(tensor)[0]
            t = tf.shape(tensor)[1]
            h = tf.shape(tensor)[2]
            w = tf.shape(tensor)[3]
            d = tf.shape(tensor)[4]
            print('b: {}, t: {}, h: {}, w: {}, d: {}'.format(b, t, h, w, d))

            # stack batch on times to fit 4D requirement of resize_tensor
            stacked_tensor = tf.reshape(tensor, [b * t, h, w, d])
            reshaped_stacked_tensor = tf.image.resize_images(stacked_tensor, size=(new_height, new_width))
            return tf.reshape(reshaped_stacked_tensor, [b, t, new_height, new_width, d])


        def expand3x(vector):
            vector = tf.expand_dims(vector, -1)
            vector = tf.expand_dims(vector, -1)
            vector = tf.expand_dims(vector, -1)
            return vector

        with tf.name_scope("reshaped"):
            print(self.x10.shape)
            b = tf.shape(self.x10)[0]
            t = tf.shape(self.x10)[1]
            px = tf.shape(self.x10)[2]

            #b,t,w,h,d = self.x10.shape()

            tf.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME,tf.identity(self.x10, name="x10"))

            # expand
            doymat = tf.multiply(expand3x(self.doy), tf.ones((b,t,px,px,1)),name="doy")
            yearmat = self.yearmat = tf.multiply(expand3x(self.year), tf.ones((b, t, px, px, 1)),name="year")

            print(tf.shape(doymat))
            print(tf.shape(yearmat))

            #x = tf.concat((self.x10,x20,x60,doymat,yearmat),axis=-1,name="x")
            x = tf.concat((self.x10,doymat,yearmat),axis=-1,name="x")
            tf.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, x)

            # set depth of x for convolutions
            depth = FLAGS.num_bands_10m + 2 # doy and year

            # dynamic shapes. Fill for debugging

            x.set_shape([None, None, FLAGS.pix10m, FLAGS.pix10m, depth])
            self.y.set_shape((None,None,FLAGS.pix10m,FLAGS.pix10m))

        return (x, sequence_lengths), (self.y,)

    def inference(self,input):

        x, sequence_lengths = input

        rnn_output_list=list()
        rnn_state_list = list()

        x_rnn=x
        for j in range(1,FLAGS.convrnn_layers+1):
            convrnn_kernel = eval(FLAGS.convrnn_kernel)
            x_rnn, state = convrnn_layer(input=x_rnn, is_train=self.is_train, filter=FLAGS.convrnn_filters,
                                     kernel=convrnn_kernel,
                                     bidirectional=FLAGS.bidirectional, convcell=FLAGS.convcell,
                                     sequence_lengths=sequence_lengths, scope="convrnn" + str(j))
            rnn_output_list.append(x_rnn)
            rnn_state_list.append(state)

        # concat outputs from cnns and rnns in a dense scheme
        x = tf.concat(rnn_output_list, axis=-1)

        # take concatenated states of last rnn block (might contain multiple conrnn layers)
        state = tf.concat(rnn_state_list, axis=-1)


        # use the cell state as featuremap for the classification step
        # cell state has dimensions (b,h,w,d) -> classification strategy
        if FLAGS.aggr_strat == 'state':
            class_input=state # shape (b,h,w,d)
            classkernel=eval(FLAGS.classkernel)
            logits = conv_bn_relu(input=class_input, is_train=self.is_train, filter=self.num_classes, kernel=classkernel, dilation_rate=(1,1), conv_fun=tf.layers.conv2d, var_scope="class")

        elif (FLAGS.aggr_strat == 'avg_output') or (FLAGS.aggr_strat == 'sum_output'):
            # last rnn output at each time t
            class_input = x_rnn  # shape (b,t,h,w,d)

            kernel = (1,FLAGS.classkernel[0],FLAGS.classkernel[1])

            # logits for each single timeframe
            logits = conv_bn_relu(input=class_input, is_train=self.is_train, filter=self.num_classes, kernel=kernel,
                         dilation_rate=(1, 1, 1), conv_fun=tf.layers.conv3d, var_scope="class")

            if FLAGS.aggr_strat == 'avg_output':
                # average logit scores at each observation
                # (b,t,h,w,d) -> (b,h,w,d)
                logits = tf.reduce_mean(logits,axis=1)
            elif FLAGS.aggr_strat == 'sum_output':
                # summarize logit scores at each observation
                # the softmax normalization later will normalize logits again
                # (b,t,h,w,d) -> (b,h,w,d)
                logits = tf.reduce_sum(logits,axis=1)

        else:
            raise ValueError("please provide valid aggr_strat flag ('state' or 'avg_output' or 'sum_output')")

        tf.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, logits)

        return logits

    def loss(self, logits, labels,mask,name):

        loss_per_px = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

        #loss_per_px = tf.boolean_mask(loss_per_px, unknown_mask, name="masked_loss_per_px")
        _ = tf.identity(loss_per_px,name="loss_per_px")
        _ = tf.identity(mask, name="mask_per_px")

        lpp = tf.boolean_mask(loss_per_px, mask)

        return tf.reduce_mean(lpp, name=name)

    def optimize(self, loss, name):

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.samples_seen = tf.Variable(0, name='samples_seen', trainable=False)

        batchsize = tf.shape(self.x10)[0]
        samples_seen_increment_op = tf.assign(self.samples_seen, self.samples_seen + batchsize)

        lr = tf.placeholder_with_default(FLAGS.learning_rate,shape=(),name="learning_rate")
        beta1 = tf.placeholder_with_default(FLAGS.beta1, shape=(), name="beta1")
        beta2 = tf.placeholder_with_default(FLAGS.beta2, shape=(), name="beta2")

        with tf.control_dependencies([samples_seen_increment_op]): # execute this every time global step is incremented
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=FLAGS.epsilon)
            return optimizer.minimize(loss, global_step=self.global_step, name=name)

    def metrics(self, logits, labels,mask):
        self.prediction_scores = tf.nn.softmax(logits=logits, name="prediction_scores")
        self.predictions = tf.argmax(self.prediction_scores, 3, name="predictions")

        targets = tf.argmax(labels, 3, name="targets")
        #correctly_predicted = tf.equal(self.predictions, targets, name="correctly_predicted")
        correctly_predicted = tf.equal(tf.boolean_mask(self.predictions, mask), tf.boolean_mask(targets,mask), name="correctly_predicted")
        self.overall_accuracy = tf.reduce_mean(tf.cast(correctly_predicted, tf.float32), name="overall_accuracy")

    def summary(self):
        """
        minimial summaries for training @ monitoring
        """

        tf.summary.scalar("oa", self.overall_accuracy)
        tf.summary.scalar("xe", self.loss)
        ## histograms

        return tf.summary.merge_all()

def convrnn_layer(input, filter, is_train=True, kernel=FLAGS.convrnn_kernel, sequence_lengths=None, bidirectional=True,
                  convcell='gru', scope="convrnn"):
    with tf.variable_scope(scope):

        x = input

        px = x.get_shape()[3]

        if FLAGS.convrnn_compression_filters > 0:
            x = conv_bn_relu(input=x, is_train=is_train, filter=FLAGS.convrnn_compression_filters, kernel=(1, 1, 1),
                             dilation_rate=(1, 1, 1), var_scope="comp")

        tf.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, tf.identity(input, "input"))

        if convcell == 'gru':
            cell = convrnn.ConvGRUCell((px, px), filter, kernel, activation=tf.nn.tanh, normalize=FLAGS.convrnn_normalize)

            # tf.Variable(tf.zeros((b,h,w)), validate_shape=False, trainable=False, name="zero_state")
            # zero_state = tf.Variable(tf.zeros((None, None, None), tf.float32), trainable=False)
        elif convcell == 'lstm':
            cell = convrnn.ConvLSTMCell((px, px), filter, kernel, activation=tf.nn.tanh, normalize=FLAGS.convrnn_normalize,peephole=FLAGS.peephole)
        else:
            raise ValueError("convcell argument {} not valid either 'gru' or 'lstm'".format(convcell))

        ## add dropout wrapper to cell
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=FLAGS.recurrent_dropout_i, state_keep_prob=FLAGS.recurrent_dropout_i)

        if bidirectional:
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell, inputs=x,
                                                                    sequence_length=sequence_lengths,
                                                                    dtype=tf.float32, time_major=False,
                                                                    swap_memory=FLAGS.swap_memory)

            concat_outputs = tf.concat(outputs, -1)
            if convcell == 'gru':
                concat_final_state = tf.concat(final_states, -1)
            elif convcell == 'lstm':
                fw_final, bw_final = final_states
                concat_final_state = LSTMStateTuple(
                    c=tf.concat((fw_final.c, bw_final.c), -1),
                    h=tf.concat((fw_final.h, bw_final.h), -1)
                )
                # else:
                #    concat_final_state = tf.concat((fw_final,bw_final),-1)
        else:
            concat_outputs, concat_final_state = tf.nn.dynamic_rnn(cell=cell, inputs=input,
                                                                   sequence_length=sequence_lengths,
                                                                   dtype=tf.float32, time_major=False)

        if convcell=='lstm':
            concat_final_state = concat_final_state.c

        tf.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, tf.identity(concat_outputs, name="outputs"))
        tf.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, tf.identity(concat_final_state, name="final_states"))

        return concat_outputs, concat_final_state

def conv_bn_relu(var_scope="name_scope", is_train=True, **kwargs):
    with tf.variable_scope(var_scope):

        if FLAGS.cnn_activation == 'relu':
            activation_function = tf.nn.relu
        elif FLAGS.cnn_activation == 'leaky_relu':
            activation_function = tf.nn.leaky_relu
        else:
            raise ValueError("please provide valid 'cnn_activation' FLAG. either 'relu' or 'leaky_relu'")

        x = conv_layer(**kwargs)
        x = Batch_Normalization(x, is_train, "bn")
        x = activation_function(x)

        tf.add_to_collection(ADVANCED_SUMMARY_COLLECTION_NAME, x)
        return x

def count_params(collector_fun=tf.trainable_variables):
    size = lambda v: reduce(lambda x, y: x * y, v.get_shape().as_list())
    n = sum(size(v) for v in collector_fun())
    return n

def conv_layer(input, filter, kernel, dilation_rate=(1, 1, 1), stride=1, conv_fun=tf.layers.conv3d, layer_name="conv"):

    with tf.name_scope(layer_name):

        # pad input to required sizes for same output dimensions
        input = pad(input,kernel,dilation_rate,padding="REFLECT")

        network = conv_fun(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,padding='VALID', dilation_rate=dilation_rate)

        return network

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True):
        return tf.cond(training,
                       lambda: batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda: batch_norm(inputs=x, is_training=training, reuse=True))


def pad(input,kernel,dilation,padding="REFLECT"):
    """https://www.tensorflow.org/api_docs/python/tf/pad"""

    # determine required padding sizes
    def padsize(kernel, dilation):
        p = []
        for k, d in zip(kernel, dilation):
            p.append(int(k / 2) * d)
        return p

    padsizes = padsize(kernel, dilation)

    # [bleft,bright], [tleft,tright], [hleft,hright], [wleft,wright],[dleft,dright]
    paddings = tf.constant([[0, 0]] + [[p, p] for p in padsizes] + [[0, 0]])

    return tf.pad(input, paddings, padding)

def test2():
    from Dataset import Dataset
    import matplotlib.pyplot as plt
    import numpy as np

    dataset = Dataset(datadir="/media/data/marc/tfrecords/fields/L1C/240", verbose=True, temporal_samples=30,section='2016')

    training_dataset, output_shapes, output_datatypes, fm_train = dataset.create_tf_dataset("eval", fold=0, batchsize=2, shuffle=False, prefetch_batches=1, num_batches=-1, threads=8, drop_remainder=False)

    iterator = training_dataset.make_initializable_iterator()

    model = Model()
    model.build_graph(modelpath=FLAGS.modelfolder)

    def get_operation(name):
        return tf.get_default_graph().get_operation_by_name(name).outputs[0]

    iterator_handle_op = get_operation("data_iterator_handle")
    is_train_op = get_operation("is_train")

    mpp_op = get_operation("mask_per_px")
    lpp_op = get_operation("loss_per_px")
    #umlpp_op = get_operation("unmasked_loss_per_px")

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        sess.run([iterator.initializer, tf.tables_initializer()])


        data_handle = sess.run(iterator.string_handle())

        feed_dict = {iterator_handle_op: data_handle, is_train_op: False}

        y,labels,alllabels,mpp,lpp = sess.run([model.y,model.labels,model.alllabels,mpp_op,lpp_op],feed_dict=feed_dict)
        #x10, x20, x60, doy, year, labels = sess.run(iterator.get_next())
        a = sess.run(tf.one_hot(alllabels-1, model.num_classes))

        plt.imshow(y[0,0])

def test(_):

    FLAGS.pix10 = 24
    FLAGS.modelfolder="tmp"
    FLAGS.convcell = "lstm"
    model = Model()
    model.build_graph(modelpath=FLAGS.modelfolder)

def main(_):
    #test2()

    model = Model()
    model.build_graph(modelpath=FLAGS.modelfolder)

if __name__ == '__main__':

    tf.app.run()




