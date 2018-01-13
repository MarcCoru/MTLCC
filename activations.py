import os
import tensorflow as tf

from Dataset import Dataset
from PIL import Image
import argparse
import numpy as np

MODEL_GRAPH_NAME = "graph.meta"
TRAINING_IDS_IDENTIFIER = "train"
TESTING_IDS_IDENTIFIER = "test"

MODEL_CFG_FILENAME = "params.ini"
MODEL_CFG_FLAGS_SECTION = "flags"
MODEL_CFG_MODEL_SECTION = "model"
MODEL_CFG_MODEL_KEY = "model"

MODEL_CHECKPOINT_NAME = "model.ckpt"
TRAINING_SUMMARY_FOLDER_NAME = "train"
TESTING_SUMMARY_FOLDER_NAME = "test"

"""
    

"""

def main():
    parser = argparse.ArgumentParser(description='Recreate RNN steps and write out activations')
    parser.add_argument('model', type=str, help='path to model')
    parser.add_argument('datadir', type=str, help='dataset directory')
    parser.add_argument('outdir', type=str, default="2016",
                        help='directory to dump png files')
    parser.add_argument('-d','--dataset', type=str, default="2016" , help='dataset within the dataset directors (default 2016)')
    parser.add_argument('-p', '--partition', type=str, default="test",
                        help='dataset partition (train, test or eval) default:test')
    parser.add_argument('-t', '--tile', type=int, default=None,
                        help='tileid to calculate activations on')


    args = parser.parse_args()

    datadir = args.datadir
    modeldir=args.model
    outfolder=args.outdir

    batchsize = 1

    dataset = Dataset(datadir=datadir,
                      verbose=True,
                      temporal_samples=None,
                      section=args.dataset)

    tfdataset, _, _, filenames = dataset.create_tf_dataset(args.partition, 0, batchsize, True, overwrite_ids=[args.tile])
    iterator = tfdataset.make_initializable_iterator()

    config = tf.ConfigProto()
    sess = tf.InteractiveSession()

    data_handle = sess.run(iterator.string_handle())

    # train_writer = tf.summary.FileWriter(os.path.join(args.modeldir, TRAINING_SUMMARY_FOLDER_NAME), sess.graph)
    # test_writer = tf.summary.FileWriter(os.path.join(args.modeldir, TESTING_SUMMARY_FOLDER_NAME))

    sess.run([iterator.initializer])

    graph = os.path.join(modeldir, MODEL_GRAPH_NAME)

    _ = tf.train.import_meta_graph(graph)

    saver = tf.train.Saver(save_relative_paths=True)

    checkpoint = os.path.join(modeldir, MODEL_CHECKPOINT_NAME)

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])

    latest_ckpt = tf.train.latest_checkpoint(modeldir)
    if latest_ckpt is not None:
        print "restoring from " + latest_ckpt
        saver.restore(sess, latest_ckpt)

    def get_op(name):
        return tf.get_default_graph().get_operation_by_name(name).outputs[0]

    ## get variables from tf.default_graph
    iterator_handle_op = get_op("data_iterator_handle")
    is_train_op = get_op("is_train")
    global_step_op = get_op("global_step")
    train_op = get_op("train_op")

    query_map = dict()

    # atrousdeep generation
    query_map["x"] = "input/reshaped/x"

    if "atrousdeep" in modeldir:
        query_map["convrnn_input"] = "convrnn/input"
        query_map["convrnn_output"] = "convrnn/outputs"
        query_map["convrnn_state"] = "convrnn/final_states"
    else:
        query_map["convrnn_input"] = "convrnn1/input"
        query_map["convrnn_output"] = "convrnn1/outputs"
        query_map["convrnn_state"] = "convrnn1/final_states"

    # query_map["comp1"]="dense1/comp1/LeakyRelu/Maximum"
    # query_map["conv1"]="dense1/1/LeakyRelu/Maximum"
    # query_map["comp6"]="dense1/comp6/LeakyRelu/Maximum"
    # query_map["conv6"]="dense1/6/LeakyRelu/Maximum"

    # query_map["class"]="class/LeakyRelu/Maximum"
    query_map["targets"] = "targets"
    query_map["predictions"] = "predictions"
    query_map["prediction_scores"] = "prediction_scores"
    query_map["correctly_predicted"] = "correctly_predicted"

    feed = {iterator_handle_op:data_handle,is_train_op:False}

    operations = ops = [get_op(query_map[key]) for key in sorted(query_map.keys())]

    queried = sess.run(operations,feed_dict=feed)

    results=dict()
    for key, array in zip(sorted(query_map.keys()),queried):
        results[key]=array

    b, t, px, px, d_in = results["convrnn_input"].shape



    x = results["convrnn_input"][:, 0]

    scope = "convrnn1/bidirectional_rnn/fw/conv_lstm_cell"

    # state = sess.run(zero_state_op)

    weights = []
    weights.append(get_op(scope + "/kernel"))
    # weights.append(get_op(scope+"/W_ci")) # peephole
    # weights.append(get_op(scope+"/W_cf")) # peephole
    # weights.append(get_op(scope+"/W_co")) # peephole
    weights.append(get_op(scope + "/LayerNorm/beta"))
    weights.append(get_op(scope + "/LayerNorm/gamma"))
    weights.append(get_op(scope + "/LayerNorm_1/beta"))
    weights.append(get_op(scope + "/LayerNorm_1/gamma"))
    weights.append(get_op(scope + "/LayerNorm_2/beta"))
    weights.append(get_op(scope + "/LayerNorm_2/gamma"))
    weights.append(get_op(scope + "/LayerNorm_3/beta"))
    weights.append(get_op(scope + "/LayerNorm_3/gamma"))
    weights.append(get_op(scope + "/LayerNorm_4/beta"))
    weights.append(get_op(scope + "/LayerNorm_4/gamma"))

    inputs = results["convrnn_input"]

    b, t, px, px, d = results["convrnn_output"].shape
    convfilters = d / 2
    zero_state_op = tf.contrib.rnn.LSTMStateTuple(c=tf.zeros(tf.TensorShape([b, px, px, convfilters])),
                                                  h=tf.zeros(tf.TensorShape([b, px, px, convfilters])))
    state = sess.run(zero_state_op)

    jGate = []
    iGate = []
    fGate = []
    oGate = []
    outputs = []
    states = []
    statesh = []

    # execute on cpu because ressource exhausted error on GPU
    with tf.device('/cpu:0'):

        for time in range(0, t):
            print("lstm iteration time: {}".format(time))

            h, state, j, i, f, o = lstm(inputs[:, time], state, weights, convfilters)
            state = tf.contrib.rnn.LSTMStateTuple(c=state.c.eval(), h=state.h.eval())

            # show_gray(i.eval(),"input_gate at t{}".format(it))

            iGate.append(i.eval())
            jGate.append(j.eval())
            fGate.append(f.eval())
            oGate.append(o.eval())
            outputs.append(h.eval())
            states.append(state.c)
            statesh.append(state.h)

    iGate = np.stack(iGate, axis=1)
    jGate = np.stack(jGate, axis=1)
    fGate = np.stack(fGate, axis=1)
    oGate = np.stack(oGate, axis=1)
    outputs = np.stack(outputs, axis=1)
    states = np.stack(np.array(states), axis=1)
    statesh = np.stack(np.array(statesh), axis=1)

    cmap = "inferno"

    print("writing images...")

    print("writing final states...")
    dump3(array=results["convrnn_state"], name="final_state", outfolder=outfolder, cmap="inferno")
    print("writing prediction scores...")
    dump3(array=results["prediction_scores"], name="prediction_scores", outfolder=outfolder, cmap="inferno")

    dump_rgb(results["x"][:, :, :, :, 0:3], "x", outfolder, stddev=4)

    dump(array=iGate, name="iGate", outfolder=outfolder, cmap="inferno")
    dump(array=fGate, name="fGate", outfolder=outfolder, cmap="inferno")
    dump(array=oGate, name="oGate", outfolder=outfolder, cmap="inferno")
    dump(array=(jGate / 2) + 0.5, name="jGate", outfolder=outfolder, cmap="RdBu_r")
    dump(array=(statesh / 2) + 0.5, name="output", outfolder=outfolder, cmap="RdBu_r")
    dump(array=(states / 2) + 0.5, name="state", outfolder=outfolder, cmap="RdBu_r")

    dump_class(results["targets"], "ground_truth", outfolder)
    dump_class(results["predictions"], "predictions", outfolder)
    for i in range(0, 17):
        dump_class(results["prediction_scores"][:, :, :, i], "prediction_scores_" + str(i), outfolder)

def norm(arr,thresmin=-1,thresmax=1):
    arr[arr<thresmin]=thresmin
    arr[arr>thresmax]=thresmax
    return ( (arr-arr.min()) / (arr-arr.min()).max()).astype('float')

def convolution(inputs, W, data_format):
    """wrapper around tf.nn.convolution with custom padding"""
    pad_h = int(W.get_shape()[0]) / 2
    pad_w = int(W.get_shape()[1]) / 2

    paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])

    inputs_padded = tf.pad(inputs, paddings, "REFLECT")

    return tf.nn.convolution(inputs_padded, W, 'VALID', data_format=data_format)

def layer_norm(inputs, beta, gamma):
    """taken from contrib tf.contrib.layers.layer_norm definition in
    tensorflow/contrib/layers/python/layers/layers.py
    """
    mean, variance = tf.nn.moments(inputs, [1, 2, 3], keep_dims=True)
    outputs = tf.nn.batch_normalization(
        inputs, mean, variance, offset=beta, scale=gamma,
        variance_epsilon=1e-12)
    return outputs

def lstm(x, state, weights, convfilters, peephole=False, activation=tf.nn.tanh):
    """Implementation modified from carlthome/tensorflow-convlstm-cell"""

    if peephole:
        kernel, W_ci, W_cf, W_co, b_j, g_j, b_i, g_i, b_f, g_f, b_o, g_o, b_c, g_c = weights
    if not peephole:
        kernel, b_j, g_j, b_i, g_i, b_f, g_f, b_o, g_o, b_c, g_c = weights

    c, h = state
    x = tf.concat([x, h], axis=3).eval()
    n = x.shape[-1]
    m = 4 * convfilters if convfilters > 1 else 4
    y = convolution(x, kernel, data_format="NHWC").eval()
    # y = tf.nn.convolution(x, kernel, 'SAME', data_format="NHWC").eval()
    j, i, f, o = tf.split(y, 4, axis=3)

    if peephole:
        # peephole connections
        i += W_ci * c
        f += W_cf * c

    # normalize
    # replacement for tf.contrib.layers.layer_norm(j)
    #
    ## normalize in cell.py
    # j = tf.contrib.layers.layer_norm(j)
    # i = tf.contrib.layers.layer_norm(i)
    # f = tf.contrib.layers.layer_norm(f)
    #
    j = layer_norm(j, b_j, g_j)

    i = layer_norm(i, b_i, g_i)

    f = layer_norm(f, b_f, g_f)

    forget_bias = 1
    f = tf.sigmoid(f + forget_bias)
    i = tf.sigmoid(i)
    c = c * f + i * activation(j)

    if peephole:
        o += W_co * c

    o = layer_norm(o, b_o, g_o)

    c = layer_norm(c, b_c, g_c)

    o = tf.sigmoid(o)
    h = o * activation(c)

    state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

    return h, state, j, i, f, o


def norm_ptp(arr):
    return (arr - arr.min()) / (arr - arr.min()).max()


def norm_std(arr, stddev=1):
    arr -= arr.mean(axis=0).mean(axis=0)
    arr /= stddev * arr.std(axis=0).std(axis=0)  # [-1,1]
    arr = (arr / 2) + 0.5  # [0,1]
    arr = np.clip(arr * 255, 0, 255)  # [0,255]
    return arr.astype("uint8")


def norm_rgb(arr):
    # taken from QGIS mean +- 2 stddev over cloudfree image
    vmin = np.array([-0.0433, -0.0054, -0.0237])
    vmax = np.array([0.1756, 0.1483, 0.1057])

    arr -= vmin
    arr /= (vmax - vmin)

    return np.clip((arr * 255), 0, 255).astype("uint8")


def write(arr, outfile):
    # norm_img = norm(arr)
    img = Image.fromarray(arr)
    img.save(outfile)


def dump3(array, name, outfolder, cmap="inferno", norm=norm_ptp):
    filenpath = "{outfolder}/sample{s}/{name}/{d}.png"

    cmap = plt.get_cmap(cmap)

    # normalize over the entire array
    # array = norm(array)

    samples, h, w, depth = array.shape
    for s in range(samples):
        for d in range(depth):
            outfilepath = filenpath.format(outfolder=outfolder, s=s, name=name, d=d)

            if not os.path.exists(os.path.dirname(outfilepath)):
                os.makedirs(os.path.dirname(outfilepath))
            arr = array[s, :, :, d]
            arr = cmap(arr)

            write((arr * 255).astype('uint8'), outfilepath)


def dump(array, name, outfolder, cmap="inferno", norm=norm_ptp):
    filenpath = "{outfolder}/sample{s}/time{t}/{d}_{name}.png"

    print("writing "+name+"...")

    cmap = plt.get_cmap(cmap)

    # normalize over the entire array
    # array = norm(array)

    samples, times, h, w, depth = array.shape
    for s in range(samples):
        for t in range(times):
            for d in range(depth):
                outfilepath = filenpath.format(outfolder=outfolder, s=s, t=t, name=name, d=d)

                if not os.path.exists(os.path.dirname(outfilepath)):
                    os.makedirs(os.path.dirname(outfilepath))
                arr = array[s, t, :, :, d]
                arr = cmap(arr)

                write((arr * 255).astype('uint8'), outfilepath)


def dump_rgb(array, name, outfolder, stddev):
    filenpath = "{outfolder}/sample{s}/time{t}_{name}.png"

    samples, times, h, w, depth = array.shape
    for s in range(samples):
        for t in range(times):
            outfilepath = filenpath.format(outfolder=outfolder, s=s, t=t, name=name)

            if not os.path.exists(os.path.dirname(outfilepath)):
                os.makedirs(os.path.dirname(outfilepath))
            arr = array[s, t, :, :, 0:3]

            arr = norm_std(arr, stddev=stddev)

            write(arr, outfilepath)


def dump_class(array, name, outfolder, cmap="Accent"):
    filenpath = "{outfolder}/sample{s}/{name}.png"
    samples, h, w = array.shape

    array = array.astype(float) / 26

    cmap = plt.get_cmap(cmap)
    for s in range(samples):
        outfilepath = filenpath.format(outfolder=outfolder, s=s, name=name)

        arr = (cmap(array[s]) * 255).astype("uint8")
        write(arr, outfilepath)

if __name__ == "__main__":
    main()
