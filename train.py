import tensorflow as tf
import os
from Dataset import Dataset
import argparse
import datetime
import pdb

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

# simple flag to track if graph is created in this session or has to be imported
graph_created_flag = False


def main(args):
    tf.reset_default_graph()

    if args.verbose: print "setting visible GPU {}".format(args.gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.datadir is None:
        args.datadir = os.environ["datadir"]

    # initialize training and validation datasets
    datasets = setupDatasets(args)

    # train
    train(args, datasets)


def setupDatasets(args):
    assert args.epochs
    assert args.batchsize
    assert args.shuffle
    assert args.limit_batches
    assert args.prefetch
    # assert args.verbose
    assert type(args.fold) == int
    assert args.datadir
    # assert args.temporal_samples



    datasets_dict=dict()
    for section in ['2017']:
        datasets_dict[section]=dict()
        dataset = Dataset(datadir=args.datadir, verbose=True, temporal_samples=args.temporal_samples, section=section)
        for partition in [TRAINING_IDS_IDENTIFIER, TESTING_IDS_IDENTIFIER]:
            datasets_dict[section][partition] = dict()
            tfdataset, _, _, filenames = dataset.create_tf_dataset(partition,
                                                                   args.fold,
                                                                   args.batchsize,
                                                                   args.shuffle,
                                                                   prefetch_batches=args.prefetch,
                                                                   num_batches=args.limit_batches)
            iterator = tfdataset.make_initializable_iterator()

            datasets_dict[section][partition]["iterator"]=iterator
            datasets_dict[section][partition]["filenames"]=filenames
            #dataset_list.append({'sec':section,'id':identifier,'iterator':iterator,'filenames':filenames})
    print('datasets_dict: ', datasets_dict)
    return datasets_dict


def train(args, datasets):
    assert args.max_models_to_keep
    assert type(args.allow_growth) == bool
    assert args.summary_frequency
    assert args.save_frequency
    assert args.train_on


    # training_iterator, fn_train = training_package
    # validate_iterator, fn_test = validate_package
    num_samples = 0
    for dataset in set(args.train_on): # conversion to set removes duplicates
        num_samples += int(datasets[dataset]["train"]["filenames"].get_shape()[0])

    # if if num batches artificially reduced -> adapt sample size
    if args.limit_batches > 0:
        num_samples = args.limit_batches * args.batchsize
        print("artificially limiting batches to {} -> number of samples {}".format(args.limit_batches, num_samples))

    graph = os.path.join(args.modeldir, MODEL_GRAPH_NAME)
    if not graph_created_flag:
        if args.verbose: print "importing graph from {}".format(graph)
        dir(tf.contrib)  # see https://github.com/tensorflow/tensorflow/issues/10130
        _ = tf.train.import_meta_graph(graph)

    def get_operation(name):
        print('get operation: ', name, tf.get_default_graph().get_operation_by_name(name).outputs[0])
        return tf.get_default_graph().get_operation_by_name(name).outputs[0]

    iterator_handle_op = get_operation("data_iterator_handle")
    is_train_op = get_operation("is_train")
    global_step_op = get_operation("global_step")
    samples_seen_op = get_operation("samples_seen")
    train_op = get_operation("train_op")
    cross_entropy_op = get_operation("loss")
    overall_accuracy_op = get_operation("overall_accuracy")
    if args.learning_rate is not None:
        learning_rate_op = get_operation("learning_rate")

    ## defined local summary and save functions
    def write_summaries(sess, datasets):

        samples, step = sess.run([samples_seen_op, global_step_op])
        cur_epoch = samples / float(num_samples)
        msg = "writing summaries: epoch {:.2f} of {}: step {} ({} tiles seen)"
        print(msg.format(cur_epoch, args.epochs, step, samples))
        for dataset in datasets.keys():
            for partition in datasets[dataset].keys():
                handle = datasets[dataset][partition]["handle"]
                writer = datasets[dataset][partition]["writer"]
                print('handle: ', handle.shape, handle)
                ops = [tf.summary.merge_all(), cross_entropy_op, overall_accuracy_op]
                sum, xe, oa = sess.run(ops, feed_dict={iterator_handle_op: handle, is_train_op: True})
                writer.add_summary(sum, samples)

                msg = "{}/{}: cross entropy: {:.2f}, overall accuracy {:.2f}"
                print(msg.format(dataset, partition, xe, oa))

        #ops = [tf.summary.merge_all(), cross_entropy_op, overall_accuracy_op]
        #sum, xe_te, oa_te = sess.run(ops, feed_dict={iterator_handle_op: validate_handle, is_train_op: False})
        #test_writer.add_summary(sum, samples)





        return cur_epoch

    def save(saver, step, sess, checkpoint):
        saver.save(sess, checkpoint, global_step=step)
        print "saving checkpoint step {}".format(step)

    saver = tf.train.Saver(max_to_keep=args.max_models_to_keep, keep_checkpoint_every_n_hours=args.save_every_n_hours,
                           save_relative_paths=True)

    checkpoint = os.path.join(args.modeldir, MODEL_CHECKPOINT_NAME)

    #config = tf.ConfigProto()
    config = tf.ConfigProto(use_per_session_threads=True)
    config.gpu_options.allow_growth = args.allow_growth
    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])

        for dataset in datasets.keys():
            for partition in datasets[dataset].keys():
                # mark partition, which will be used for training with capital letters
                if (dataset in args.train_on) and (partition == 'train'):
                    summaryname=partition+dataset+"(t)"
                else:
                    summaryname=partition+dataset

                iterator = datasets[dataset][partition]["iterator"]
                datasets[dataset][partition]["handle"] = sess.run(iterator.string_handle())

                writer = tf.summary.FileWriter(os.path.join(args.modeldir, summaryname), sess.graph)
                datasets[dataset][partition]["writer"] = writer

                print "initializing dataset {}, partition {}".format(dataset,partition)
                sess.run([iterator.initializer])

        latest_ckpt = tf.train.latest_checkpoint(args.modeldir)
        if latest_ckpt is not None:
            print "restoring from " + latest_ckpt
            saver.restore(sess, latest_ckpt)

        step, samples = sess.run([global_step_op, samples_seen_op])
        current_epoch = samples / float(num_samples)
        while current_epoch <= args.epochs:
            #try:

            for dataset in args.train_on:
                # normal training operation
                print "{} {} training step {}...".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),dataset, step)

                feed_dict = {iterator_handle_op: datasets[dataset]["train"]["handle"], is_train_op: True}
                print('checkpoint0')
                if args.learning_rate is not None:
                     feed_dict[learning_rate_op] = args.learning_rate

                print('checkpoint1')
                sess.run(train_op, feed_dict=feed_dict)

                print('checkpoint2')

                # write summary
                if step % args.summary_frequency == 0:
                    current_epoch = write_summaries(sess, datasets)
                    print('checkpoint3')

                print('checkpoint4')
                # write checkpoint
                if step % args.save_frequency == 0:
                    save(saver, step, sess, checkpoint)
                    print('checkpoint5')
                    # print "saving to " + checkpoint
                    # saver.save(sess, checkpoint, global_step=step)

                print('checkpoint6')
                step += 1  # keep local step counter



            #except KeyboardInterrupt:
            #    print "Training aborted at step {}".format(step)
            #    break

        # if loop ends or any caught exception
        write_summaries(sess, datasets)
        save(saver, step, sess, checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training of models')

    parser.add_argument('modeldir', type=str, help="directory containing TF graph definition 'graph.meta'")
    # parser.add_argument('--modelzoo', type=str, default="modelzoo", help='directory of model definitions (as referenced by flags.txt [model]). Defaults to environment variable $modelzoo')
    parser.add_argument('--datadir', type=str, default=None,
                        help='directory containing the data (defaults to environment variable $datadir)')
    parser.add_argument('-g', '--gpu', type=str, default="0", help='GPU')
    parser.add_argument('-d','--train_on', type=str, default="2016",nargs='+', help='Dataset partition to train on. Datasets are defined as sections in dataset.ini in datadir')
    parser.add_argument('-b', '--batchsize', type=int, default=2, help='batchsize')
    parser.add_argument('-v', '--verbose', action="store_true", help='verbosity')
    # parser.add_argument('-o', '--overwrite', action="store_true", help='overwrite graph. may lead to problems with checkpoints compatibility')
    parser.add_argument('-s', '--shuffle', type=bool, default=True, help="batch shuffling")
    parser.add_argument('-e', '--epochs', type=int, default=1, help="epochs")
    parser.add_argument('-t', '--temporal_samples', type=int, default=None, help="Temporal subsampling of dataset. "
                                                                                 "Will at each get_next randomly choose "
                                                                                 "<temporal_samples> elements from "
                                                                                 "timestack. Defaults to None -> no temporal sampling")
    parser.add_argument('--save_frequency', type=int, default=64, help="save frequency")
    parser.add_argument('--summary_frequency', type=int, default=64, help="summary frequency")
    parser.add_argument('-f', '--fold', type=int, default=0, help="fold (requires train<fold>.ids)")
    parser.add_argument('--prefetch', type=int, default=6, help="prefetch batches")
    parser.add_argument('--max_models_to_keep', type=int, default=5, help="maximum number of models to keep")
    parser.add_argument('--learning_rate', type=float, default=None,
                        help="overwrite learning rate. Required placeholder named 'learning_rate' in model")
    parser.add_argument('--save_every_n_hours', type=int, default=1, help="save checkpoint every n hours")
    parser.add_argument('--queue_capacity', type=int, default=256, help="Capacity of queue")
    parser.add_argument('--allow_growth', type=bool, default=False, help="Allow dynamic VRAM growth of TF")
    parser.add_argument('--limit_batches', type=int, default=-1,
                        help="artificially reduce number of batches to encourage overfitting (for debugging)")

    args = parser.parse_args()

    main(args)
