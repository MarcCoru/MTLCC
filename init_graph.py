import tensorflow as tf
import argparse
from os.path import dirname, join

def main(args):

    # dummy dataset
    tf.data.TFRecordDataset("").make_initializable_iterator()

    # import graph
    print("importing {}".format(args.graph))
    tf.train.import_meta_graph(args.graph)
    saver = tf.train.Saver()

    checkpoint = join(dirname(args.graph),"model.ckpt")
    with tf.Session() as sess:
        print("initializing global variables")
        sess.run(tf.global_variables_initializer())
        print("writing {}".format(checkpoint))
        saver.save(sess, checkpoint)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='creates 0 checkpoint from a directory containing meta graph')
    parser.add_argument('graph', type=str,
                        help='directory containing the source model (must contain graph.meta and checkpoint files)')

    args = parser.parse_args()

    main(args)
