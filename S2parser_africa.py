import tensorflow as tf
import numpy as np
import sys
import os

class S2parser():
    """ defined the Sentinel 2 .tfrecord format """
    def __init__(self):

        self.feature_format= {
            'x10/data': tf.FixedLenFeature([], tf.string),
            'x10/shape': tf.FixedLenFeature([4], tf.int64),
            'dates/doy': tf.FixedLenFeature([], tf.string),
            'dates/year': tf.FixedLenFeature([], tf.string),
            'dates/shape': tf.FixedLenFeature([1], tf.int64),
            'labels/data': tf.FixedLenFeature([], tf.string),
            'labels/shape': tf.FixedLenFeature([3], tf.int64)
        }

        return None

    def write(self, filename, x10, doy, year, labels):
        # https://stackoverflow.com/questions/39524323/tf-sequenceexample-with-multidimensional-arrays

        writer = tf.python_io.TFRecordWriter(filename)

        x10=x10.astype(np.int64)
        doy=doy.astype(np.int64)
        year=year.astype(np.int64)
        labels=labels.astype(np.int64)

        # Create a write feature
        feature={
            'x10/data' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[x10.tobytes()])),
            'x10/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=x10.shape)),
            'labels/data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.tobytes()])),
            'labels/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=labels.shape)),
            'dates/doy': tf.train.Feature(bytes_list=tf.train.BytesList(value=[doy.tobytes()])),
            'dates/year': tf.train.Feature(bytes_list=tf.train.BytesList(value=[year.tobytes()])),
            'dates/shape': tf.train.Feature(int64_list=tf.train.Int64List(value=doy.shape))
        }


        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()

    def get_shapes(self, sample):
        print("reading shape of data using the sample "+sample)
        data = self.read_and_return(sample)
        return [tensor.shape for tensor in data]

    def parse_example(self,serialized_example):
        """
        example proto can be obtained via
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        or by passing this function in dataset.map(.)
        """

        feature = tf.parse_single_example(serialized_example, self.feature_format)
        # decode and reshape x10
        x10 = tf.reshape(tf.decode_raw(feature['x10/data'], tf.int64),tf.cast(feature['x10/shape'], tf.int32))

        doy = tf.reshape(tf.decode_raw(feature['dates/doy'], tf.int64), tf.cast(feature['dates/shape'], tf.int32))
        year = tf.reshape(tf.decode_raw(feature['dates/year'], tf.int64), tf.cast(feature['dates/shape'], tf.int32))

        labels = tf.reshape(tf.decode_raw(feature['labels/data'], tf.int64), tf.cast(feature['labels/shape'], tf.int32))

        return x10, doy, year, labels

    def read(self,filenames):
        """ depricated! """

        if isinstance(filenames,list):
            filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        elif isinstance(filenames,tf.FIFOQueue):
            filename_queue = filenames
        else:
            print("please insert either list or tf.FIFOQueue")

        reader = tf.TFRecordReader()
        f, serialized_example = reader.read(filename_queue)

        print(f)

        feature = tf.parse_single_example(serialized_example, features=self.feature_format)

        # decode and reshape x10
        x10 = tf.reshape(tf.decode_raw(feature['x10/data'], tf.int64),tf.cast(feature['x10/shape'], tf.int32))

        doy = tf.reshape(tf.decode_raw(feature['dates/doy'], tf.int64), tf.cast(feature['dates/shape'], tf.int32))
        year = tf.reshape(tf.decode_raw(feature['dates/year'], tf.int64), tf.cast(feature['dates/shape'], tf.int32))

        labels = tf.reshape(tf.decode_raw(feature['labels/data'], tf.int64), tf.cast(feature['labels/shape'], tf.int32))

        return x10, doy, year, labels

    def tfrecord_to_pickle(self,tfrecordname,picklename):
        import cPickle as pickle

        reader = tf.TFRecordReader()

        # read serialized representation of *.tfrecord
        filename_queue = tf.train.string_input_producer([tfrecordname], num_epochs=None)
        filename_op, serialized_example = reader.read(filename_queue)
        feature = self.parse_example(serialized_example)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            feature = sess.run(feature)

            coord.request_stop()
            coord.join(threads)

        pickle.dump(feature, open(picklename, "wb"), protocol=2)

    def read_and_return(self,filename):
        """ depricated! """

        # get feature operation containing
        feature_op = self.read([filename])

        with tf.Session() as sess:

            tf.global_variables_initializer()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            return sess.run(feature_op)

def test():
    print("Running self test:")
    print("temporary tfrecord file is written with random numbers")
    print("tfrecord file is read back")
    print("contents are compared")

    filename="tmptile.tfrecord"

    # create dummy dataset
    x10 = (np.random.rand(6,48,48,6)*1e3).astype(np.int64)
    labels = (np.random.rand(6,24,24)*1e3).astype(np.int64)
    doy = (np.random.rand(6)*1e3).astype(np.int64)
    year = (np.random.rand(6)*1e3).astype(np.int64)

    # init parser
    parser=S2parser()

    parser.write(filename, x10, doy, year, labels)

    x10_, doy_, year_, labels_ = read_and_return(filename)

    # test if wrote and read data is the same
    print("TEST")
    if np.all(x10_==x10) and np.all(labels_==labels) and np.all(doy_==doy) and np.all(year_==year):
        print("PASSED")
    else:
        print("NOT PASSED")


    # remove file
    os.remove(filename)

    #return tf.reshape(x10, (1,48,48,6))
    #return feature['x10shape']

if __name__=='__main__':
    #test()

    #x10, x20, x60, doy, year, labels = read_and_return("data/bavaria/1.tfrecord")
    parser = S2parser()

    parser.tfrecord_to_pickle("1.tfrecord","1.pkl")
