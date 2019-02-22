from S2parser_africa import S2parser
import tensorflow as tf
import os
import configparser
import csv


class Dataset():
    """ A wrapper class around Tensorflow Dataset api handling data normalization and augmentation """

    def __init__(self, datadir, verbose=False, temporal_samples=None, section="dataset", augment=False, country=None):
        self.verbose = verbose

        self.augment = augment

        # parser reads serialized tfrecords file and creates a feature object
        parser = S2parser()
        self.parsing_function = parser.parse_example

        self.temp_samples = temporal_samples
        self.section = section

        # if datadir is None:
        #    dataroot=os.environ["datadir"]
        # else:
        dataroot = datadir

        # csv list of geotransforms of each tile: tileid, xmin, xres, 0, ymax, 0, -yres, srid
        # use querygeotransform.py or querygeotransforms.sh to generate csv
        # fills dictionary:
        # geotransforms[<tileid>] = (xmin, xres, 0, ymax, 0, -yres)
        # srid[<tileid>] = srid
        self.geotransforms = dict()
        # https://en.wikipedia.org/wiki/Spatial_reference_system#Identifier
        self.srids = dict()
        with open(os.path.join(dataroot, "geotransforms.csv"),'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                self.geotransforms[int(row[0])] = (
                float(row[1]), int(row[2]), int(row[3]), float(row[4]), int(row[5]), int(row[6]))
                self.srids[int(row[0])] = int(row[7])


        classes = os.path.join(dataroot,"classes.txt")
        with open(classes, 'r') as f:
            classes = f.readlines()

        self.ids=list()
        self.classes=list()
        for row in classes:
            row=row.replace("\n","")
            if '|' in row:
                id,cl = row.split('|')
                self.ids.append(int(id))
                self.classes.append(cl)

        ## create a lookup table to map labelids to dimension ids

        # map data ids [0, 1, 2, 3, 5, 6, 8, 9, 12, 13, 15, 16, 17, 19, 22, 23, 24, 25, 26]
        labids = tf.constant(self.ids, dtype=tf.int64)

        # to dimensions [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        dimids = tf.constant(range(len(self.ids)), dtype=tf.int64)

        self.id_lookup_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(labids, dimids),
                                            default_value=-1)

        self.inverse_id_lookup_table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(dimids,labids),
                                            default_value=-1)

        #self.classes = [cl.replace("\n","") for cl in f.readlines()]

        cfgpath = os.path.join(dataroot, "dataset.ini")
        # load dataset configs
        datacfg = configparser.ConfigParser()
        datacfg.read(cfgpath)
        cfg = datacfg[section]

        self.country = country
        if self.country is None:
            self.tileidfolder = os.path.join(dataroot, "tileids")
        else:
            self.tileidfolder = os.path.join(dataroot, "tileids/" + self.country + "/tileids")
        self.datadir = os.path.join(dataroot, cfg["datadir"])

        assert 'pix10' in cfg.keys()
        assert 'nobs' in cfg.keys()
        assert 'nbands10' in cfg.keys()

        self.tiletable=cfg["tiletable"]

        self.nobs = int(cfg["nobs"])

        self.expected_shapes = self.calc_expected_shapes(int(cfg["pix10"]),
                                                         int(cfg["nobs"]),
                                                         int(cfg["nbands10"])
                                                         )


        # expected datatypes as read from disk
        self.expected_datatypes = (tf.float32, tf.float32, tf.float32, tf.int64)

    def calc_expected_shapes(self, pix10, nobs, bands10):
        x10shape = (nobs, pix10, pix10, bands10)
        doyshape = (nobs,)
        yearshape = (nobs,)
        labelshape = (nobs, pix10, pix10)

        return [x10shape, doyshape, yearshape, labelshape]

    def transform_labels(self,feature):
        """
        1. take only first labelmap, as labels are not supposed to change
        2. perform label lookup as stored label ids might be not sequential labelid:[0,3,4] -> dimid:[0,1,2]
        """

        x10, doy, year, labels = feature

        # take first label time [46,24,24] -> [24,24]
        # labels are not supposed to change over the time series
        #labels = labels[0]
        labels = self.id_lookup_table.lookup(labels)

        return x10, doy, year, labels

    def normalize(self, feature):
        """
        Normalizes between 0 and 1.
        """ 
        x10, doy, year, labels = feature
        
        doy = tf.cast(doy, tf.float32) / 365
        # year = (2016 - tf.cast(year, tf.float32)) / 2017
        year = tf.cast(year, tf.float32) - 2016
        
        if self.country == 'ghana':

        x10 = tf.scalar_mul(1e-4, tf.cast(x10, tf.float32))



        return x10, doy, year, labels

    def augment(self, feature):

        x10, doy, year, labels = feature

        ## Flip UD

        # roll the dice
        condition = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32), 0.5)

        # flip
        x10 = tf.cond(condition, lambda: tf.reverse(x10, axis=[1]), lambda: x10)
        labels = tf.cond(condition, lambda: tf.reverse(labels, axis=[1]), lambda: labels)


        ## Flip LR

        # roll the dice
        condition = tf.less(tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32), 0.5)

        # flip
        x10 = tf.cond(condition, lambda: tf.reverse(x10, axis=[2]), lambda: x10)
        labels = tf.cond(condition, lambda: tf.reverse(labels, axis=[2]), lambda: labels)

        return x10, doy, year, labels


    def temporal_sample(self, feature):
        """ randomy choose <self.temp_samples> elements from temporal sequence """

        n = self.temp_samples

        # skip if not specified
        if n is None:
            return feature

        x10, doy, year, labels = feature

        # data format 1, 2, 1, 2, -1,-1,-1
        # sequence lengths indexes are negative values.
        # sequence_lengths = tf.reduce_sum(tf.cast(x10[:, :, 0, 0, 0] > 0, tf.int32), axis=1)

        # tf.sequence_mask(sequence_lengths, n_obs)

        # max_obs = tf.shape(x10)[1]
        max_obs = self.nobs

        shuffled_range = tf.random_shuffle(tf.range(max_obs))[0:n]

        idxs = -tf.nn.top_k(-shuffled_range, k=n).values

        x10 = tf.gather(x10, idxs)
        doy = tf.gather(doy, idxs)
        year = tf.gather(year, idxs)

        return x10, doy, year, labels

    def get_ids(self, partition, fold=0):

        def readids(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            return [l.replace("\n", "") for l in lines]

        traintest = "{partition}_fold{fold}.tileids"
        eval = "{partition}.tileids"

        if partition == 'train':
            # e.g. train240_fold0.tileids
            path = os.path.join(self.tileidfolder, traintest.format(partition=partition, fold=fold))
            return readids(path)
        elif partition == 'test':
            # e.g. test240_fold0.tileids
            path = os.path.join(self.tileidfolder, traintest.format(partition=partition, fold=fold))
            return readids(path)
        elif partition == 'eval':
            # e.g. eval240.tileids
            path = os.path.join(self.tileidfolder, eval.format(partition=partition))
            return readids(path)
        else:
            raise ValueError("please provide valid partition (train|test|eval)")

    def create_tf_dataset(self, partition, fold, batchsize, shuffle, prefetch_batches=None, num_batches=-1, threads=8,
                          drop_remainder=False, overwrite_ids=None):

        # set of ids as present in database of given partition (train/test/eval) and fold (0-9)
        allids = self.get_ids(partition=partition, fold=fold)
       
        # set of ids present in local folder (e.g. 1.tfrecord)
        tiles = os.listdir(self.datadir)
        if tiles[0].endswith(".gz"):
            compression = "GZIP"
            ext = ".tfrecord.gz"
        else:
            compression = ""
            ext = ".tfrecord"

        downloaded_ids = [t.replace(".gz", "").replace(".tfrecord", "") for t in tiles]

        # intersection of available ids and partition ods
        if overwrite_ids is None:
            ids = list(set(downloaded_ids).intersection(allids))
        else:
            print "overwriting data ids! due to manual input"
            ids = overwrite_ids

        filenames = [os.path.join(self.datadir, str(id) + ext) for id in ids]

        if self.verbose:
            print "dataset: {}, partition: {}, fold:{} {}/{} tiles downloaded ({:.2f} %)".format(self.section, partition, fold, len(ids), len(allids),
                                                                               len(ids) / float(len(allids)) * 100)

        def mapping_function(serialized_feature):
            # read data from .tfrecords
            feature = self.parsing_function(serialized_example=serialized_feature)
            # sample n times out of the timeseries
            feature = self.temporal_sample(feature)
            # perform data normalization [0,1000] -> [0,1]
            feature = self.normalize(feature)
            # perform data augmentation
            if self.augment: feature = self.augment(feature)
            # replace potentially non sequential labelids with sequential dimension ids
            feature = self.transform_labels(feature)
            return feature



        if num_batches > 0:
            filenames = filenames[0:num_batches * batchsize]

        # shuffle sequence of filenames
        if shuffle:
            filenames = tf.random_shuffle(filenames)

        dataset = tf.data.TFRecordDataset(filenames, compression_type=compression)

        dataset = dataset.map(mapping_function, num_parallel_calls=threads)

        # repeat forever until externally stopped
        dataset = dataset.repeat()

        # Don't trust buffer size -> manual shuffle beforehand
        # if shuffle:
        #    dataset = dataset.shuffle(buffer_size=int(min_after_dequeue))

        if drop_remainder:
            dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(int(batchsize)))
        else:
            dataset = dataset.batch(int(batchsize))

        if prefetch_batches is not None:
            dataset = dataset.prefetch(prefetch_batches)

        # assign output_shape to dataset

        # modelshapes are expected shapes of the data stacked as batch
        output_shape = []
        for shape in self.expected_shapes:
            output_shape.append(tf.TensorShape((batchsize,) + shape))

        return dataset, output_shape, self.expected_datatypes, filenames


def main():
    dataset = Dataset(datadir="/media/data/marc/tfrecords/fields/L1C/480", verbose=True, temporal_samples=30,section="2016")

    training_dataset, output_shapes, output_datatypes, fm_train = dataset.create_tf_dataset("train", 0, 1, 5, True, 32)

    iterator = training_dataset.make_initializable_iterator()

    with tf.Session() as sess:
        sess.run([iterator.initializer, tf.tables_initializer()])
        x10, doy, year, labels = sess.run(iterator.get_next())

if __name__ == "__main__":
    main()
