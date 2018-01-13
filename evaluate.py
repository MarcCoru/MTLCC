import tensorflow as tf
import os
from Dataset import Dataset
import argparse
import datetime
from osgeo import gdal, osr
#import psycopg2
from os.path import join
import sklearn.metrics as skmetrics

import threading

#os.environ["GDAL_DATA"] = os.environ["HOME"] + "/.conda/envs/MTLCC/share/gdal"

MODEL_GRAPH_NAME = "graph.meta"
EVAL_IDS_IDENTIFIER = "eval"

MODEL_CFG_FILENAME = "params.ini"
MODEL_CFG_FLAGS_SECTION = "flags"
MODEL_CFG_MODEL_SECTION = "model"
MODEL_CFG_MODEL_KEY = "model"

MODEL_CHECKPOINT_NAME = "model.ckpt"

MASK_FOLDERNAME="mask"
GROUND_TRUTH_FOLDERNAME="ground_truth"
PREDICTION_FOLDERNAME="prediction"
LOSS_FOLDERNAME="loss"
CONFIDENCES_FOLDERNAME="confidences"

TRUE_PRED_FILENAME="truepred.npy"

import numpy as np

# simple flag to track if graph is created in this session or has to be imported
graph_created_flag = False


def main(args):
    # if args.verbose: print "setting visible GPU {}".format(args.gpu)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.datadir is None:
        args.datadir = os.environ["datadir"]

    #with open(os.path.join(os.environ["HOME"], ".pgpass"), 'r') as f:
    #    pgpass = f.readline().replace("\n", "")
    #host, port, db, user, password = pgpass.split(':')
    #conn = psycopg2.connect('postgres://{}:{}@{}/{}'.format(user, password, host, db))

    # train
    eval(args)


def eval(args):
    # assert args.max_models_to_keep
    # assert type(args.allow_growth)==bool
    # assert args.summary_frequency
    # assert args.save_frequency

    dataset = Dataset(datadir=args.datadir, verbose=True, section=args.dataset)

    if args.verbose: print "initializing training dataset"
    tfdataset, output_shapes, output_datatypes, filenames = dataset.create_tf_dataset(EVAL_IDS_IDENTIFIER, 0,
                                                                                      args.batchsize,
                                                                                      False,
                                                                                      prefetch_batches=args.prefetch)
    iterator = tfdataset.make_initializable_iterator()

    num_samples = len(filenames)

    # load meta graph
    graph = os.path.join(args.modeldir, MODEL_GRAPH_NAME)
    _ = tf.train.import_meta_graph(graph)

    def get_operation(name):
        return tf.get_default_graph().get_operation_by_name(name).outputs[0]

    iterator_handle_op = get_operation("data_iterator_handle")
    is_train_op = get_operation("is_train")
    global_step_op = get_operation("global_step")
    samples_seen_op = get_operation("samples_seen")
    train_op = get_operation("train_op")
    cross_entropy_op = get_operation("loss")
    overall_accuracy_op = get_operation("overall_accuracy")
    predictions_scores_op = get_operation("prediction_scores")
    predictions_op = get_operation("predictions")
    loss_per_px_op = get_operation("loss_per_px")
    mask_per_px_op = get_operation("mask_per_px")
    labels_op = get_operation("targets")

    saver = tf.train.Saver(save_relative_paths=True)

    #checkpoint = os.path.join(args.modeldir, MODEL_CHECKPOINT_NAME)



    config = tf.ConfigProto()
    config.gpu_options.allow_growth = args.allow_growth
    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])

        sess.run([iterator.initializer])

        data_handle = sess.run(iterator.string_handle())

        latest_ckpt = tf.train.latest_checkpoint(args.modeldir)
        if latest_ckpt is not None:
            print "restoring from " + latest_ckpt
            saver.restore(sess, latest_ckpt)

        #widgets = ['iterating through ds: ', Percentage(), ' ', Bar(marker='#', left='[', right=']'), ' ',
                   #ETA()]  # see docs for other options
        #pbar = ProgressBar(widgets=widgets, maxval=num_samples)


        step, samples = sess.run([global_step_op, samples_seen_op])
        # current_epoch = samples / float(num_samples)
        batches = num_samples / args.batchsize

        #n_classes = len(dataset.classes)-1 #without unknown
        #confusion_matrix = np.zeros((n_classes, n_classes),dtype=int)

        # create appropiate folders
        def makedir(outfolder):
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)

        makedir(join(args.storedir, PREDICTION_FOLDERNAME))
        makedir(join(args.storedir, MASK_FOLDERNAME))
        makedir(join(args.storedir, PREDICTION_FOLDERNAME))
        makedir(join(args.storedir, GROUND_TRUTH_FOLDERNAME))
        makedir(join(args.storedir, LOSS_FOLDERNAME))

        try:
            truepred = np.empty((0, 2),dtype=int)

            #pbar.start()
            starttime = datetime.datetime.now()
            for i in range(0, batches):
                #pbar.update(i)
                now = datetime.datetime.now()
                dt = now - starttime
                seconds_per_batch = dt.seconds/(i+1e-10)
                seconds_to_go = seconds_per_batch*(batches - i)
                eta = now+datetime.timedelta(seconds=seconds_to_go)
                print "{} eval batch {}/{}, eta: {}".format(now.strftime("%Y-%m-%d %H:%M:%S"), i,batches, eta.strftime("%Y-%m-%d %H:%M:%S"))

                stepfrom = i * args.batchsize
                stepto = stepfrom + args.batchsize + 1

                # take with care...
                files = filenames[stepfrom:stepto]

                # normal training operation
                #print "{} evaluation step {}...".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), step)

                feed_dict = {iterator_handle_op: data_handle, is_train_op: False}

                pred, pred_sc, label, lpp, mpp = sess.run([predictions_op, predictions_scores_op, labels_op, loss_per_px_op, mask_per_px_op], feed_dict=feed_dict)

                # transform sequential dimension ids to labelids
                #label = np.array(dataset.ids)[label]
                #pred = np.array(dataset.ids)[pred]

                # add one since the 0 (unknown dimension is used for mask)
                label = sess.run(dataset.inverse_id_lookup_table.lookup(tf.constant(label + 1, dtype=tf.int64)))
                pred = sess.run(dataset.inverse_id_lookup_table.lookup(tf.constant(pred + 1, dtype=tf.int64)))

                y_true = label[mpp]
                y_pred = pred[mpp]

                truepred = np.row_stack((truepred, np.column_stack((y_true, y_pred))))
                # with open(join(args.storedir, TRUE_PRED_FILENAME), "a") as file:
                #     np.savetxt(file, data, fmt='%d', delimiter=",")

                # set masked pixel back from 1 to 0
                lpp[~mpp] = 0
                label[~mpp] = 0

                if args.writetiles:

                    threadlist=list()

                    for tile in range(pred.shape[0]):
                        tileid = int(os.path.basename(files[tile]).split(".")[0])
                        geotransform = dataset.geotransforms[tileid]
                        srid = dataset.srids[tileid]

                        threadlist.append(write_tile(pred[tile], files[tile], join(args.storedir,PREDICTION_FOLDERNAME), geotransform, srid))
                        #threading.Thread(target=write_tile, args=writeargs).start()

                        threadlist.append(write_tile(mpp[tile], files[tile], join(args.storedir,MASK_FOLDERNAME), geotransform, srid))
                        #thread.start_new_thread(write_tile, writeargs)

                        #write_tile(conn, label[tile], datafilename=files[tile],outfolder=join(args.storedir,GROUND_TRUTH_FOLDERNAME), tiletable=args.tiletable)
                        threadlist.append(write_tile(label[tile], files[tile],join(args.storedir,GROUND_TRUTH_FOLDERNAME), geotransform, srid))
                        #thread.start_new_thread(write_tile,writeargs)

                        #write_tile(conn, lpp[tile], datafilename=files[tile], tiletable=args.tiletable,outfolder=join(args.storedir,LOSS_FOLDERNAME))
                        threadlist.append(write_tile(lpp[tile], files[tile], join(args.storedir,LOSS_FOLDERNAME), geotransform, srid))
                        #thread.start_new_thread(write_tile, writeargs)

                        if args.writeconfidences:
                            for cl in range(pred_sc.shape[-1]):
                                classname = dataset.classes[cl+1].replace(" ","_")

                                foldername="{}_{}".format(cl+1,classname)

                                outfolder = join(args.storedir,CONFIDENCES_FOLDERNAME,foldername)
                                makedir(outfolder) # if not exists

                                # write_tile(conn, pred_sc[tile, :, :, cl], datafilename=files[tile], tiletable=args.tiletable, outfolder=outfolder)
                                threadlist.append(write_tile(pred_sc[tile, :, :, cl], files[tile], outfolder, geotransform, srid))
                                #thread.start_new_thread(write_tile, writeargs)

                    # start all write threads!
                    for x in threadlist:
                        x.start()

                    # wait for all to finish
                    for x in threadlist:
                        x.join()

        except KeyboardInterrupt:
            print "Evaluation aborted at step {}".format(step)
            pass
            #
            # print "pre active threads: {}" + format(threading.activeCount())
            # # wait for threads to finish
            # for thread in threading.enumerate():
            #     thread.join()
            #print "post active threads: {}" + format(threading.activeCount())

    ## write metrics and confusion matrix
    #csv_rows = []
    #csv_rows.append(", ".join(dataset.classes[1:]))

    np.save(join(args.storedir,TRUE_PRED_FILENAME), truepred)

    y_true = truepred[:, 0]
    y_pred = truepred[:, 1]

    ids_without_unknown = np.array(dataset.ids[1:]) - 1
    confusion_matrix = skmetrics.confusion_matrix(y_true, y_pred, labels=dataset.ids[1:])

    with open(join(args.storedir,'confusion_matrix.csv'), 'wb') as f:
        f.write(", ".join(dataset.classes[1:])+"\n")
        np.savetxt(f, confusion_matrix, fmt='%d', delimiter=', ', newline='\n')

    classreport = skmetrics.classification_report(y_true, y_pred, labels=dataset.ids[1:], target_names=dataset.classes[1:])

    classes = np.column_stack((dataset.ids[1:], dataset.classes[1:]))
    np.savetxt(join(args.storedir,'classes.csv'), classes, fmt='%s', delimiter=', ', newline='\n')

    with open(join(args.storedir, 'classification_report.txt'), 'w') as f:
        f.write(classreport)

    dec = 2 # round scores to dec significant digits
    prec, rec, fscore, sup = skmetrics.precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=dataset.ids[1:])
    avg_prec = skmetrics.precision_score(y_true, y_pred, labels=dataset.ids[1:], average='weighted')
    avg_rec = skmetrics.recall_score(y_true, y_pred, labels=dataset.ids[1:], average='weighted')
    avg_fscore = skmetrics.f1_score(y_true, y_pred, labels=dataset.ids[1:], average='weighted')

    metrics = np.column_stack((classes, np.around(prec, dec), np.around(rec, dec), np.around(fscore, dec), sup))
    with open(join(args.storedir, "metrics.csv"), 'w') as f:
        f.write(b'id,name,precision,recall,fscore,support\n')
        np.savetxt(f, metrics, fmt='%s', delimiter=', ', newline='\n')
        f.write(b',,,,,\n'.format(prec=avg_prec, rec=avg_rec, fsc=avg_fscore))
        f.write(b',weight. avg,{prec},{rec},{fsc},\n'.format(prec=np.around(avg_prec, dec), rec=np.around(avg_rec, dec),
                                                             fsc=np.around(avg_fscore, dec)))



        #pbar.finish()
def write_tile(array, datafilename, outfolder, geotransform, srid):
    """gave up for now... wrapper around write_tile_ to implement multithreaded writing"""
    writeargs = (array, datafilename, outfolder, geotransform, srid)
    #write_tile_(array, datafilename, outfolder, geotransform, srid)
    thread = threading.Thread(target=write_tile_, args=writeargs)
    # thread.start()
    return thread


def write_tile_(array, datafilename, outfolder, geotransform, srid):

    tile = os.path.basename(datafilename).replace(".tfrecord", "").replace(".gz", "")

    outpath = os.path.join(outfolder, tile + ".tif")

    #curs = conn.cursor()

    #sql = "select ST_xmin(geom) as xmin, ST_ymax(geom) as ymax, ST_SRID(geom) as srid from {tiletable} where id = {tileid}".format(
    #    tileid=tile, tiletable=tiletable)
    #curs.execute(sql)
    #xmin, ymax, srid = curs.fetchone()

    nx = array.shape[0]
    ny = array.shape[1]
    #xres = 10
    #yres = 10
    #geotransform = (xmin, xres, 0, ymax, 0, -yres)

    if array.dtype == int:
        gdaldtype = gdal.GDT_Int16
    else:
        gdaldtype = gdal.GDT_Float32

    # create the 3-band raster file
    dst_ds = gdal.GetDriverByName('GTiff').Create(outpath, ny, nx, 1, gdaldtype)
    dst_ds.SetGeoTransform(geotransform)  # specify coords
    srs = osr.SpatialReference()  # establish encoding
    srs.ImportFromEPSG(srid)  # WGS84 lat/long
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file

    dst_ds.GetRasterBand(1).WriteArray(array)

    dst_ds.FlushCache()  # write to disk
    dst_ds = None  # save, close


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evalutation of models')
    parser.add_argument('modeldir', type=str, help="directory containing TF graph definition 'graph.meta'")
    # parser.add_argument('--modelzoo', type=str, default="modelzoo", help='directory of model definitions (as referenced by flags.txt [model]). Defaults to environment variable $modelzoo')
    parser.add_argument('--datadir', type=str, default=None,
                        help='directory containing the data (defaults to environment variable $datadir)')
    parser.add_argument('-v', '--verbose', action="store_true", help='verbosity')
    parser.add_argument('-t', '--writetiles', action="store_true",
                        help='write out pngs for each tile with prediction, label etc.')
    parser.add_argument('-c', '--writeconfidences', action="store_true", help='write out confidence maps for each class')
    parser.add_argument('-b', '--batchsize', type=int, default=32,
                        help='batchsize')
    parser.add_argument('-d', '--dataset', type=str, default="2016",
                        help='Dataset partition to train on. Datasets are defined as sections in dataset.ini in datadir')
    parser.add_argument('--prefetch', type=int, default=6, help="prefetch batches")
    # tiletable now read from dataset.ini
    #parser.add_argument('--tiletable', type=str, default="tiles240", help="tiletable (default tiles240)")
    parser.add_argument('--allow_growth', type=bool, default=True, help="Allow VRAM growth")
    parser.add_argument('--storedir', type=str, default="tmp", help="directory to store tiles")

    args = parser.parse_args()

    main(args)
