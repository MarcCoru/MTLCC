import argparse
import tensorflow as tf
from os.path import join
from tensorflow.python import pywrap_tensorflow

def main(args):
    """
    restores first source graph and (if exists) checkpoints
    restores selected variables from target graph with checkpoint weights
        selected variables of target graph will be overwritten
    saves new target graph with checkpoints
    """
    if args.sourcecheckpoint is None:
        source_ckpt_path = tf.train.latest_checkpoint(args.source)
    else:
        source_ckpt_path = join(args.source, args.sourcecheckpoint)

    if args.targetcheckpoint is None:
        target_ckpt_path = tf.train.latest_checkpoint(args.target)
    else:
        target_ckpt_path = join(args.target, args.targetcheckpoint)

    if target_ckpt_path is None:
        print("no target checkpoint present...")
        print("run 'python init_graph.py {}' to initialize variables and create model.ckpt".format(
            join(args.target, "meta.graph")))
        return
    if source_ckpt_path is None:
        print("no source checkpoint present...")
        print("run 'python init_graph.py {}' to initialize variables and create model.ckpt".format(
            join(args.source, "meta.graph")))
        return

    if args.mode=='all':
        print_all(source_ckpt_path, target_ckpt_path)
    elif args.mode=='valid':
        print_valid(source_ckpt_path, target_ckpt_path)
    else:
        print("please provide valid --mode argument (either 'all' or 'valid')")
        return

def print_valid(srccheckpoint, trgcheckpoint):
    """
    search through variables of two checkpoints
    returns variables, which are present in both checkpoints and have equal shape
    """

    srcreader = pywrap_tensorflow.NewCheckpointReader(srccheckpoint)
    src_var_to_shape_map = srcreader.get_variable_to_shape_map()

    trgreader = pywrap_tensorflow.NewCheckpointReader(trgcheckpoint)
    trg_var_to_shape_map = trgreader.get_variable_to_shape_map()

    # common variables in terms of name
    common_vars = set(src_var_to_shape_map).intersection(trg_var_to_shape_map)

    # containing common vars with equal shape
    valid_vars = []

    # check for equal shapes
    for var in common_vars:

        # if shapes are equal
        if trg_var_to_shape_map[var] == src_var_to_shape_map[var]:
            valid_vars.append(var)

    print_vars(valid_vars,src_map=src_var_to_shape_map, trg_map=trg_var_to_shape_map)

def print_vars(vars, src_map, trg_map):
    print("{shpsrc: >17} <-> {shptrg: <17} name: {var} ".format(var="<variable name>", shpsrc="<source shape>",
                                                                shptrg="<target shape>"))
    for var in sorted(vars):

        if var in src_map.keys():
            shpsrc = src_map[var]
        else:
            shpsrc = "x"

        if var in trg_map.keys():
            shptrg = trg_map[var]
        else:
            shptrg = "x"

        print("{shpsrc: >17} <-> {shptrg: <17} name: {var} ".format(var=var, shpsrc=shpsrc, shptrg=shptrg))

def print_all(srccheckpoint, trgcheckpoint):

    srcreader = pywrap_tensorflow.NewCheckpointReader(srccheckpoint)
    src_var_to_shape_map = srcreader.get_variable_to_shape_map()

    trgreader = pywrap_tensorflow.NewCheckpointReader(trgcheckpoint)
    trg_var_to_shape_map = trgreader.get_variable_to_shape_map()

    all_vars = set(src_var_to_shape_map).union(trg_var_to_shape_map)

    print_vars(all_vars, src_map=src_var_to_shape_map, trg_map=trg_var_to_shape_map)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Compare two graphs in terms of variables and variable shapes')
    parser.add_argument('source', type=str,
                        help='directory containing the source model (must contain graph.meta and checkpoint files)')
    parser.add_argument('target', type=str,
                        help='directory containing the target model (must contain graph.meta)')
    parser.add_argument('--mode', default="all", type=str,
                        help="print 'all' or only 'valid' variables (default 'all')")
    parser.add_argument('--sourcecheckpoint', default=None, type=str,
                        help="specify specific checkpoint base name (e.g. 'model.ckpt-13824') to be restored "
                             "(defaults to latest checkpoint, or initializes variables if no checkpoint is present)")

    parser.add_argument('--targetcheckpoint', default=None, type=str,
                        help="specify specific checkpoint base name (e.g. 'model.ckpt-13824') to be restored "
                             "(defaults to latest checkpoint)")
    parser.add_argument('--compatible', action="store_true",
                        help="print only compatible variables")

    args = parser.parse_args()

    main(args)