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

    if args.compare:
        if target_ckpt_path is not None:
            print_compare(source_ckpt_path, target_ckpt_path)
        else:
            print "no target checkpoint present..."
        return # exit

    graph = tf.Graph()
    with graph.as_default() as g:

        # create dummy data iterator
        tf.data.TFRecordDataset("").make_initializable_iterator()

        graph_path = join(args.target,"graph.meta")
        # import meta graph from target model
        print("importing meta graph {}".format(graph_path))
        tf.train.import_meta_graph(graph_path)

        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())

            targetsaver = tf.train.Saver()

            # if target checkpoint exists restore variables first
            if target_ckpt_path is not None:
                print("target checkpoint {}".format(target_ckpt_path))
                targetsaver.restore(sess, target_ckpt_path)
                print("restoring variables from target checkpoint")

            global_step_op = tf.get_default_graph().get_operation_by_name("global_step").outputs[0]
            samples_seen_op = tf.get_default_graph().get_operation_by_name("samples_seen").outputs[0]

            # parse restore variables from --variable flag or --scope flag
            if args.variables is not None:
                vars_dict = get_variable_dictionary_by_variables(args.variables)
            elif args.scopes is not None:
                vars_dict = get_variable_dictionary_by_scope(args.scopes)
            else:
                vars_dict = get_variable_dictionary_by_valid_variables(srccheckpoint=source_ckpt_path,
                                                                       trgcheckpoint=target_ckpt_path)

            for var in sorted(vars_dict.keys()):
                print("restoring variable {}".format(var))

            sourcesaver = tf.train.Saver(var_list=vars_dict)

            print("restoring selected variables from source checkpoing {}".format(source_ckpt_path))
            sourcesaver.restore(sess, source_ckpt_path)

            step = sess.run(global_step_op)

            checkpoint = join(args.target, "model.ckpt")

            if args.reset:
                sess.run([tf.assign(global_step_op,0),tf.assign(samples_seen_op,0)])
                step=0

            if not args.dry:
                print("saving variables to {}".format(checkpoint))
                targetsaver.save(sess, checkpoint, global_step=step)

def get_var(varname):
    return tf.get_default_graph().get_operation_by_name(varname).outputs[0]

def get_variable_dictionary_by_variables(variables):
    # convert dict {src1:trg1,src2:trg2} to list of tuples [(src1,trg1),(src2,trg2)]
    restorevars = [(src, trg) for src, trg in variables.iteritems()]

    vars_dict = dict()
    for source_varname, target_varname in restorevars:
        vars_dict[target_varname] = get_var(source_varname)

    if len(vars_dict.keys()) == 0:
        print("no variables specified. try to restore all")
        vars_dict = None

    return vars_dict

def get_variable_dictionary_by_scope(scopes):
    restorescopes = [(src, trg) for src, trg in scopes.iteritems()]

    vars_dict=dict()

    # for each scope tuple
    for srcscope,trgscope in restorescopes:
        srcvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=srcscope)
        trgvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=trgscope)
            #[var.name.replace(":0","") for var in ]

        # for each variable in the scope
        for src, trg in zip(srcvars,trgvars):
            vars_dict[trg.name.replace(":0","")]=src

    return vars_dict

def get_variable_dictionary_by_valid_variables(srccheckpoint, trgcheckpoint):
    vars_dict = dict()

    valid_vars = find_valid(srccheckpoint=srccheckpoint, trgcheckpoint=trgcheckpoint)
    vars_dict = dict()
    for v in valid_vars:
        vars_dict[v] = get_var(v)

    return vars_dict

def find_valid(srccheckpoint, trgcheckpoint):
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

    return valid_vars

def print_compare(srccheckpoint, trgcheckpoint):

    srcreader = pywrap_tensorflow.NewCheckpointReader(srccheckpoint)
    src_var_to_shape_map = srcreader.get_variable_to_shape_map()

    trgreader = pywrap_tensorflow.NewCheckpointReader(trgcheckpoint)
    trg_var_to_shape_map = trgreader.get_variable_to_shape_map()

    all_vars = set(src_var_to_shape_map).union(trg_var_to_shape_map)

    print("{shpsrc: >17} <-> {shptrg: <17} name: {var} ".format(var="<variable name>", shpsrc="<source shape>", shptrg="<target shape>"))
    for var in sorted(all_vars):

        if var in src_var_to_shape_map.keys():
            shpsrc = src_var_to_shape_map[var]
        else:
            shpsrc = "x"

        if var in trg_var_to_shape_map.keys():
            shptrg = trg_var_to_shape_map[var]
        else:
            shptrg = "x"

        print("{shpsrc: >17} <-> {shptrg: <17} name: {var} ".format(var=var, shpsrc=shpsrc, shptrg=shptrg))

def custom_parser(arg):
    out = {}
    # -v 'a=b,c=d' -> dict {a:b,c:d}
    if "=" in arg:
        vars = arg.split(',')
        for var in vars:
            k, v = var.split('=')
            out[k] = v
    # -v 'a,b' = 'a=a,b=b' -> dict {a:a,b:b}
    else:
        vars = arg.split(',')
        for var in vars:
            out[var] = var

    return out

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Restores some variables from the source model to the target model and saves as checkpoint'
                    'Usage')
    parser.add_argument('source', type=str,
                        help='directory containing the source model (must contain graph.meta and checkpoint files)')
    parser.add_argument('target', type=str,
                        help='directory containing the target model (must contain graph.meta)')
    parser.add_argument("-v", "--variables", type=custom_parser, default=None,
                        help="variables of the source model to be restored to target model. "
                              "This option allows differently named variables"
                             "Format 'src_varname1=trg_varname1,src_varname2=trg_varname2' or 'varname1,varname2' "
                             "(short for 'varname1=varname1,varname2=varname2')")
    parser.add_argument("-s", "--scopes", type=custom_parser, default=None,
                        help="scopes of the source model to be restored to target model. "
                             "This option requires same named variables within differently names scopes"
                             "Format 'src_scopename1=trg_scopename1,src_scopename2=trg_scopename2' or 'scopename1,scopename2' "
                             "(short for 'scopename1=scopename1,scopename2=scopename2')")
    parser.add_argument('--sourcecheckpoint', default=None, type=str,
                        help="specify specific checkpoint base name (e.g. 'model.ckpt-13824') to be restored "
                             "(defaults to latest checkpoint, or initializes variables if no checkpoint is present)")

    parser.add_argument('--targetcheckpoint', default=None, type=str,
                        help="specify specific checkpoint base name (e.g. 'model.ckpt-13824') to be restored "
                             "(defaults to latest checkpoint)")
    parser.add_argument('-d','--dry', action="store_true",help="do not overwrite target checkpoint")

    parser.add_argument('-c', '--compare', action="store_true", help="lists variables and shapes of source and target checkpoints. no further processing")

    parser.add_argument('-r', '--reset', action="store_true",
                        help="resets global_step and samples_seen variables to zero")

    args = parser.parse_args()

    main(args)