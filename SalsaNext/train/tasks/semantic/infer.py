#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger

from tasks.semantic.modules.user import *
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean expected')

if __name__ == '__main__':
    splits = ["train", "valid", "test"]
    parser = argparse.ArgumentParser("./infer.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=os.path.expanduser("~") + '/logs/' + str(datetime.datetime.now()) + '/',
        help='Directory to put the predictions. Default: ~/logs/date+time'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        default=None,
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--uncertainty', '-u',
        type=str2bool, nargs='?',
        const=True, default=False,
        help='Set this if you want to use the Uncertainty Version'
    )

    parser.add_argument(
        '--monte-carlo', '-c',
        type=int, default=30,
        help='Number of samplings per scan'
    )


    parser.add_argument(
        '--split', '-s',
        type=str,
        required=False,
        default=None,
        help='Split to evaluate on. One of ' +
             str(splits) + '. Defaults to %(default)s',
    )
    parser.add_argument(
        '--criticality_threshold',
        type=float, default=None,
        help='Softmax threshold of criticality 1 over which the prediction is overriden as True'
    )
    parser.add_argument(
        '--percentile_threshold',
        type=float, default=None,
        help='Percentile threshold to be reached to assign a criticality'
    )
    parser.add_argument(
        '--percentile_sort',
        default=False, action='store_true',
        help='Whether softmax values should be sorted for percentile calculation or not'
    )
    parser.add_argument(
        '--save_softmax',
        default=False, action='store_true',
        help='If set, save the softmax values of each class and scan into a softmax folder parallel to the prediction folder',
    )
    parser.add_argument(
        '--save_velodyne',
        default=False, action='store_true',
        help='If set, save scans in velodyne format separated by class.',
    )
    parser.add_argument(
        '--save_draco',
        default=False, action='store_true',
        help='If set, save scans in draco format separated by class.',
    )
    parser.add_argument(
        '--save_draco_qp_default',
        type=int,
        default=11,
        help='Default position quantization value used for draco compression. Defaults to %(default)s',
    )
    parser.add_argument(
        '--save_draco_qp',
        default=[],
        nargs='+',
        help='List of class:qp pairs delimited by spaces for draco encoding.',
    )
    parser.add_argument(
        '--percentile_descending',
        default=False, action='store_true',
        help='Sum up softmax in descending order if set, otherwise ascending order is used.'
    )
    parser.add_argument(
        '--load_draco',
        default=False, action='store_true',
        help='Reload files that where saved as draco files to measure also decoding time. Works only if --save_draco is set.'
    )
    parser.add_argument(
        '--softmax_remap_config',
        type=str,
        required=False,
        default=None,
        help='The data configuration file to use. Defaults to the models original configuration.'
    )
    parser.add_argument(
        '--metrics_log',
        type=str,
        required=False,
        default=None,
        help='Path to file where metrics for each scan are saved. By default does not save metrics.'
    )
    parser.add_argument(
        '--metrics_group_size',
        type=int,
        default=1,
        help='Number of scans for which metrics should be combined',
    )

    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    for key, value in vars(FLAGS).items():
        print(f"{key} {value}")
    print("----------\n")
    #print("Commit hash (training version): ", str(
    #    subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
    print("----------\n")

    # open arch config file
    try:
        print("Opening arch config file from %s" % FLAGS.model)
        ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data input config file from %s" % FLAGS.model)
        DATA = yaml.safe_load(open(FLAGS.model + "/data_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    if FLAGS.softmax_remap_config is not None:
        try:
            print("Opening softmax remap config file from %s" % FLAGS.softmax_remap_config)
            SOFTMAX_REMAP_CONFIG = yaml.safe_load(open(FLAGS.softmax_remap_config, 'r'))
        except Exception as e:
            print(e)
            print("Error opening softmax remap yaml file.")
            quit()
    else:
        SOFTMAX_REMAP_CONFIG = None

    # create log folder
    try:
        if os.path.isdir(FLAGS.log):
            shutil.rmtree(FLAGS.log)
        os.makedirs(FLAGS.log)
        os.makedirs(os.path.join(FLAGS.log, "sequences"))
        for seq in DATA["split"]["train"]:
            seq = '{0:02d}'.format(int(seq))
            print("train", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "softmax"))
        for seq in DATA["split"]["valid"]:
            seq = '{0:02d}'.format(int(seq))
            print("valid", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "softmax"))
        for seq in DATA["split"]["test"]:
            seq = '{0:02d}'.format(int(seq))
            print("test", seq)
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "predictions"))
            os.makedirs(os.path.join(FLAGS.log, "sequences", seq, "softmax"))
    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        raise

    except Exception as e:
        print(e)
        print("Error creating log directory. Check permissions!")
        quit()

    # does model folder exist?
    if os.path.isdir(FLAGS.model):
        print("model folder exists! Using model from %s" % (FLAGS.model))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()

    # create user and infer dataset
    user = User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model,FLAGS.split,FLAGS.uncertainty,FLAGS.monte_carlo,FLAGS.criticality_threshold, FLAGS.save_softmax, FLAGS.percentile_threshold, FLAGS.percentile_sort, FLAGS.percentile_descending, save_velodyne=FLAGS.save_velodyne, save_draco=FLAGS.save_draco, save_draco_qp=FLAGS.save_draco_qp, save_draco_qp_default=FLAGS.save_draco_qp_default, load_draco=FLAGS.load_draco, SOFTMAX_REMAP_CONFIG=SOFTMAX_REMAP_CONFIG, metrics_log=FLAGS.metrics_log, metrics_group_size=FLAGS.metrics_group_size)
    user.infer()
