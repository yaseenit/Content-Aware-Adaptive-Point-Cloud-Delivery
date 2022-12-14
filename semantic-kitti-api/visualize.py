#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import numpy as np
from generate_sequential import parse_calibration, parse_poses
from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.laserscanvis import LaserScanVis
def load_scan_paths(path):
  # populate the pointclouds
  scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
      os.path.expanduser(path)) for f in fn]
  scan_names.sort()
  return scan_names

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./visualize.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to visualize. No Default',
  )
  parser.add_argument(
      '--config', '-c',
      type=str,
      required=False,
      default="config/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--sequence', '-s',
      type=str,
      default="00",
      required=False,
      help='Sequence to visualize. Defaults to %(default)s',
  )
  parser.add_argument(
      '--predictions', '-p',
      type=str,
      default=None,
      required=False,
      help='Alternate location for labels, to use predictions folder. '
      'Must point to directory containing the predictions in the proper format '
      ' (see readme)'
      'Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_semantics', '-i',
      dest='ignore_semantics',
      default=False,
      action='store_true',
      help='Ignore semantics. Visualizes uncolored pointclouds.'
      'Defaults to %(default)s',
  )
  parser.add_argument(
      '--do_instances', '-di',
      dest='do_instances',
      default=False,
      action='store_true',
      help='Visualize instances too. Defaults to %(default)s',
  )
  parser.add_argument(
      '--offset',
      type=int,
      default=0,
      required=False,
      help='Sequence to start. Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_safety',
      dest='ignore_safety',
      default=False,
      action='store_true',
      help='Normally you want the number of labels and ptcls to be the same,'
      ', but if you are not done inferring this is not the case, so this disables'
      ' that safety.'
      'Defaults to %(default)s',
  )
  parser.add_argument(
      '--log_path', '-l',
      default=None,
      help='Path where logs are saved at each update. Default is no logging.'
  )
  parser.add_argument(
      '--log_filtered',
      dest='log_filtered',
      choices=["npy", "ply"],
      default=None,
      help='Saves the filtered points to disk if a format is provided. Must be one of [%(choices)s]. Defaults to %(default)s',
  )
  parser.add_argument(
      '--log_screenshots',
      default=None,
      help='Path where screenshots are saved at each update. Default is off.'
  )
  parser.add_argument(
    '--whitelist',
    type=int,
    default=None,
    nargs="+",
    required=False,
    help='Space delimited list of labels to filter out before visualization'
  )
  parser.add_argument(
    '--additional_sequences_paths',
    type=str,
    default=[],
    nargs="+",
    required=False,
    help='Space delimited list of paths to sequences to visualize too'
  )
  parser.add_argument(
    '--additional_sequences_labels',
    type=int,
    default=[],
    nargs="+",
    required=False,
    help='Space delimited list of labels corresponding to additional_sequences_paths'
  )
  parser.add_argument(
    '--additional_sequences_strides',
    type=int,
    default=[],
    nargs="+",
    required=False,
    help='Space delimited list of stride values corresponding to additional_sequences_paths'
  )
  parser.add_argument(
    '--additional_sequences_strides_compensation',
    type=str,
    choices=['drop', 'freeze', 'pose'],
    default='drop',
    required=False,
    help='Method to use for position compensation'
  )
  parser.add_argument(
    '--background_color',
    type=str,
    default='black',
    required=False,
    help='The color of the visualizations background'
  )
  FLAGS, unparsed = parser.parse_known_args()


  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  for key, value in vars(FLAGS).items():
      print(f"{key} {value}")
  print("*" * 80)

  # open config file
  try:
    print("Opening config file %s" % FLAGS.config)
    CFG = yaml.safe_load(open(FLAGS.config, 'r'))
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # fix sequence name
  FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

  # does sequence folder exist?
  scan_paths = os.path.join(FLAGS.dataset, "sequences",
                            FLAGS.sequence, "velodyne")
  if os.path.isdir(scan_paths):
    print("Sequence folder exists! Using sequence from %s" % scan_paths)
  else:
    print("Sequence folder doesn't exist! Exiting...")
    quit()

  # populate the pointclouds
  scan_names = load_scan_paths(scan_paths)

  poses = None
  if FLAGS.additional_sequences_strides_compensation == "pose":
    pose_path = os.path.join(FLAGS.dataset, "sequences",
                              FLAGS.sequence, "poses.txt")
    calibration_path = os.path.join(FLAGS.dataset, "sequences",
                              FLAGS.sequence, "calib.txt")

    calibration = parse_calibration(calibration_path)
    poses = parse_poses(pose_path, calibration)

  # does sequence folder exist?
  if not FLAGS.ignore_semantics:
    if FLAGS.predictions is not None:
      label_paths = os.path.join(FLAGS.predictions, "sequences",
                                 FLAGS.sequence, "predictions")
    else:
      label_paths = os.path.join(FLAGS.dataset, "sequences",
                                 FLAGS.sequence, "labels")
    if os.path.isdir(label_paths):
      print("Labels folder exists! Using labels from %s" % label_paths)
    else:
      print("Labels folder doesn't exist! Exiting...")
      quit()

    label_names = load_scan_paths(label_paths)

    # check that there are same amount of labels and scans
    if not FLAGS.ignore_safety:
      assert(len(label_names) == len(scan_names)), "Number of scans and labels do not match"


  # create a scan
  if FLAGS.ignore_semantics:
    scan = LaserScan(project=True)  # project all opened scans to spheric proj
  else:
    color_dict = CFG["color_map"]
    nclasses = len(color_dict)
    scan = SemLaserScan(nclasses, color_dict, project=True)
  
  if not FLAGS.whitelist:
    scan.sem_label_whitelist = [id for id,label in CFG['labels'].items()]
  else:
    scan.sem_label_whitelist = FLAGS.whitelist

  if len(scan.sem_label_whitelist) == 0:
      print("No labels to show are configured! Exiting...")

  print("whitelist: ", scan.sem_label_whitelist)

  # create a visualizer
  semantics = not FLAGS.ignore_semantics
  instances = FLAGS.do_instances
  if not semantics:
    label_names = None
  
  assert len(FLAGS.additional_sequences_paths) == len(FLAGS.additional_sequences_labels), "Must specify as many paths as labels"
  additional_sequences = [] # List of tuples (label, [scan_paths])
  for i in range(len(FLAGS.additional_sequences_paths)):
    additional_sequence = load_scan_paths(FLAGS.additional_sequences_paths[i])
    additional_sequences.append((additional_sequence, FLAGS.additional_sequences_labels[i], FLAGS.additional_sequences_strides[i]))

  vis = LaserScanVis(scan=scan,
                     scan_names=scan_names,
                     label_names=label_names,
                     poses=poses,
                     offset=FLAGS.offset,
                     semantics=semantics,
                     instances=instances and semantics,
                     log_path=FLAGS.log_path,
                     log_filtered_format=FLAGS.log_filtered,
                     additional_sequences = additional_sequences,
                     additional_sequences_compensation = FLAGS.additional_sequences_strides_compensation,
                     color_map = CFG["color_map"],
                     background_color = FLAGS.background_color,
                     border_color = FLAGS.background_color,
                     screenshot_path = FLAGS.log_screenshots,
  )

  # print instructions
  print("To navigate:")
  print("\tb: back (previous scan)")
  print("\tn: next (next scan)")
  print("\tl: toggle loop")
  print("\tf: toggle fast forward")
  print("\tq: quit (exit program)")

  # run the visualizer
  vis.run()
