#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import numpy as np
import csv
import pandas as pd

from tasks.semantic.modules.SalsaNext import *
from tasks.semantic.modules.SalsaNextAdf import *
from tasks.semantic.postproc.KNN import KNN
from tasks.semantic.postproc.softmax import threshold_criticality, threshold_percentile
from tasks.semantic.modules.io import write_velodyne, write_labels, write_draco, read_draco
from tasks.semantic.modules.metrics import group_metrics

class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir,split,uncertainty,mc=30, criticality_threshold=None, save_softmax=False, percentile_threshold=None, percentile_sort=False, percentile_descending=False, save_velodyne=False, save_draco=False, save_draco_qp=[], save_draco_qp_default=11, load_draco=False, SOFTMAX_REMAP_CONFIG=None, metrics_log=None, metrics_group_size=1):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir
    self.uncertainty = uncertainty
    self.split = split
    self.mc = mc
    self.criticality_threshold = criticality_threshold
    self.save_softmax = save_softmax
    self.percentile_threshold = percentile_threshold
    self.percentile_sort = percentile_sort
    self.percentile_descending = percentile_descending
    self.save_velodyne = save_velodyne
    self.save_draco = save_draco
    self.save_draco_qp = save_draco_qp
    self.save_draco_qp_default = save_draco_qp_default
    self.load_draco = load_draco
    self.metrics_log = metrics_log
    self.metrics_group_size = metrics_group_size

    if SOFTMAX_REMAP_CONFIG is not None:
        self.SOFTMAX_REMAP_CONFIG = SOFTMAX_REMAP_CONFIG
    else:
        self.SOFTMAX_REMAP_CONFIG = DATA

    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                   self.DATA["name"] + '/parser.py')
    self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=self.DATA["split"]["test"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)

    # concatenate the encoder and the head
    with torch.no_grad():
        torch.nn.Module.dump_patches = True
        if self.uncertainty:
            self.model = SalsaNextUncertainty(self.parser.get_n_classes())
            self.model = nn.DataParallel(self.model)
            w_dict = torch.load(modeldir + "/SalsaNext",
                                map_location=lambda storage, loc: storage)
            self.model.load_state_dict(w_dict['state_dict'], strict=True)
        else:
            self.model = SalsaNext(self.parser.get_n_classes())
            self.model = nn.DataParallel(self.model)
            w_dict = torch.load(modeldir + "/SalsaNext",
                                map_location=lambda storage, loc: storage)
            self.model.load_state_dict(w_dict['state_dict'], strict=True)

    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def remap_softmax(self, softmax):
    num_classes_old = self.parser.get_n_classes()
    num_classes_new = self.parser.get_n_classes(self.SOFTMAX_REMAP_CONFIG["learning_map_inv"])

    indices = np.arange(num_classes_old)
    indices = self.parser.to_original(indices, self.DATA["learning_map_inv"])
    indices = self.parser.to_xentropy(indices, self.SOFTMAX_REMAP_CONFIG["learning_map"])

    softmax_new = torch.tensor(
      torch.zeros((1, num_classes_new, softmax.shape[-2], softmax.shape[-1])),
      dtype=softmax.dtype,
      device=self.device
    )

    for i in range(num_classes_old):
      softmax_new[0, indices[i], :, :] += softmax[0, i, : , :]
      pass

    return softmax_new

  def infer(self):
    metrics = {
      "cnn_inference_time_s": [],
      "knn_inference_time_s": [],
    }
    if self.split == None:

        self.infer_subset(loader=self.parser.get_train_set(),
                          to_orig_fn=self.parser.to_original, metrics=metrics)

        # do valid set
        self.infer_subset(loader=self.parser.get_valid_set(),
                          to_orig_fn=self.parser.to_original, metrics=metrics)
        # do test set
        self.infer_subset(loader=self.parser.get_test_set(),
                          to_orig_fn=self.parser.to_original, metrics=metrics)


    elif self.split == 'valid':
        self.infer_subset(loader=self.parser.get_valid_set(),
                        to_orig_fn=self.parser.to_original, metrics=metrics)
    elif self.split == 'train':
        self.infer_subset(loader=self.parser.get_train_set(),
                        to_orig_fn=self.parser.to_original, metrics=metrics)
    else:
        self.infer_subset(loader=self.parser.get_test_set(),
                        to_orig_fn=self.parser.to_original, metrics=metrics)
    print("Finished Infering")

    df = pd.DataFrame.from_dict(metrics)

    print(f"Using group size of {self.metrics_group_size} for metrics calculation")  
    df = group_metrics(df, self.metrics_group_size)

    print("Printing metrics summary...")
    summary = pd.DataFrame.from_dict({
      "mean": df.mean(),
      "std": df.std(),
    })
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
      print(summary)

    print("Total Frames:{}".format(len(metrics['cnn_inference_time_s'])))

    if self.metrics_log:
      print(f"Saving metrics log to {self.metrics_log}")
      df.to_csv(self.metrics_log)
    return

  def infer_subset(self, loader, to_orig_fn, metrics):
    # switch to evaluate mode
    if not self.uncertainty:
      self.model.eval()
    total_time=0
    total_frames=0
    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()

      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, unproj_xyz, _, unproj_remissions, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        #compute output
        if self.uncertainty:
            proj_output_r,log_var_r = self.model(proj_in)
            for i in range(self.mc):
                log_var, proj_output = self.model(proj_in)
                log_var_r = torch.cat((log_var, log_var_r))
                proj_output_r = torch.cat((proj_output, proj_output_r))

            proj_output2,log_var2 = self.model(proj_in)
            proj_output = proj_output_r.var(dim=0, keepdim=True).mean(dim=1)
            log_var2 = log_var_r.mean(dim=0, keepdim=True).mean(dim=1)
            if self.post:
                # knn postproc
                unproj_argmax = self.post(proj_range,
                                          unproj_range,
                                          proj_argmax,
                                          p_x,
                                          p_y)
            else:
                # put in original pointcloud using indexes
                unproj_argmax = proj_argmax[p_y, p_x]

            # measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            frame_time = time.time() - end
            print("Infered seq", path_seq, "scan", path_name,
                  "in", frame_time, "sec")
            total_time += frame_time
            total_frames += 1
            end = time.time()

            # save scan
            # get the first scan in batch and project scan
            pred_np = unproj_argmax.cpu().numpy()
            pred_np = pred_np.reshape((-1)).astype(np.int32)

            # log_var2 = log_var2[0][p_y, p_x]
            # log_var2 = log_var2.cpu().numpy()
            # log_var2 = log_var2.reshape((-1)).astype(np.float32)

            log_var2 = log_var2[0][p_y, p_x]
            log_var2 = log_var2.cpu().numpy()
            log_var2 = log_var2.reshape((-1)).astype(np.float32)
            # assert proj_output.reshape((-1)).shape == log_var2.reshape((-1)).shape == pred_np.reshape((-1)).shape

            # map to original label
            pred_np = to_orig_fn(pred_np)

            # save scan
            path = os.path.join(self.logdir, "sequences",
                                path_seq, "predictions", path_name)
            pred_np.tofile(path)

            path = os.path.join(self.logdir, "sequences",
                                path_seq, "log_var", path_name)
            if not os.path.exists(os.path.join(self.logdir, "sequences",
                                               path_seq, "log_var")):
                os.makedirs(os.path.join(self.logdir, "sequences",
                                         path_seq, "log_var"))
            log_var2.tofile(path)

            proj_output = proj_output[0][p_y, p_x]
            proj_output = proj_output.cpu().numpy()
            proj_output = proj_output.reshape((-1)).astype(np.float32)

            path = os.path.join(self.logdir, "sequences",
                                path_seq, "uncert", path_name)
            if not os.path.exists(os.path.join(self.logdir, "sequences",
                                               path_seq, "uncert")):
                os.makedirs(os.path.join(self.logdir, "sequences",
                                         path_seq, "uncert"))
            proj_output.tofile(path)

            print(total_time / total_frames)
        else:
            end = time.time()
            proj_output = self.model(proj_in)

            # Remap softmax:
            proj_output = self.remap_softmax(proj_output)

            # Apply threshold:

            # If both methods are set
            if self.criticality_threshold != None and self.percentile_threshold != None:
              raise ValueError("Only one thresholding method may be used!")

            # If no method is set
            if self.criticality_threshold == None and self.percentile_threshold == None:
              proj_argmax = proj_output[0].argmax(dim=0)

            if self.criticality_threshold != None:
              proj_argmax = threshold_criticality(proj_output, self.criticality_threshold)
            
            if self.percentile_threshold != None:              
              proj_argmax = threshold_percentile(proj_output, self.percentile_threshold, sort_softmax=self.percentile_sort, ascending=not self.percentile_descending)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
            print("Network seq", path_seq, "scan", path_name,
                  "in", res, "sec")
            end = time.time()
            metrics['cnn_inference_time_s'].append(res)

            if self.post:
                # knn postproc
                print("Applying post processing")
                unproj_argmax = self.post(proj_range,
                                          unproj_range,
                                          proj_argmax,
                                          p_x,
                                          p_y)
            else:
                # put in original pointcloud using indexes
                print("Skipping post processing")
                unproj_argmax = proj_argmax[p_y, p_x]

            # measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
            print("KNN Infered seq", path_seq, "scan", path_name,
                  "in", res, "sec")
            metrics['knn_inference_time_s'].append(res)
            end = time.time()

            # save scan
            # get the first scan in batch and project scan
            pred_np = unproj_argmax.cpu().numpy()
            pred_np = pred_np.reshape((-1)).astype(np.int32)

            # map to original label
            pred_np = self.parser.to_original(pred_np, self.SOFTMAX_REMAP_CONFIG['learning_map_inv'])

            # save scan
            path = os.path.join(self.logdir, "sequences",
                                path_seq, "predictions", path_name)
            print(f"Saving prediction values to {path}")
            write_labels(pred_np, path)

            if self.save_velodyne or self.save_draco:
              num_classes = proj_output.shape[1]

              draco_qp = {}
              for pair in self.save_draco_qp:
                key, value = pair.split(":")
                draco_qp[int(key)] = int(value)

              for label in range(num_classes):
                label = int(self.parser.to_original(label, self.SOFTMAX_REMAP_CONFIG['learning_map_inv']))

                # Combine and filter scan
                scan = np.concatenate((unproj_xyz, torch.unsqueeze(unproj_remissions, dim=-1)), axis=-1)
                scan = np.squeeze(scan, axis=0)
                scan = scan [:npoints]
                scan = scan[np.where(pred_np == label)]
                if scan.size==0:
                  scan = np.array([
                    [0, 0, 0, 0]
                ])

                def prepare_path(format: str, suffix: str, base_name: str, extension: str) -> str:
                  dir  = os.path.join(
                    self.logdir,
                    "sequences",
                    path_seq,
                    f"{format}-{suffix}"
                  )

                  # TODO: Move to infer.py
                  if not os.path.exists(dir):
                    os.mkdir(dir)

                  return os.path.join(dir, f"{base_name}.{extension}")

                base_name = os.path.splitext(path_name)[0]

                if self.save_velodyne:
                  path = prepare_path("velodyne", label, base_name, "bin")
                  print(f"Saving velodyne scan to {path}")

                  start = time.time()
                  write_velodyne(scan, path)
                  res = time.time() - start

                  key = f"velodyne_encoding_time_s_{label}"
                  if key in metrics:
                    metrics[key].append(res)
                  else:
                    metrics[key] = [res]

                size = os.path.getsize(path)
                key = f"velodyne_file_size_B_{label}"
                if key in metrics:
                  metrics[key].append(size)
                else:
                  metrics[key] = [size]

                if self.save_draco:
                  path = prepare_path("draco", label, base_name, "drc")

                  qp = self.save_draco_qp_default
                  if label in draco_qp:
                    qp = draco_qp[label]

                  print(f"Saving draco [qp:{qp}] scan to {path}")
                  start = time.time()
                  write_draco(scan, path, qp)
                  res = time.time() - start

                  key = f"draco_encoding_time_s_{label}"
                  if key in metrics:
                    metrics[key].append(res)
                  else:
                    metrics[key] = [res]

                size = os.path.getsize(path)
                key = f"draco_file_size_B_{label}"
                if key in metrics:
                  metrics[key].append(size)
                else:
                  metrics[key] = [size]

                if self.save_draco and self.load_draco:
                  res = time.time() - end
                  read_draco(path)
                  key = f"draco_decoding_time_s_{label}"
                  if key in metrics:
                    metrics[key].append(res)
                  else:
                    metrics[key] = [res]

                key = f"scan_num_points_{label}"
                res = len(scan)
                if key in metrics:
                  metrics[key].append(res)
                else:
                  metrics[key] = [res]

            if self.save_softmax:
              softmax_np = np.vstack([proj_output[0][i][p_y, p_x].cpu().numpy() for i in range(proj_output.shape[1])])
              path_softmax = os.path.join(self.logdir, "sequences",
                                  path_seq, "softmax", f"{os.path.splitext(path_name)[0]}.npy")
              print(f"Saving softmax values to {path_softmax}")
              np.save(path_softmax, softmax_np)