#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy.scene import visuals, SceneCanvas,Text
import numpy as np
from matplotlib import pyplot as plt
from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.stride_compensation import DropStrideCompensator, FreezeStrideCompensator, PoseStrideCompensator
from sys import getsizeof
import csv
import os
import open3d as o3d
from plyfile import PlyData, PlyElement
from typing import List, Tuple, Dict

class LaserScanVis:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self,
    scan: LaserScan,
    scan_names: List[str],
    label_names: List[str],
    poses: List[np.array] = None,
    offset: int = 0,
    semantics: bool = True,
    instances: bool = False,
    log_path: str = None,
    log_filtered_format: str = None,
    whitelist: bool = True,
    additional_sequences: Tuple[List[str], int, int] = None,
    additional_sequences_compensation: str = "drop",
    color_map: Dict[int, List[int]] = {},
    background_color = 'black',
    border_color = 'black',
  ):
    self.scan = scan
    self.scan_names = scan_names
    self.label_names = label_names
    self.poses = poses
    self.offset = offset
    self.total = len(self.scan_names)
    self.semantics = semantics
    self.instances = instances
    self.log = []
    self.log_path = log_path
    self.log_filtered_format = log_filtered_format
    self.fast_forward=False
    self.loop=True
    self.whitelist = whitelist
    self.additional_sequences = additional_sequences
    self.additional_sequences_compensation = additional_sequences_compensation
    self.color_map = color_map
    self.background_color = background_color
    self.border_color = border_color

    strides: List[int] = [int(stride) for paths, label, stride in self.additional_sequences]

    if self.additional_sequences_compensation == "drop":
      self.stride_compensator = DropStrideCompensator(strides)
    elif self.additional_sequences_compensation == "freeze":
      self.stride_compensator = FreezeStrideCompensator(strides, self.offset)
    elif self.additional_sequences_compensation == "pose":
      self.stride_compensator = PoseStrideCompensator(strides, self.poses, None, self.offset)
    else:
      raise ValueError(f"Invalid stride compensation method {self.additional_sequences_compensation}")

    # sanity check
    if not self.semantics and self.instances:
      print("Instances are only allowed in when semantics=True")
      raise ValueError

    self.reset()
    
    self.update_scan()

  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities

    # new canvas prepared for visualizing data
    self.canvas = SceneCanvas(keys='interactive', show=True)
    # interface (n next, b back, q quit, very simple)

    self.canvas.events.key_press.connect(self.key_press)
    self.canvas.events.draw.connect(self.draw)
    # grid
    self.grid = self.canvas.central_widget.add_grid()

    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(
        border_color=self.border_color, bgcolor=self.background_color, parent=self.canvas.scene)
    self.grid.add_widget(self.scan_view, 0, 0)
    self.scan_vis = visuals.Markers()
    self.scan_view.camera = 'turntable'
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)
    # add semantics
    if self.semantics:
      print("Using semantics in visualizer")
      self.sem_view = vispy.scene.widgets.ViewBox(
          border_color=self.border_color, bgcolor=self.background_color, parent=self.canvas.scene)
      self.grid.add_widget(self.sem_view, 0, 1)
      self.sem_vis = visuals.Markers()
      self.sem_view.camera = 'turntable'
      self.sem_view.add(self.sem_vis)
      visuals.XYZAxis(parent=self.sem_view.scene)
      # self.sem_view.camera.link(self.scan_view.camera)
      
      # add filted pointcloud view
      print("Using filted pointcloud  in visualizer")
      self.whitelisted_view = vispy.scene.widgets.ViewBox(
          border_color=self.border_color, bgcolor=self.background_color, parent=self.canvas.scene)
      self.grid.add_widget(self.whitelisted_view, 0, 2)
      self.whitelisted_vis= visuals.Markers()
      self.whitelisted_view.camera = 'turntable'
      self.whitelisted_view.add(self.whitelisted_vis)
      visuals.XYZAxis(parent=self.whitelisted_view.scene)

    if len(self.additional_sequences) > 0:
      print("Showing additional sequences")
      self.additional_sequences_view = vispy.scene.widgets.ViewBox(
          border_color=self.border_color, bgcolor=self.background_color, parent=self.canvas.scene)
      self.grid.add_widget(self.additional_sequences_view, 0, 3)
      self.additional_sequences_vis = visuals.Markers()
      self.additional_sequences_view.camera = 'turntable'
      self.additional_sequences_view.add(self.additional_sequences_vis)
      visuals.XYZAxis(parent=self.additional_sequences_view.scene)

    if self.instances:
      print("Using instances in visualizer")
      self.inst_view = vispy.scene.widgets.ViewBox(
          border_color=self.border_color, bgcolor=self.background_color, parent=self.canvas.scene)
      self.grid.add_widget(self.inst_view, 0, 2)
      self.inst_vis = visuals.Markers()
      self.inst_view.camera = 'turntable'
      self.inst_view.add(self.inst_vis)
      visuals.XYZAxis(parent=self.inst_view.scene)
      # self.inst_view.camera.link(self.scan_view.camera)

    # img canvas size
    self.multiplier = 1
    self.canvas_W = 1024
    self.canvas_H = 64
    if self.semantics:
      self.multiplier += 1
    if self.instances:
      self.multiplier += 1

    # new canvas for img
    self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                  size=(self.canvas_W, self.canvas_H * self.multiplier))
    # grid
    self.img_grid = self.img_canvas.central_widget.add_grid()
    # interface (n next, b back, q quit, very simple)
    self.img_canvas.events.key_press.connect(self.key_press)
    self.img_canvas.events.draw.connect(self.draw)

    # add a view for the depth
    self.img_view = vispy.scene.widgets.ViewBox(
        border_color=self.border_color, bgcolor=self.background_color, parent=self.img_canvas.scene)
    self.img_grid.add_widget(self.img_view, 0, 0)
    self.img_vis = visuals.Image(cmap='viridis')
    self.img_view.add(self.img_vis)

    # add semantics
    if self.semantics:
      self.sem_img_view = vispy.scene.widgets.ViewBox(
          border_color=self.border_color, bgcolor=self.background_color, parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.sem_img_view, 1, 0)
      self.sem_img_vis = visuals.Image(cmap='viridis')
      self.sem_img_view.add(self.sem_img_vis)
      
      
    # add fliter view
    if self.semantics:
      self.filter_img_view = vispy.scene.widgets.ViewBox(
          border_color=self.border_color, bgcolor=self.background_color, parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.filter_img_view, 2, 0)
      self.filter_img_vis = visuals.Image(cmap='viridis')
      self.filter_img_view.add(self.filter_img_vis)
    

    # add instances
    if self.instances:
      self.inst_img_view = vispy.scene.widgets.ViewBox(
          border_color=self.border_color, bgcolor=self.background_color, parent=self.img_canvas.scene)
      self.img_grid.add_widget(self.inst_img_view, 2, 0)
      self.inst_img_vis = visuals.Image(cmap='viridis')
      self.inst_img_view.add(self.inst_img_vis)
      
      
      

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  def save_points(self, points: np.array, path: str, format="npy") -> None:
    """
    Save an array of points to disk in the given format.


    :param points: Numpy array of shape (num_points, 3)
    :param path: File path to save to
    :param format: One of ["npy", "ply"], defaults to "npy"
    """
    if format == "npy":
      np.save(path, points, allow_pickle=False)
    elif format == "ply":
      pointcloud = o3d.geometry.PointCloud()
      pointcloud.points = o3d.utility.Vector3dVector(points)

      elements = []
      types = []

      elements.append(np.asarray(pointcloud.points, dtype=np.float32))
      types.extend((('x', 'single'), ('y', 'single'), ('z', 'single')))
      elements = np.hstack(tuple(elements))

      vertices = np.array(
          [tuple(x) for x in elements.tolist()], # Slow! Should be replaced with something faster if possible
          dtype=types)
      PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(path)

    else:
      raise ValueError(f"{format} format not supported")
    
  def update_scan(self):
    print(f"Updating frame {self.offset+1}/{self.total}")
    # first open data
    self.scan.open_scan(self.scan_names[self.offset])
    if self.semantics:
      self.scan.open_label(self.label_names[self.offset])
      self.scan.colorize()

    # then change names
    title = "scan " + str(self.offset)
    self.canvas.title = title
    self.img_canvas.title = title




    ##################
    if self.log_path is not None:
        self.log.append(
            (
                self.offset,
                self.scan.points.shape[0],
                self.scan.points.nbytes,
                self.scan.whitelisted_points.shape[0],
                self.scan.whitelisted_points.nbytes,
            )
        )
    if len(self.scan.whitelisted_points) > 0:
        print(f"Original Pointcloud Shape: {self.scan.points.shape}")
        print(f"Whitelisted Pointcloud Shape: {self.scan.whitelisted_points.shape}")
        
        print(f"Original Pointcloud Size (kB): {self.scan.points.nbytes/1024.0}")
        print(f"Whitelisted Pointcloud Size (kB): {self.scan.whitelisted_points.nbytes/1024.0}")

        ratio = self.scan.points.nbytes/self.scan.whitelisted_points.nbytes
        print(f"{self.scan.points.nbytes:d} B compressed to {self.scan.whitelisted_points.nbytes:d} B, ratio {ratio:.3f}")
    else:
        print("Whitelisted point cloud contains no points")
    if self.log_filtered_format is not None:
      print("Saving filtered point cloud to disk")
      points_whitelisted_path = os.path.join(self.log_path, "points_whitelisted", str(self.offset)+f".{self.log_filtered_format}")
      self.save_points(self.scan.whitelisted_points, points_whitelisted_path, self.log_filtered_format)

    ''' 
    t1 = Text( parent=self.canvas.scene, color='red')
    t1.font_size = 24
    t1.pos = self.canvas.size[0] // 2, self.canvas.size[1] // 3
    t1.text = str(getsizeof(self.scan.filterd_points))
    '''

    # then do all the point cloud stuff

    # plot scan
    power = 16
    # print()
    range_data = np.copy(self.scan.unproj_range)
    # print(range_data.max(), range_data.min())
    range_data = range_data**(1 / power)
    # print(range_data.max(), range_data.min())
    viridis_range = ((range_data - range_data.min()) /
                     (range_data.max() - range_data.min()) *
                     255).astype(np.uint8)
    viridis_map = self.get_mpl_colormap("viridis")
    viridis_colors = viridis_map[viridis_range]
    self.scan_vis.set_data(self.scan.points,
                           face_color=viridis_colors[..., ::-1],
                           edge_color=viridis_colors[..., ::-1],
                           size=1)

    # plot semantics
    if self.semantics:
      self.sem_vis.set_data(self.scan.points,
                            face_color=self.scan.sem_label_color[..., ::-1],
                            edge_color=self.scan.sem_label_color[..., ::-1],
                            size=1)
       
    # plot filterd pointcloud  
    if self.semantics and len(self.scan.whitelisted_points) > 0:
      self.whitelisted_vis.set_data(self.scan.whitelisted_points,
                            face_color=self.scan.whitelisted_labels_colors[..., ::-1],
                            edge_color=self.scan.whitelisted_labels_colors[..., ::-1],
                            size=1)

    if self.semantics:
      uncompensated_scans = []
      colors = []
      for additional_sequence in self.additional_sequences:
        paths, label, stride = additional_sequence
        path = paths[self.offset]

        color_dict = self.color_map

        additional_scan = o3d.io.read_point_cloud(path)
        uncompensated_scans.append(np.asarray(additional_scan.points))
        colors.append(color_dict[label])
      compensated_scans = self.stride_compensator.update(uncompensated_scans, self.offset)
      colors = np.vstack([np.ones((len(compensated_scan), 3)) / 255.0 * color for color, compensated_scan in zip(colors, compensated_scans)])
      points = np.vstack(compensated_scans)

      self.additional_sequences_vis.set_data(points,
                            face_color=colors,
                            edge_color=colors,
                            size=1)

    # plot instances
    if self.instances:
      self.inst_vis.set_data(self.scan.points,
                             face_color=self.scan.inst_label_color[..., ::-1],
                             edge_color=self.scan.inst_label_color[..., ::-1],
                             size=1)

    # now do all the range image stuff
    # plot range image
    data = np.copy(self.scan.proj_range)
    # print(data[data > 0].max(), data[data > 0].min())
    data[data > 0] = data[data > 0]**(1 / power)
    data[data < 0] = data[data > 0].min()
    # print(data.max(), data.min())
    data = (data - data[data > 0].min()) / \
        (data.max() - data[data > 0].min())
    # print(data.max(), data.min())
    self.img_vis.set_data(data)
    self.img_vis.update()

    if self.semantics:
      self.sem_img_vis.set_data(self.scan.proj_sem_color[..., ::-1])
      self.sem_img_vis.update()
      
      
    if self.semantics:
        self.filter_img_vis.set_data(self.scan.proj_sem_color[..., ::-1])
        self.filter_img_vis.update()

    if self.instances:
      self.inst_img_vis.set_data(self.scan.proj_inst_color[..., ::-1])
      self.inst_img_vis.update()

  # interface
  def key_press(self, event):
    self.canvas.events.key_press.block()
    self.img_canvas.events.key_press.block()
    if event.key == 'N':
      self.next()
    elif event.key == 'B':
      self.back()
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()
    elif event.key == 'F':
      self.fast_forward = not self.fast_forward
      print(f"Fast forward {self.fast_forward}")
      self.next()
    elif event.key == 'L':
      self.loop = not self.loop
      self.update_scan()
      print(f"Loop {self.loop}")
  
  def save_log(self):
    path = os.path.join(self.log_path, "log.csv")
    with open(path, 'w', newline='', encoding='utf-8') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(["scan", "num_points_all", "num_bytes_all", "num_points_filtered", "num_bytes_filtered"])
        for row in self.log:
            csv_out.writerow(row)

  def next(self):
      self.offset += 1
      if self.offset >= self.total:
        if self.loop:
            self.offset = 0
        else:
            self.destroy()
      self.update_scan()
    
  def back(self):
      self.offset -= 1
      if self.offset < 0:
        if self.loop:
            self.offset = self.total - 1
        else:
            self.destroy()
      self.update_scan()

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()
    if self.img_canvas.events.key_press.blocked():
      self.img_canvas.events.key_press.unblock()
    
    if self.fast_forward:
        self.next()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    self.img_canvas.close()
    vispy.app.quit()
    if self.log_path is not None:
        self.save_log()

  def run(self):
    vispy.app.run()
    
    
  def format_bytes(self,size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]+'bytes'
