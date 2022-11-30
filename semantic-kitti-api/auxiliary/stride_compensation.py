from typing import List
import numpy as np
from numpy.linalg import inv

class DropStrideCompensator():
    def __init__(self, strides: list, scans: list = None):
        if not scans:
            self.scans = [None for _ in range(len(strides))]
        else:
            self.scans = scans
    def update(self, scans: List[np.array], offset: int=None):
        self.scans = scans
        return self.get_compensated_scans()

    def get_compensated_scans(self):
        return self.scans

class FreezeStrideCompensator():
    def __init__(self, strides: list, scans: list = None, offset: int=0):
        self.strides = strides
        self.offset = offset

        if not scans:
            self.scans = [None for _ in range(len(strides))]
        else:
            self.scans = scans
    
    def update(self, scans: list, offset: int=None):
        if offset:
            self.offset = offset
        else:
            self.offset += 1

        for scan, stride, i in zip(scans, self.strides, np.arange(len(scans))):
            should_update = (self.offset % stride) == 0
            print(self.offset, stride, should_update )
            if should_update or self.scans[i] is None:
                self.scans[i] = scan

        return self.get_compensated_scans()

    def get_compensated_scans(self):
        return self.scans

class PoseStrideCompensator():
    def __init__(self, strides: list, poses: List[np.array], scans: list = None, offset: int=0):
        self.strides = strides
        self.offset = offset
        self.poses = poses

        if not scans:
            self.scans = [None for _ in range(len(strides))]
        else:
            self.scans = scans
    
    def update(self, scans: list, offset: int=None):
        if offset:
            self.offset = offset
        else:
            self.offset += 1

        for scan, stride, i in zip(scans, self.strides, np.arange(len(scans))):
            should_update = (self.offset % stride) == 0
            if should_update or self.scans[i] is None:
                self.scans[i] = scan
        return self.get_compensated_scans()

    def get_compensated_scans(self):
        compensated = []
        for scan, stride in zip(self.scans, self.strides):
            should_compensate = (self.offset % stride) != 0
            if should_compensate:
                pose_new = self.poses[self.offset]
                pose_old = self.poses[self.offset - (self.offset % stride)]

                scan_old = np.ones((len(scan), 4))
                scan_old[:, :3] = scan

                pose_diff = np.matmul(inv(pose_new), pose_old)

                scan_new =  np.matmul(pose_diff, scan_old.T).T                
                compensated.append(scan_new[:, :3])
            else:
                compensated.append(scan)
        return compensated

