import numpy as np
import open3d as o3d
import DracoPy

def write_draco(scan, path, qp):
    binary = DracoPy.encode(
        scan[:, :3],
        quantization_bits=qp, compression_level=1,
        quantization_range=-1, quantization_origin=None,
        create_metadata=False, preserve_order=False
    )

    with open(path, 'wb') as file:
        file.write(binary)

def read_draco(path):
    with open(path, 'rb') as file:
        data = file.read()
        mesh = DracoPy.decode(data)
    return mesh

def write_ply(scan, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scan[:3])
    o3d.io.write_point_cloud(path, pcd)

def read_velodyne(path):
    scan = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
    return scan

def write_velodyne(scan, path):
    scan.tofile(path)

def write_labels(labels, path):
    labels.tofile(path)

def read_labels(path):
    label = np.fromfile(path, dtype=np.int32) & 0xFFFF
    return label