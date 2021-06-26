
from multiprocessing import Process, Queue

from spatial_transformations import hamiltonian_quaternion_to_rot_matrix
from dataset_utils import csv_read_matrix
import click
import os
from viewer import *


@click.command()
@click.option('--euroc_gt_data_file', required=True,help="Path to the ground truth data file of a Euroc dataset")
def run_viewer_example(euroc_gt_data_file):

    assert(os.path.exists(euroc_gt_data_file))
    poses = []
    first_pos = None
    ground_truth_data = csv_read_matrix(euroc_gt_data_file)
    for gt_line in ground_truth_data[::10]:
        gt = np.array([gt_line[1:]]).astype(np.float64).squeeze()
        gt_pos = gt[0:3]
        gt_quat = gt[3:7]
        if first_pos is None:
            first_pos = gt_pos
        # Transform so that our first position is at the origin.
        trans = gt_pos - first_pos
        rot_mat = hamiltonian_quaternion_to_rot_matrix(gt_quat)
        transform = np.eye(4, dtype=np.float32)
        transform[0:3, 0:3] = rot_mat
        transform[0:3, 3] = trans
        poses.append(transform)
    print("Loaded Euroc")

    pose_q = Queue()
    gt_queue = Queue()

    viewer_process = Process(target=create_and_run, args=(pose_q, gt_queue))
    viewer_process.start()
    import time
    for pose in poses:
        pose_q.put(pose)
        time.sleep(0.01)

if __name__ == '__main__':
    run_viewer_example()
