import numpy as np
import os
from data_loader.skeleton import Skeleton


class DatasetAMASS:

    def __init__(self, mode, t_his=30, t_pred=120, use_vel=False):
        self.use_vel = use_vel
        self.mode = mode
        if mode == 'train':
            self.data_file = os.path.join('data', 'data_3d_amass.npz')
        elif mode == 'test':
            self.data_file = os.path.join('data', 'data_3d_amass_test.npz')
        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.use_vel = use_vel
        self.prepare_data()

    def prepare_data(self):
        self.skeleton = Skeleton(parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
                                 joints_left=[1, 4, 7, 10, 13, 16, 18, 20],
                                 joints_right=[2, 5, 8, 11, 14, 17, 19, 21])
        self.kept_joints = np.arange(22)
        self.process_data()

    def process_data(self):
        data_o = np.load(self.data_file, allow_pickle=True)
        data_f = data_o['arr_0']  # (#samples, 150, 22, 3)
        if self.use_vel:
            raise NotImplementedError
        self.data = data_f

    def sample(self):
        n_samples = self.data.shape[0]
        idx = np.random.randint(0, n_samples)
        traj = self.data[idx]
        return traj[None, ...]

    def sampling_generator(self, num_samples=1000, batch_size=8, aug=True):
        if self.mode != 'train':
            aug = False
        for i in range(num_samples // batch_size):
            sample = []
            for i in range(batch_size):
                sample_i = self.sample()
                sample.append(sample_i)
            sample = np.concatenate(sample, axis=0)
            if aug is True:
                if np.random.uniform() > 0.5:  # x-y rotating
                    theta = np.random.uniform(0, 2 * np.pi)
                    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                    rotate_xy = np.matmul(sample.transpose([0, 2, 1, 3])[..., 0:2], rotate_matrix)
                    sample[..., 0:2] = rotate_xy.transpose([0, 2, 1, 3])
                    del theta, rotate_matrix, rotate_xy
                if np.random.uniform() > 0.5:  # x-z mirroring
                    sample[..., 0] = - sample[..., 0]
                if np.random.uniform() > 0.5:  # y-z mirroring
                    sample[..., 1] = - sample[..., 1]
            yield sample

    def iter_generator(self, step=None):
        num_samples = self.data.shape[0]
        for i in range(num_samples):
            seq = self.data[i]
            yield seq[None, ...]

    def sample_iter_action(self, ds_category, dataset_type='amass'):
        sample = [] 

        # ['Transitions', 'SSM', 'DFaust', 'DanceDB', 'GRAB', 'HUMAN4D', 'SOMA']
        if ds_category == 'Transitions':
            i0, i = [0, 233]
        elif ds_category == 'SSM':
            i0, i = [233, 245]
        elif ds_category == 'DFaust':
            i0, i = [245, 342]
        elif ds_category == 'DanceDB':
            i0, i = [342, 6321]
        elif ds_category == 'GRAB':
            i0, i = [6321, 10437]
        elif ds_category == 'HUMAN4D':
            i0, i = [10437, 12317]
        elif ds_category == 'SOMA':
            i0, i = [12317, 12727]
        else:
            raise

        idx = np.random.randint(i0, i)
        traj = self.data[idx]
        sample.append(traj[None, ...])

        sample = np.concatenate(sample, axis=0)
        return sample


if __name__ == '__main__':
    np.random.seed(0)
    dataset = DatasetAMASS('train')
