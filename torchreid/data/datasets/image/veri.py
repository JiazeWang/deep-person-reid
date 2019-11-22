from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import re
import os.path as osp

from .base import BaseImageDataset


class VeRi(ImageDataset):
    """
    VeRi
    Reference:
    Liu, X., Liu, W., Ma, H., Fu, H.: Large-scale vehicle re-identification in urban surveillance videos. In: IEEE   %
    International Conference on Multimedia and Expo. (2016) accepted.
    Dataset statistics:
    # identities: 776 vehicles(576 for training and 200 for testing)
    # images: 37778 (train) + 11579 (query)
    """
    dataset_dir = 'VeRi'

    def __init__(self, root='/mnt/SSD/jzwang/reid/ReID_data/first_round/test', verbose=True, **kwargs):

        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'query_a_list.txt')
        self.query_dir = osp.join(self.dataset_dir, 'query_a_list.txt')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery_new.txt')

        #self.check_before_run()

        train = self.process_split(self.train_dir, relabel=True)
        query = self.process_split(self.query_dir, relabel=False)
        gallery = self.process_split(self.gallery_dir, relabel=False)

        if verbose:
            print('=> VeRi loaded')
            self.print_dataset_statistics(train, query, gallery)
        """
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        """
        super(VeRi, self).__init__(train, query, gallery, **kwargs)

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('"{}" is not available'.format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError('"{}" is not available'.format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError('"{}" is not available'.format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError('"{}" is not available'.format(self.gallery_dir))

    def parse_img_pids(self, nl_pairs, pid2label=None):
        # il_pair is the pairs of img name and label
        output = []
        for info in nl_pairs:
            name = info[0]
            pid = info[1]
            if pid2label is not None:
                pid = pid2label[pid]
            camid = 1  # don't have camid information use 1 for all
            img_path = osp.join(self.img_dir, name)
            output.append((img_path, pid, camid))
        return output

    def process_split(self,dir_path,relabel=False):
        # read train paths
        train_pid_dict = defaultdict(list)

        # 'train_list.txt' format:
        # the first number is the number of image
        # the second number is the id of vehicle
        with open(dir_path) as f_train:
            train_data = f_train.readlines()
            for data in train_data:
                name, pid = data.split(' ')
                pid = int(pid)
                train_pid_dict[pid].append([name, pid])
        train_pids = list(train_pid_dict.keys())
        num_train_pids = len(train_pids)
        #assert num_train_pids == 13164, 'There should be 13164 vehicles for training,' \
        #                                ' but but got {}, please check the data'\
        #                                .format(num_train_pids)
        print('num of train ids: {}'.format(num_train_pids))
        train_data = []
        for pid in train_pids:
            imginfo = train_pid_dict[pid]  # imginfo include image name and id
            train_data.extend(imginfo)
        train = self.parse_img_pids(train_data)

        return train



    """
    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c([-\d]+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
    """
