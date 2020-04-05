import torch
import cv2
import numpy as np
import os
from .utils import campose_to_extrinsic, read_intrinsics
from PIL import Image


class IBRStaticDataset(torch.utils.data.Dataset):

    def __init__(self,fn_point_xyz, fn_campose, fn_camintrinsic, img_path, transforms, near_far_size, is_need_all_data):
        super(IBRStaticDataset, self).__init__()

        self.vs = torch.Tensor( np.loadtxt(fn_point_xyz) )

        #self.vs = self.vs + 3.0*torch.randn_like(self.vs)


        self.files = os.listdir(img_path)
        self.files.sort()
        self.file_path = img_path

        camposes = np.loadtxt(fn_campose)
        self.Ts = torch.Tensor( campose_to_extrinsic(camposes) )
        self.Ks = torch.Tensor( read_intrinsics(fn_camintrinsic)) /2
        self.Ks[:,2,2] = 1

        self.transforms = transforms
        self.near_far_size = torch.Tensor(near_far_size)

        self.black_list = [625,747,745,738,62,750,746,737,739,762]

        print('load %d Ts, %d Ks, %d vertices' % (self.Ts.size(0),self.Ks.size(0),self.vs.size(0)))


        self._all_imgs = None
        self._all_Ts = None
        self._all_Ks = None
        self._all_width_height = None


        if is_need_all_data:
            self._prepare_rgb()

        print('done preparation.')




    def __len__(self):
        return self.Ts.size(0)

    def __getitem__(self, index, need_transform = True):

        if index in self.black_list:
            index = 0 
        
        img = Image.open(os.path.join(self.file_path,self.files[index]))
        if need_transform:
            img = self.transforms(img)

        return img, self.vs, index, self.Ts[index], self.Ks[index], self.near_far_size

    def get_vertex_num(self):
        return torch.Tensor([self.vs.size(0)])

    def _prepare_rgb(self):

        imgs = []
        Ts = []
        Ks = []
        width_height = []
        for i in range(self.__len__()):
            img, _, _, T, K = self.__getitem__(i, need_transform = True)
            img = img.permute(1,2,0)
            imgs.append(img*255.0)
            Ts.append(T)
            Ks.append(K)
            width_height.append( torch.Tensor([img.shape[1],img.shape[0]]))

        imgs = torch.stack(imgs)
        Ts = torch.stack(Ts)
        Ks = torch.stack(Ks)
        width_height = torch.cat(width_height)

        self._all_imgs = imgs.byte()
        self._all_Ts = Ts
        self._all_Ks = Ks
        self._all_width_height = width_height.int()

        return self._all_imgs, self._all_Ts, self._all_Ks, self._all_width_height






