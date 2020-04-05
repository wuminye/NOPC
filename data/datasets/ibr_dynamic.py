import torch
import cv2
import numpy as np
import os
from .utils import campose_to_extrinsic, read_intrinsics
from PIL import Image
import torchvision


class IBRDynamicDataset(torch.utils.data.Dataset):

    def __init__(self,data_folder_path, use_mask, center_coords, transforms, near_far_size):
        super(IBRDynamicDataset, self).__init__()

        self.data_folder_path = data_folder_path
        self.use_mask = use_mask
        self.center_coords = center_coords

        self.file_path = os.path.join(data_folder_path,'img')

        self.vs = []
        self.vs_rgb = []
        self.vs_num = []
        self.vs_index =[]

        sum_tmp = 0
        tmp = np.loadtxt(os.path.join(data_folder_path,'pointclouds/frame1.xzy'), usecols = (0,1,2))
        vs_tmp = tmp[:,0:3] 
        vs_rgb_tmp = tmp[:,3:6]
        self.vs_index.append(sum_tmp)
        self.vs.append(torch.Tensor(vs_tmp))
        self.vs_rgb.append(torch.Tensor(vs_rgb_tmp))
        self.vs_num.append(vs_tmp.shape[0])
        sum_tmp = sum_tmp + vs_tmp.shape[0]


        self.vs = torch.cat( self.vs, dim=0 )
        self.vs_rgb = torch.cat( self.vs_rgb, dim=0 )

        #self.vs = self.vs + 3.0*torch.randn_like(self.vs)
        
        

        camposes = np.loadtxt(os.path.join(data_folder_path,'CamPose.inf'))
        self.Ts = torch.Tensor( campose_to_extrinsic(camposes) )
        self.cam_num = self.Ts.size(0)
        
        self.Ks = torch.Tensor(read_intrinsics(os.path.join(data_folder_path,'Intrinsic.inf')))

        '''
        for i in range(self.Ks.size(0)):
            # self.Ks[i] = self.Ks[i] / (4000/500)
            # self.Ks[i] = self.Ks[i] / (800/600)
            self.Ks[i] = self.Ks[i] / (1920/720)

        self.Ks[:,2,2] = 1
        '''


        self.transforms = transforms
        self.near_far_size = torch.Tensor(near_far_size)

        #self.black_list = [625,747,745,738,62,750,746,737,739,762]

        print('load %d Ts, %d Ks, %d vertices' % (self.Ts.size(0),self.Ks.size(0),self.vs.size(0)))


        self._all_imgs = None
        self._all_Ts = None
        self._all_Ks = None
        self._all_width_height = None

        print('dataset initialed.')




    def __len__(self):
        return self.cam_num

    def __getitem__(self, index, need_transform = True):

        frame_id = index // self.cam_num
        cam_id = index % self.cam_num
        
        img_rgb = Image.open(os.path.join(self.file_path,'%d/img_%04d.jpg' % ( frame_id, cam_id+1)))
        img_alpha = Image.open(os.path.join(self.file_path,'%d/img_%04d_alpha.png' % ( frame_id, cam_id+1)))

        # if need_transform:
        #     img_rgb = self.transforms(img_rgb)
        #     img_alpha = self.transforms(img_alpha)
        # img=torch.cat([img_rgb,img_alpha],0)  

        img_mask=img_alpha.copy()
        if self.use_mask:
            unique_cam_num=len(self.center_coords)//2
            img_per_num=self.cam_num//unique_cam_num
            unique_cam_id = cam_id // img_per_num
            img,K,T,img_mask, ROI = self.transforms(img_rgb,self.Ks[cam_id],self.Ts[cam_id],img_mask,self.center_coords[unique_cam_id*2:unique_cam_id*2+2])
        else:
            img,K,T,img_mask, ROI = self.transforms(img_rgb,self.Ks[cam_id],self.Ts[cam_id],img_mask)
        img = torch.cat([img,img_mask[0:1,:,:]], dim=0)
        
        img=torch.cat([img,ROI],dim=0)

        # return img, self.vs[self.vs_index[frame_id]:self.vs_index[frame_id]+self.vs_num[frame_id],:], index, self.Ts[cam_id], self.Ks[cam_id], self.near_far_size, self.vs_rgb[self.vs_index[frame_id]:self.vs_index[frame_id]+self.vs_num[frame_id],:]
        return img, self.vs[self.vs_index[frame_id]:self.vs_index[frame_id]+self.vs_num[frame_id],:], index, T, K, self.near_far_size, self.vs_rgb[self.vs_index[frame_id]:self.vs_index[frame_id]+self.vs_num[frame_id],:]

    def get_vertex_num(self):
        return torch.Tensor(self.vs_num)





