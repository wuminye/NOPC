import torch
from ..UNet import UNet
from layers.pcpr_layer import PCPRModel
import pcpr
import numpy as np 


class PCPRender(torch.nn.Module):
    def __init__(self, feature_dim, tar_width, tar_height, dataset = None):
        super(PCPRender, self).__init__()
        self.feature_dim = feature_dim
        self.tar_width = tar_width
        self.tar_height = tar_height

        add_rgb_input = 0

        if dataset is not None:
            add_rgb_input = 3

        self.pcpr_layer = PCPRModel(tar_width, tar_height)
        self.unet = UNet(feature_dim  + 3 + add_rgb_input, # input channel: feature[feature_dim] + depth[1] + viewin directions[3] + %%%points color[3]%%%(no used for now)
                #   4) # output channel: 3 RGB 
                  3, 1) # output channel: 3 RGB 1
        self.unet = self.unet.cuda()

        self.dataset = dataset

        # generate meshgrid
        xh, yw = torch.meshgrid([torch.arange(0,tar_height), torch.arange(0,tar_width)])
        self.coord_meshgrid = torch.stack([yw, xh, torch.ones_like(xh)],dim =0).float()
        self.coord_meshgrid = self.coord_meshgrid.view(1,3,-1)
        self.coord_meshgrid = self.coord_meshgrid.cuda()






    def forward(self, point_features, default_features,
           point_clouds,
           cam_intrinsic, cam_extrinsic, 
           near_far_max_splatting_size, num_points, inds = None):

        batch_num = cam_intrinsic.size(0)



        # out_feature (batch, feature_dim, tar_height, tar_width )
        # out_depth (batch, 1, tar_height, tar_width )
        out_feature, out_depth = self.pcpr_layer(point_features, default_features,
                                point_clouds,
                                cam_intrinsic, cam_extrinsic, 
                                near_far_max_splatting_size, num_points)

        # generate viewin directions
        Kinv = torch.inverse(cam_intrinsic)
        coord_meshgrids = self.coord_meshgrid.repeat(batch_num,1,1)
        dir_in_camera = torch.bmm(Kinv, coord_meshgrids)
        dir_in_camera = torch.cat([dir_in_camera, torch.ones(batch_num,1,dir_in_camera.size(2)).cuda()],dim = 1)
        dir_in_world = torch.bmm(cam_extrinsic, dir_in_camera)
        dir_in_world = dir_in_world / dir_in_world[:,3:4,:].repeat(1,4,1)
        dir_in_world = dir_in_world[:,0:3,:]
        dir_in_world = torch.nn.functional.normalize(dir_in_world, dim=1)
        dir_in_world = dir_in_world.reshape(batch_num,3,self.tar_height,self.tar_width)

        #set direction to zeros for depth==0
        depth_mask = out_depth.repeat(1,3,1,1)
        dir_in_world[depth_mask==0] = 0


        #render RGB images


        out_rgb = None
        if self.dataset is not None:

            if inds is None:
                inds = torch.Tensor(0).cuda()

            
            out_rgb = torch.zeros(out_depth.size(0),out_depth.size(2),out_depth.size(3),3).byte()

            imgs, Ts, Ks, width_height = self.dataset._all_imgs, self.dataset._all_Ts.clone(), self.dataset._all_Ks.clone(), self.dataset._all_width_height.clone()

            Ts = Ts.cuda()
            Ks = Ks.cuda()
            width_height = width_height.cuda().int()

            Ts = torch.cat([Ts[:,0:3,2], Ts[:,0:3,0],
                            Ts[:,0:3,1],Ts[:,0:3,3]],dim = 1)

            for b in range(batch_num):
                out_rgb_index = torch.zeros(out_depth.size(2),out_depth.size(3),3).cuda().short()

                tar_T = torch.cat([cam_extrinsic[b,0:3,2], cam_extrinsic[b,0:3,0],
                            cam_extrinsic[b,0:3,1],cam_extrinsic[b,0:3,3]])

                pcpr.rgb_index_calc( cam_intrinsic[b], tar_T,
                            Ks, Ts, width_height,
                            inds,
                            out_depth[b][0], out_rgb_index)

                out_rgb_index = out_rgb_index.cpu()

               
                pcpr.rgb_index_render(out_rgb_index, imgs, out_rgb[b])


            out_rgb = out_rgb.cuda().float()/255.0

            out_rgb = out_rgb.permute(0,3,1,2)

                

            





        # fuse all features
        fused_features = torch.cat([out_feature,dir_in_world],dim = 1)
        if self.dataset is not None:
            fused_features = torch.cat([fused_features,out_rgb],dim = 1)

        # rendering
        x = self.unet(fused_features)

       
        
    
        return x, out_depth.detach(), out_feature.detach(), dir_in_world.detach(), out_rgb