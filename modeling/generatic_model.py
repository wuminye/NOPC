import torch
from .UNet import UNet
from .PCPR import PCPRender
from .PCPR import PCPRParameters




class Generatic_Model(torch.nn.Module):


    def __init__(self, tar_width, tar_height, vertex_list, feature_dim, dataset=None, use_rgb = False ):
        super(Generatic_Model, self).__init__()
        self.dataset = dataset
        self.pcpr_parameters = PCPRParameters(vertex_list, feature_dim)
        self.render = PCPRender(feature_dim,tar_width,tar_height, dataset = dataset)

        input_channels = 0
        self.use_rgb = use_rgb
        if use_rgb:
            input_channels = 3


    def forward(self, point_indexes, in_points, K, T,
           near_far_max_splatting_size, num_points, rgbs, inds=None):


        num_points = num_points.int()

        # _,default_features,_ = self.pcpr_parameters(point_indexes)
        p_parameters,default_features,_ = self.pcpr_parameters(point_indexes)

        batch_size = K.size(0)
        dim_features = default_features.size(0)

        m_point_features = []
        beg = 0

        for i in range(batch_size):
            
            m_point_features.append(p_parameters)

        point_features = torch.cat(m_point_features, dim = 1).requires_grad_()
        

        res,depth,features,dir_in_world, rgb = self.render(point_features, default_features,
                             in_points,
                             K, T,
                             near_far_max_splatting_size, num_points, inds)

        
        return res,depth, features, dir_in_world, rgb, point_features

        
