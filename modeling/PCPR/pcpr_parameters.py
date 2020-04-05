import torch

class PCPRParameters(torch.nn.Module):
    def __init__(self, vertices_num, feature_dim):
        super(PCPRParameters, self).__init__()
        self.feature_dim = feature_dim
        self.vertices_num = vertices_num
        self.p_parameters = torch.nn.ParameterList()

        self.default_features = torch.nn.Parameter(torch.randn(feature_dim, 1).cuda())

        for i in range(self.vertices_num.size(0)):
           self.p_parameters.append(torch.nn.Parameter(torch.randn(feature_dim, self.vertices_num[i]).cuda()))
        
        ## just for test, need reimplement here.
        # self.p_parameters = torch.nn.Parameter(torch.randn(feature_dim, self.vertices_num[0]).cuda())



        
    def forward(self, indexes):
        res = []
        v_num = []
        p_param = []

        
        for i in indexes:
            #res.append(self.p_parameters[i])
            v_num.append(self.vertices_num[i]) 
            p_param.append(self.p_parameters[i])
        p_params = torch.cat(p_param, dim = 1)
        # return None, self.default_features, torch.Tensor(v_num).int()
        return p_params, self.default_features, torch.Tensor(v_num).int()




