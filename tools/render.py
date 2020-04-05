import argparse
import os
import sys
from os import mkdir
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append('..')
sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer, WarmupMultiStepLR
from layers import make_loss
from utils.logger import setup_logger
from data.datasets.utils import campose_to_extrinsic, read_intrinsics
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import cv2
torch.cuda.set_device(0)

model_path=sys.argv[1]
epoch = sys.argv[2]
camposFile=os.path.join(sys.argv[3],'CamPose.inf')
intriFile=os.path.join(sys.argv[3],'Intrinsic.inf')
para_file = 'nr_model_%s.pth' % epoch

cfg.merge_from_file(os.path.join(model_path,'config.yml'))
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.freeze()

writer = SummaryWriter(log_dir=os.path.join(model_path,'tensorboard_test'))
test_loader, vertex_list,dataset = make_data_loader(cfg, is_train=False)

model = build_model(cfg, vertex_list)
model.load_state_dict(torch.load(os.path.join(model_path,para_file),map_location='cpu'))
model.eval()
model = model.cuda()


for batch in test_loader:
    in_points = batch[1].cuda()
    K = batch[2].cuda()
    T = batch[3].cuda()
    near_far_max_splatting_size = batch[5]
    num_points = batch[4]
    point_indexes = batch[0]
    target = batch[-1].cuda()
    break
    

picScale=cfg.INPUT.SIZE_TEST[0]/cfg.INPUT.SIZE_RAW[0]
# picScale=720/1920

camposes = np.loadtxt(camposFile)
Ts = torch.Tensor( campose_to_extrinsic(camposes) )
camNum = Ts.size(0)

Ks = read_intrinsics(intriFile)
for i in range(camNum):
    Ks[i,0:2,:]=Ks[i,0:2,:]*picScale
Ks = torch.Tensor(Ks)
    

if not os.path.exists(os.path.join(model_path,'res_%s'%epoch)):
    os.mkdir(os.path.join(model_path,'res_%s'%epoch))
if not os.path.exists(os.path.join(model_path,'res_%s/rgb'%epoch)):
    os.mkdir(os.path.join(model_path,'res_%s/rgb'%epoch))
if not os.path.exists(os.path.join(model_path,'res_%s/alpha'%epoch)):
    os.mkdir(os.path.join(model_path,'res_%s/alpha'%epoch))
if not os.path.exists(os.path.join(model_path,'res_%s/rgba'%epoch)):
    os.mkdir(os.path.join(model_path,'res_%s/rgba'%epoch))

for ID in range(camNum):
    T = Ts[ID:ID+1,:,:].cuda()
    K = Ks[ID:ID+1,:,:].cuda()
    res,depth,features,dir_in_world,rgb,point_features = model(point_indexes, in_points, K, T,
                        near_far_max_splatting_size, num_points,target)

    depth = (depth - torch.min(depth))
    depth = depth / torch.max(depth)
        
    img_t = res.detach().cpu()[0]
    mask_t = img_t[3:4,:,:]
    img = cv2.cvtColor(img_t.permute(1,2,0).numpy()*255.0,cv2.COLOR_BGR2RGB)
    mask = mask_t.permute(1,2,0).numpy()*255.0
    rgba=img*mask/255.0+(255.0-mask)
    
    cv2.imwrite(os.path.join(model_path,'res_%s/rgb/img_%04d.jpg'%(epoch,ID+1)),img)
    cv2.imwrite(os.path.join(model_path,'res_%s/alpha/img_%04d.jpg'%(epoch,ID+1)  ),mask)
    cv2.imwrite(os.path.join(model_path,'res_%s/rgba/img_%04d.jpg'%(epoch,ID+1)  ),rgba)

print('Render done.')


