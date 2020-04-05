

from .UNet import UNet
from .PCPR import PCPRender
from .PCPR import PCPRParameters
from .generatic_model import Generatic_Model





def build_model(cfg, vertex_list, dataset = None):

    model = Generatic_Model(cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1],
                 vertex_list, cfg.MODEL.FEATURE_DIM, dataset =dataset, use_rgb = cfg.INPUT.USE_RGB)

    return model
