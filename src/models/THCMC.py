import torch as th
import torch.nn as nn
import helpers
from lib.loss import Loss
from lib.optimizer import Optimizer
from lib.backbones import Backbones, MLP

from lib.fusion import get_fusion_module
from models.clustering_module import DDC
from models.model_base import ModelBase
from torch.nn.functional import normalize
from graph.GMC import GMC


class THCMC(ModelBase):
    def __init__(self, cfg):
        """

        :param cfg: Model config. See `config.defaults.CoMVC` for documentation on the config object.
        """
        super().__init__()

        self.cfg = cfg
        self.output = self.hidden = self.fused = self.backbone_outputs = self.projections = self.label_projections = None

        # Define Backbones and Fusion modules

        self.backbones = Backbones(cfg.backbone_configs)
        self.fusion = get_fusion_module(cfg.fusion_config, self.backbones.output_sizes)

        bb_sizes = self.backbones.output_sizes
        assert all([bb_sizes[0] == s for s in bb_sizes]), f"CoMVC requires all backbones to have the same " \
                                                          f"output size. Got: {bb_sizes}"

        if cfg.projector_config is None:
            self.projector = nn.Identity()
        else:
            self.projector = MLP(cfg.projector_config, input_size=bb_sizes[0])

        # Define clustering module
        self.ddc = DDC(input_dim=256, cfg=cfg.cm_config)

        # Define loss-module
        self.loss = Loss(cfg=cfg.loss_config)
        # Initialize weights.
        self.apply(helpers.he_init_weights)
        # Instantiate optimizer
        self.optimizer = Optimizer(cfg.optimizer_config, self.parameters())

        self.instance_projector = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        )



    def forward(self, views):
        self.backbone_outputs = self.backbones(views)
        self.views = []
        for i in range (len(views)):
            self.views.append(views[i])
        self.fused = self.fusion(self.backbone_outputs)
        self.projections = self.projector(th.cat(self.backbone_outputs, dim=0))
        self.z = []
        for i in range(len(views)):
            self.z_ins = normalize(self.instance_projector(self.backbone_outputs[i]), dim=1)
            # self.z_ins = self.instance_projector(self.backbone_outputs[i])
            self.z.append(self.z_ins)
        # self.z1 = normalize(self.instance_projector(self.backbone_outputs[0]), dim=1)
        # self.z2 = normalize(self.instance_projector(self.backbone_outputs[1]), dim=1)
        self.backbone_ddc = []
        for i in range(len(views)):
            self.backbone_clus, _ = self.ddc(self.z[i])
            self.backbone_ddc.append(self.backbone_clus)
        # for i in range(len(self.backbone_ddc)):
        #     for j in range (self.backbone_ddc[0].size(0)):
        #         row = self.backbone_ddc[i][j]
        #         row[row<=0.05]=0
        #         positive_values = row[row>0.05]
        #         if positive_values.numel()>0:
        #             positive_values = th.nn.functional.softmax(positive_values, dim=0)
        #             row[row>0.05] = positive_values
        #     self.backbone_ddc[i][j] =row
        self.output, self.hidden = self.ddc(self.fused)
        self.label_projections = []
        for i in range (len(views)):
            self.label_projections.append(self.projector(th.cat((self.backbone_ddc[i], self.output), dim=0)))
        # undate fused via diffusion
        # self.fused = self.DiffProcess.p_sample(self.SDNet, self.fused, steps=0, sampling_noise=False)
        return self.output

