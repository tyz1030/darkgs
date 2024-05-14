from .brdf import BRDFFactory
from .light import LightFactory
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from lietorch import SO3
from scene.cameras import Camera

class ShadingModel(nn.Module):
    def __init__(self, brdf: str = "Lambertian", light: str = "Gaussian1D", albedo: float = 100.,  device: str = "gpu") -> None:        
        super(ShadingModel, self).__init__()
        self.light = LightFactory.get_light(light)
        self.brdf = BRDFFactory.get_brdf(brdf)
        self.albedo_log = nn.Parameter(torch.tensor(albedo), requires_grad=True)
        self.ambient_light_log = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        # target's own coordinate system used as world coordinate. Right-Down-Forward. Hardcoded here that normal pointing from the origin of the target to camera.

        self.scaling_factor = nn.Parameter(torch.tensor(0.1), requires_grad=True) # When calibrating, should not optimize this scaling factor
        self._warmup_factor : float = 1.0 # DEPRECATED 

        self.set_optimizer()


    def set_optimizer(self)->None:
        l = [
            {'params': [self.ambient_light_log], 'lr': 0.001, "name": "ambient_light"},
            {'params': [self.scaling_factor], 'lr': 0.001, "name": "scaling"},
            # uncomment for extensive finetuning (experimental)
            # {'params': [self.light.tau_log], 'lr': 0.001, "name": "tau"}, 
            # {'params': [self.light.gamma_log], 'lr': 0.001, "name": "gamma"},
            # {'params': [self.light._r_l2c_SO3], 'lr': 0.001, "name": "r_vec"},
            # {'params': [self.light._t_vec], 'lr': 0.001, "name": "_t_vec"},
            # {'params': [self.light.sigma], 'lr': 0.001, "name": "sigma"},
            # {'params': [self.light.mlp.parameters()], 'lr': 0.001, "name": "mlp0"},
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    @property
    def ambient_light(self):
        return torch.exp(self.ambient_light_log)
    

    @property
    def albedo(self):
        return torch.exp(self.albedo_log)
    
    def set_albedo(self, albedo: float, require_grad: bool = True) -> None:
        self.albedo_log = nn.Parameter(torch.log(torch.tensor(albedo)), requires_grad=require_grad)


    # DEPRECATED FUNCTION
    # @property
    # def warmup_factor(self)->float:
    #     return self._warmup_factor
    
    # # DEPRECATED FUNCTION
    # @warmup_factor.setter
    # def warmup_factor(self, value: float)->None:
    #     if 0. <= value <= 1.:
    #         self._warmup_factor = value
    #     else:
    #         raise ValueError(f"Invalid input for warmup factor {value}. Please check input.")

    def set_ambient_light(self, ambient_light: float, require_grad: bool = True) -> None:
        self.ambient_light_log = nn.Parameter(torch.log(torch.tensor(ambient_light)), requires_grad=require_grad)
        self.set_optimizer()
        
    def set_scaling_factor(self, scaling_factor: float, require_grad: bool = True) -> None:
        self.scaling_factor = nn.Parameter((torch.tensor(scaling_factor)), requires_grad=require_grad)
        self.set_optimizer()

    def forward(self, pts: Tensor, camera: Camera, albedos: Tensor, normals: Tensor)-> Tensor:
        '''
        Arguments:
            pts: 3D points in world coordinate.
        '''
        # Monocular SfM poses are up-to-scale. So need to optimize for scale here.
        pts_scaled = pts*self.scaling_factor
        t_w2c: Tensor = camera.world_view_transform[3:4, :3]*self.scaling_factor
        rmat_w2c: Tensor = camera.world_view_transform[:3, :3]
        t_c2w = -torch.matmul(rmat_w2c, t_w2c.transpose(0,1))
        t_l2c = self.light.t_l2c().squeeze(0).transpose(0,1) 
        p_l_in_w = torch.matmul(rmat_w2c, t_l2c)+t_c2w
        light_dir = -p_l_in_w.transpose(0, 1)+pts_scaled
        view_dir = t_c2w.transpose(0,1)-pts_scaled

        reflectance = self.brdf(view_dir, normals, light_dir)[..., None] # n*1
        pts_in_cam = torch.matmul(pts_scaled, rmat_w2c)+t_w2c
        incident_light = self.light(pts_in_cam.unsqueeze(0)).transpose(0,1) # n*1, convert to camera coordinate first
        
        reflected_light = F.softplus(albedos)*(incident_light+self.ambient_light)*reflectance
        return reflected_light
    