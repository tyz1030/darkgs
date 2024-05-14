import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from lietorch import SO3, LieGroupParameter

class LightBaseLie(nn.Module):
    name = "Light Base"
    def __init__(self, t_x: float = -0.2266, t_y: float = -0.0022, t_z: float = 0.0761, r_x: float = -0.0027, r_y: float = 0.36, r_z: float = -0.027) -> None:
        super().__init__()
        # Translation vector and Rotation vector are light-to-camera transformation.
        # Light is using same convention to camera coordinate: x-right, y-down, z-forward
        # Apply following transformation will transform [ ] from light coordinate to camera coordiante
        self._t_vec = nn.Parameter(torch.tensor([[[t_x, t_y, t_z]]], device="cuda:0", dtype=torch.float32), requires_grad=True) #[-0.4, 0.0, 0.0]
        
        _r_vec = torch.tensor([[[r_x, r_y, r_z]]], device="cuda:0", dtype=torch.float32)
        self._r_l2c_SO3 = LieGroupParameter(SO3.exp((_r_vec)))

        self.gamma_log = nn.Parameter(torch.tensor(-1.59), requires_grad=True)
        self.tau_log = nn.Parameter(torch.tensor(-1.59), requires_grad=True)

    def set_t_vec(self, t_tuple, require_grad: bool = True)->None:
        self._t_vec = nn.Parameter(torch.tensor([[[t_tuple[0], t_tuple[1], t_tuple[2]]]], device="cuda:0", dtype=torch.float32), requires_grad=require_grad)

    def set_r_vec(self, r_tuple)->None:
        _r_vec = torch.tensor([[[r_tuple[0], r_tuple[1], r_tuple[2]]]], device="cuda:0", dtype=torch.float32)
        self._r_l2c_SO3 = LieGroupParameter(SO3.exp((_r_vec)))

    def set_sigma(self, sigma: [float], require_grad: bool = True)->None:
        pass

    @property
    def gamma(self):
        return torch.exp(self.gamma_log)

    def set_gamma(self, gamma: float, require_grad: bool = True)->None:
        self.gamma_log = nn.Parameter(torch.log(torch.tensor(gamma)), requires_grad=require_grad)

    @property
    def tau(self):
        return torch.exp(self.tau_log)

    def set_tau(self, tau: float, require_grad: bool = True)->None:
        self.tau_log = nn.Parameter(torch.log(torch.tensor(tau)), requires_grad=require_grad)
    
    def c2l(self, pts)-> Tensor:
        return self._r_l2c_SO3.inv().act(pts)
    
    def l2c(self, pts)-> Tensor:
        return self._r_l2c_SO3.act(pts)
    
    def t_c2l(self)-> Tensor:
        return -self.c2l(self._t_vec)

    def t_l2c(self)-> Tensor:
        return self._t_vec
    
    def lorenzian(self, x_in_l: Tensor)->Tensor:
        '''
            Lorentzian Function Model, used to model light fall off with distance
            Args:
                x_in_l: 3D point in the light's coordinate system
            Return:
                i_falloff: fall off of light intensity
        '''
        dist_square = torch.sum(x_in_l*x_in_l, dim=-1)
        i_falloff = 1/torch.pow(self.tau+dist_square, self.gamma)
        return i_falloff
    
    @property
    def name(self)->str:
        return self.name

class LightMLPBase(LightBaseLie):
    name: str = "MLP"
    def __init__(self) -> None:
        super().__init__()

    def create_mlp(self, width: int = 20, input_dim: int = 1, output_dim: int = 1, depth: int = 2)->nn.Sequential:
        layers = [nn.Linear(input_dim, width), nn.Softplus()]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Softplus())
        layers.append(nn.Linear(width, output_dim))
        layers.append(nn.Softplus())
        mlp = nn.Sequential(*layers)
        for m in mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        return mlp
    
    def forward(self, pts: Tensor)-> Tensor:
        '''
        Args:
            pts: points in camera coordinate
        '''
        x_in_l = self.c2l(pts)+self.t_c2l()
        i_falloff = self.lorenzian(x_in_l).to(torch.float32)
        i_mlp = self.mlp_process(x_in_l).squeeze(-1)
        # return i_mlp
        return i_falloff*i_mlp
    
    def mlp_process(self, x_in_l: Tensor)->Tensor:
        raise NotImplementedError("MLP Process not implemented")
                
        
class LightMLP1D(LightMLPBase):
    name = "1D MLP"
    # Not using due to nosies and peaks in MLP especially at transit of edge.
    def __init__(self) -> None:
        super().__init__()
        self.mlp = self.create_mlp(width=20, input_dim=5, output_dim=1, depth=3)
    
    def mlp_process(self, x_in_l: Tensor)->Tensor:
        '''
            2D Gaussian model
            Args:
                x_in_l: 3D point in the light's coordinate system
        '''
        x_in_l = x_in_l/x_in_l[..., 2, None]
        x_y = (torch.square(x_in_l[..., 0])+torch.square(x_in_l[..., 1])).to(torch.float32)[..., None]
        x_y = torch.sqrt(x_y)
        # x_y = x_y*2
        
        
        ######################################
        # x_y = torch.min(x_y)*torch.ones_like(x_y)
        ######################################
        
        
        
        x_y1 = torch.cos(x_y)
        x_y2 = torch.cos(2*x_y)
        x_y3 = torch.cos(4*x_y)
        x_y4 = torch.cos(8*x_y)
        x_y  = torch.concat([x_y, x_y1, x_y2, x_y3, x_y4], dim=-1)

        i_mlp = self.mlp(x_y) #*0.1
        return i_mlp #*1.4 # tcs1.4


class LightMLP2D(LightMLPBase):
    name = "2D MLP"
    # Not using due to nosies and peaks in MLP especially at transit of edge.
    def __init__(self) -> None:
        super().__init__()
        self.mlp = self.create_mlp(width=128, input_dim=22, output_dim=1, depth=4)
    
    def mlp_process(self, x_in_l: Tensor)->Tensor:
        '''
            2D Gaussian model
            Args:
                x_in_l: 3D point in the light's coordinate system
        '''
        x_in_l = x_in_l/x_in_l[..., 2, None]
        x_y = torch.abs((x_in_l[..., 0:2])).to(torch.float32)
        xy1 = torch.cos(x_y)
        xy2 = torch.cos(2*x_y)
        xy3 = torch.cos(4*x_y)
        xy4 = torch.cos(8*x_y)
        xy5 = torch.cos(16*x_y)
        xy1_ = torch.sin(x_y)
        xy2_ = torch.sin(2*x_y)
        xy3_ = torch.sin(4*x_y)
        xy4_ = torch.sin(8*x_y)
        xy5_ = torch.sin(16*x_y)
        # x_y = torch.concat([x_y, torch.sum(x_y, dim=-1, keepdim=True)], dim=-1)
        x_y  = torch.concat([x_y, xy1, xy2, xy3, xy4, xy5, xy1_, xy2_, xy3_, xy4_, xy5_], dim=-1)
        i_mlp = self.mlp(x_y)
        return i_mlp
    
class PointLightSource(LightBaseLie):
    name = "Point Light Souce"
    def __init__(self) -> None:
        super().__init__()

    def forward(self, pts: Tensor)->Tensor:
        x_in_l = self.c2l(pts)+self.t_c2l()
        return self.lorenzian(x_in_l)
    

class Light2DGaussian(LightBaseLie):
    name = "Gaussian2D"
    def __init__(self) -> None:
        super().__init__()
        self.sigma = nn.Parameter(Tensor([13.4, 14.763]), requires_grad=True) # 2D distribution of light intensity

    def set_sigma(self, sigma: [float], require_grad: bool = True)->None:
        self.sigma = nn.Parameter(Tensor([sigma[0], sigma[1]]), requires_grad=require_grad)

    def forward(self, pts: Tensor)-> Tensor:
        '''
        Args:
            pts: points in camera coordinate
        '''
        x_in_l = self.c2l(pts)+self.t_c2l()
        i_falloff = self.lorenzian(x_in_l)
        i_gaussian2d = self.gaussian2d(x_in_l)
        return i_falloff*i_gaussian2d
    
    def gaussian2d(self, x_in_l: Tensor)->Tensor:
        '''
            2D Gaussian model
            Args:
                x_in_l: 3D point in the light's coordinate system
        '''
        x_in_l = x_in_l/x_in_l[..., 2, None]
        x = torch.arctan(torch.abs(x_in_l[..., 0]))
        y = torch.arctan(torch.abs(x_in_l[..., 1]))
        i_gaussian2d = torch.exp(-(x*self.sigma[0])**2-(y*self.sigma[1])**2)
        return i_gaussian2d
    

class Light1DGaussian(LightBaseLie):
    name = "Gaussian 1D"
    def __init__(self) -> None:
        super().__init__()
        # Gaussian std
        self.sigma = nn.Parameter(torch.tensor(11.0), requires_grad=True) # 2D distribution of light intensity

        # self.max_cap = nn.Parameter(torch.tensor(0.55), requires_grad=True)

    def set_sigma(self, sigma: [float], require_grad: bool = True)->None:
        self.sigma = nn.Parameter(torch.tensor(sigma[0]), requires_grad=require_grad)

    def forward(self, pts: Tensor)-> Tensor:
        '''
            Args:
                pts: points in camera coordinate
        '''
        x_in_l = self.c2l(pts)+self.t_c2l()
        i_falloff = self.lorenzian(x_in_l)
        i_gaussian1d = self.gaussian1d(x_in_l)
        return i_falloff*i_gaussian1d
    
    def gaussian1d(self, x_in_l: Tensor)->Tensor:
        '''
            2D Gaussian model
            Args:
                x_in_l: 3D point in the light's coordinate system
        '''
        x_in_l = x_in_l/x_in_l[..., 2, None]
        x = torch.arctan(torch.sqrt(x_in_l[..., 0]**2+x_in_l[..., 1]**2))
        i_gaussian1d = torch.exp(-(x*self.sigma)**2)
        return i_gaussian1d
    

class LightFactory:
    @staticmethod
    def get_light(Light_type):
        if Light_type == "Gaussian2D":
            return Light2DGaussian()
        elif Light_type == "Gaussian1D":
            return Light1DGaussian()
        elif Light_type == "PointLightSource":
            return PointLightSource()
        elif Light_type == "1DMLP":
            return LightMLP1D()
        elif Light_type == "2DMLP":
            return LightMLP2D()
        else:
            raise ValueError(f"Light type {Light_type} not recognized!")

# test code
if __name__ == "__main__":

    light_gauss = LightFactory().get_light("Gaussian1D")
    # light_gauss.set_t_vec(Tensor([-0.4, 0.0, 0.0]))
    # light_gauss.set_r_vec(Tensor([0.0, 0.8, 0.0]))

    # Test Points
    grid_x, grid_y = torch.meshgrid(
            torch.arange(401), torch.arange(401), indexing="xy",
        )
    
    grid_x = (grid_x-200)/1000
    grid_y = (grid_y-200)/1000
    grid_z = 0.7*torch.ones([401, 401])
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    grid = torch.permute(grid, [2,0,1])
    print(grid.shape)
    grid = grid.view(3, -1)
    grid = grid.transpose(0,1)
    print(grid.shape)
    print(grid.dtype)
    grid = grid.cuda()
    res = light_gauss(grid[None])
    res = res.view(401, 401)
    print(res.shape)
    import matplotlib.pyplot as plt
    plt.imshow(res.detach().cpu().numpy())
    plt.show()
    # print(grid_y)
    # print(grid_z)

    # OLD TEST CODE
    # k = 4
    # gmm = LightGMM1D(k = k)
    # light_mlp = LightMLP1D().cuda()
    # light_sigmoid = LightSigmoid1D(k=k).cuda()

    # samples = torch.linspace(0, 4, 4000)[..., None].cuda()
    # log_probs = light_sigmoid(samples)

    # import matplotlib.pyplot as plt
    # plt.plot(samples.cpu().detach().numpy(), log_probs.cpu().detach().numpy())
    # plt.show()

    # y_target = torch.zeros(4000)
    # y_target[0:1000] = 2
    # y_target[1000:2000] = 1
    # y_target[2000:3000] = 0.5
    # y_target = y_target*log_probs.cpu().detach()
    # y_target = y_target.cuda()

    # plt.plot(samples.cpu().detach().numpy(), y_target.cpu().detach().numpy())
    # plt.show()

    # learning_rate = 1e-3
    # optimizer = torch.optim.Adam(
    #     light_sigmoid.parameters(), lr=learning_rate)
    # loss_function = nn.MSELoss()
    # for t in range(100000):
    #     prediction = light_sigmoid(samples)
    #     loss = loss_function(prediction, y_target)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     if not t%10000: print(t)

    # plt.plot(samples.cpu().detach().numpy(), light_sigmoid(samples).cpu().detach().numpy())
    # plt.plot(samples.cpu().detach().numpy(), y_target.cpu().detach().numpy())
    # plt.show()
