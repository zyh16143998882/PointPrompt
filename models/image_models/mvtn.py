import torch
from torch.autograd import Variable
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .util import unit_spherical_grid, batch_tensor
from torch import nn



class CircularViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_elevation=35.0, canonical_distance=2.2, transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        self.input_view_noise = input_view_noise
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim = torch.linspace(-180, 180, self.nb_views+1)[:-1] - 90.0
        views_elev = torch.ones_like(
            views_azim, dtype=torch.float, requires_grad=False)*canonical_elevation
        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((c_batch_size, self.nb_views), device=c_views_dist.device) - 0.5)
        if self.input_view_noise > 0.0 and self.training:
            c_views_azim = c_views_azim + \
                torch.normal(0.0, 180.0 * self.input_view_noise,
                             c_views_azim.size(), device=c_views_azim.device)
            c_views_elev = c_views_elev + \
                torch.normal(0.0, 90.0 * self.input_view_noise,
                             c_views_elev.size(), device=c_views_elev.device)
            c_views_dist = c_views_dist + \
                torch.normal(0.0, self.canonical_distance * self.input_view_noise,
                             c_views_dist.size(), device=c_views_dist.device)
        return c_views_azim, c_views_elev, c_views_dist


class SphericalViewSelector(nn.Module):
    def __init__(self, nb_views=12,canonical_distance=2.2, transform_distance=False, input_view_noise=0.0):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        self.input_view_noise = input_view_noise
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_azim, views_elev = unit_spherical_grid(self.nb_views)
        views_azim, views_elev = torch.from_numpy(views_azim).to(
            torch.float), torch.from_numpy(views_elev).to(torch.float)
        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((c_batch_size, self.nb_views), device=c_views_dist.device) - 0.5)
        if self.input_view_noise > 0.0 and self.training:
            c_views_azim = c_views_azim + \
                torch.normal(0.0, 180.0 * self.input_view_noise,
                             c_views_azim.size(), device=c_views_azim.device)
            c_views_elev = c_views_elev + \
                torch.normal(0.0, 90.0 * self.input_view_noise,
                             c_views_elev.size(), device=c_views_elev.device)
            c_views_dist = c_views_dist + \
                torch.normal(0.0, self.canonical_distance * self.input_view_noise,
                             c_views_dist.size(), device=c_views_dist.device)
        return c_views_azim, c_views_elev, c_views_dist

class GcnViewSelector(nn.Module):
    def __init__(self, nb_views=12):
        super().__init__()
        self.nb_views = nb_views
        if self.nb_views == 20:
            phi = (1 + np.sqrt(5)) / 2
            vertices = [[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                        [0, 1 / phi, phi], [0, 1 / phi, -phi], [0, -1 / phi, phi], [0, -1 / phi, -phi],
                        [phi, 0, 1 / phi], [phi, 0, -1 / phi], [-phi, 0, 1 / phi], [-phi, 0, -1 / phi],
                        [1 / phi, phi, 0], [-1 / phi, phi, 0], [1 / phi, -phi, 0], [-1 / phi, -phi, 0]]
        elif self.nb_views == 12:
            phi = np.sqrt(3)
            vertices = [[1, 0, phi / 3], [phi / 2, -1 / 2, phi / 3], [1 / 2, -phi / 2, phi / 3],
                        [0, -1, phi / 3], [-1 / 2, -phi / 2, phi / 3], [-phi / 2, -1 / 2, phi / 3],
                        [-1, 0, phi / 3], [-phi / 2, 1 / 2, phi / 3], [-1 / 2, phi / 2, phi / 3],
                        [0, 1, phi / 3], [1 / 2, phi / 2, phi / 3], [phi / 2, 1 / 2, phi / 3]]
        elif self.nb_views == 8:
            phi = np.sqrt(3)
            vertices = [[phi, phi, phi], [phi, phi, -phi], [phi, -phi , phi], [phi, -phi , -phi],
                        [-phi, phi, phi], [-phi, phi, -phi], [-phi, -phi, phi], [-phi, -phi , -phi]]
        elif self.nb_views == 4:
            phi = np.sqrt(3)
            vertices = [[phi, phi, phi], [phi, phi, -phi],
                        [-phi, phi, phi], [-phi, phi, -phi]]
        elif self.nb_views == 1:
            phi = np.sqrt(3)
            vertices = [[phi, 0, 0]]
        self.vertices = torch.tensor(vertices,dtype=torch.float32)

    def forward(self, shape_features=None, c_batch_size=1):
        vertices = self.vertices.unsqueeze(0).repeat(c_batch_size,1,1)
        return torch.flatten(vertices,start_dim=0,end_dim=1), None, None


class RandomViewSelector(nn.Module):
    def __init__(self, nb_views=12, canonical_distance=2.2,  transform_distance=False):
        super().__init__()
        self.nb_views = nb_views
        self.transform_distance = transform_distance
        self.canonical_distance = canonical_distance
        views_dist = torch.ones(
            (self.nb_views), dtype=torch.float, requires_grad=False) * canonical_distance
        views_elev = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        views_azim = torch.zeros(
            (self.nb_views), dtype=torch.float, requires_grad=False)
        self.register_buffer('views_azim', views_azim)
        self.register_buffer('views_elev', views_elev)
        self.register_buffer('views_dist', views_dist)

    def forward(self, shape_features=None, c_batch_size=1):
        c_views_azim = self.views_azim.expand(c_batch_size, self.nb_views)
        c_views_elev = self.views_elev.expand(c_batch_size, self.nb_views)
        c_views_dist = self.views_dist.expand(c_batch_size, self.nb_views)
        c_views_azim = c_views_azim + \
            torch.rand((c_batch_size, self.nb_views),
                       device=c_views_azim.device) * 360.0 - 180.0
        c_views_elev = c_views_elev + \
            torch.rand((c_batch_size, self.nb_views),
                       device=c_views_elev.device) * 180.0 - 90.0
        c_views_dist = c_views_dist + float(self.transform_distance) * 1.0 * c_views_dist * (
            torch.rand((c_batch_size, self.nb_views), device=c_views_dist.device) - 0.499)
        return c_views_azim, c_views_elev, c_views_dist



class ViewSelector(nn.Module):
    def __init__(self, nb_views=12, views_config="circular", canonical_elevation=30.0, canonical_distance=2.2, transform_distance=False, input_view_noise=0.0,):
        super().__init__()
        self.views_config = views_config
        self.nb_views = nb_views
        if self.views_config == "circular" or self.views_config == "custom" or (self.views_config == "spherical" and self.nb_views == 4):
            self.chosen_view_selector = CircularViewSelector(nb_views=nb_views, canonical_elevation=canonical_elevation,
                                                             canonical_distance=canonical_distance, transform_distance=transform_distance, input_view_noise=input_view_noise)
        elif self.views_config == "spherical":
            self.chosen_view_selector = SphericalViewSelector(nb_views=nb_views,canonical_distance=canonical_distance,transform_distance=transform_distance, input_view_noise=input_view_noise)
        elif self.views_config == "random":
            self.chosen_view_selector = RandomViewSelector(nb_views=nb_views, canonical_distance=canonical_distance, transform_distance=transform_distance)
        elif self.views_config == "gcn":
            self.chosen_view_selector = GcnViewSelector(nb_views=nb_views)

    def forward(self, shape_features=None, c_batch_size=1):
        return self.chosen_view_selector(shape_features=shape_features, c_batch_size=c_batch_size)



class FeatureExtractor(nn.Module):
    def __init__(self,  shape_features_size, views_config, shape_extractor, screatch_feature_extractor=False):
        super().__init__()
        self.shape_features_size = shape_features_size
        # self.features_type = features_type
        if views_config == "circular" or views_config == "random" or views_config == "spherical" or views_config == "custom":
            self.features_origin = "zeros"
        # elif setup["return_extracted_features"]:
        #     self.features_origin = "pre_extracted"


    def forward(self, extra_info=None, c_batch_size=1):
        if self.features_origin == "zeros":
            return torch.zeros((c_batch_size, self.shape_features_size))


class MVTN(nn.Module):
    """
    The MVTN main class that includes two components. one that extracts features from the object and one that predicts the views and other rendering setup. It is trained jointly with the main multi-view network.
    Args: 
        `nb_views` int , The number of views used in the multi-view setup
        `views_config`: str , The type of view selection method used. Choices: ["circular", "random", "learned_circular", "learned_direct", "spherical", "learned_spherical", "learned_random", "learned_transfer", "custom"]  
        `canonical_elevation`: float , the standard elevation of the camera viewpoints (if `views_config` == circulart).
        `canonical_distance`: float , the standard distance to the object of the camera viewpoints.
        `transform_distance`: bool , flag to allow for distance transformations from 0.5 `canonical_distance` to 1.5 `canonical_distance`
        `input_view_noise` : bool , flag to allow for adding noise to the camera viewpoints positions
        `shape_extractor` : str , The type of network used to extract features necessary for MVTN. Choices: ["PointNet", "DGCNN",]
        `shape_features_size`: float , the features size extracted used in MVTN. It depends on the `shape_extractor` used 
        `screatch_feature_extractor` : bool , flag to not use pretrained weights for the `shape_extractor`. default is to use the pretrinaed weights on ModelNet40
    Returns:
        an MVTN object that can render multiple views according to predefined setup
    """

    def __init__(self, nb_views=12, views_config="circular", canonical_elevation=30.0, canonical_distance=2.2):
        super().__init__()
        self.views_config = views_config
        self.view_selector = ViewSelector(nb_views=nb_views, views_config=views_config, canonical_elevation=canonical_elevation, canonical_distance=canonical_distance)


    def forward(self, points=None, c_batch_size=1):
        return self.view_selector(shape_features=None, c_batch_size=c_batch_size)


    def load_mvtn(self,weights_file):
        # Load checkpoint.
        print('\n==> Loading checkpoint..')
        assert os.path.isfile(weights_file
                            ), 'Error: no checkpoint file found!'
        checkpoint = torch.load(weights_file)
        self.load_state_dict(checkpoint['mvtn'])
