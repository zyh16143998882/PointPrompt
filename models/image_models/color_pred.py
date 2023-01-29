from torch import nn

from models.dgcnn import DGCNNColor, DGCNNPrompt
from models.image_models.pointnet import PointNetColor, PointNet
from .blocks import *


class ColorPredict(nn.Module):
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

    def __init__(self, shape_extractor="pointnet", use_avgpool=False,output_channels=40):
        super().__init__()
        self.shape_extractor = shape_extractor

        if shape_extractor == "pointnet":
            self.cp = PointNetColor(alignment=True)
        elif shape_extractor == "dgcnn":
            class Args:
                def __init__(self):
                    self.k = 20
                    self.emb_dims = 1024
                    self.dropout = 0.5
                    self.leaky_relu = 1
                    self.use_avgpool = use_avgpool

            args = Args()
            self.cp = DGCNNColor(args, output_channels)


    def forward(self, points=None):
        points_color = self.cp(points)
        return points_color[0].permute(0,2,1)


class PromptPredict(nn.Module):
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

    def __init__(self, shape_extractor="pointnet", use_avgpool=False,output_channels=40, nb_views=None):
        super().__init__()
        self.shape_extractor = shape_extractor
        self.nb_views = nb_views
        if shape_extractor == "pointnet":
            self.cp = PointNet(output_channels, alignment=True)
        elif shape_extractor == "dgcnn":
            class Args:
                def __init__(self):
                    self.k = 20
                    self.emb_dims = 1024
                    self.dropout = 0.5
                    self.leaky_relu = 1
                    self.use_avgpool = use_avgpool

            args = Args()
            self.cp = DGCNNPrompt(args, output_channels)

        self.view_transformer = nn.Sequential(MLP([output_channels, output_channels, output_channels, 5 *self.nb_views, 3 * self.nb_views], dropout=0.5, norm=True),
                                    MLP([3 * self.nb_views, 3 * self.nb_views], act=None, dropout=0, norm=False),
                                    nn.Tanh())


    def forward(self, points=None):
        points_color = self.cp(points)
        adjutment_vector = self.view_transformer(points_color[0])
        adjutment_vector = torch.chunk(adjutment_vector, 3, dim=1)
        adjutment_vector = torch.cat([adjutment_vector[0].unsqueeze(2),adjutment_vector[1].unsqueeze(2),adjutment_vector[2].unsqueeze(2)],dim=2)
        return adjutment_vector