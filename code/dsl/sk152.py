import torch
from .library_functions import AffineFeatureSelectionFunction

import pdb

DEFAULT_BBALL_FEATURE_SUBSETS = {
    "arms"      : torch.LongTensor([2, 3, 4, 5, 6, 7]),
    "legs"   : torch.LongTensor([8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24]),
    "faces"   : torch.LongTensor([0, 1, 15, 16, 17, 18])
}
SK_FULL_FEATURE_DIM = 75


class ArmSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        # full
        self.full_feature_dim = SK_FULL_FEATURE_DIM
        # selected feature
        feature_ids = DEFAULT_BBALL_FEATURE_SUBSETS["arms"]
        feature_tensor = torch.cat([torch.arange(f_i*3, (f_i+1)*3, dtype=torch.long) for f_i in feature_ids])
        self.feature_tensor = feature_tensor
        super().__init__(input_size, output_size, num_units, name="ArmsXYZAffine")

class LegSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        # full
        self.full_feature_dim = SK_FULL_FEATURE_DIM
        # selected feature
        feature_ids = DEFAULT_BBALL_FEATURE_SUBSETS["legs"]
        feature_tensor = torch.cat([torch.arange(f_i*3, (f_i+1)*3, dtype=torch.long) for f_i in feature_ids])
        self.feature_tensor = feature_tensor
        super().__init__(input_size, output_size, num_units, name="LegsXYZAffine")

class FaceSelection(AffineFeatureSelectionFunction):

    def __init__(self, input_size, output_size, num_units):
        # full
        self.full_feature_dim = SK_FULL_FEATURE_DIM
        # selected feature
        feature_ids = DEFAULT_BBALL_FEATURE_SUBSETS["faces"]
        feature_tensor = torch.cat([torch.arange(f_i*3, (f_i+1)*3, dtype=torch.long) for f_i in feature_ids])
        self.feature_tensor = feature_tensor
        super().__init__(input_size, output_size, num_units, name="FaceXYZAffine")
