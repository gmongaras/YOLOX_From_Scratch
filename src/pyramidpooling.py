import math
import torch
import torch.nn as nn
import torch.nn.functional as F



#### Note: Code from https://github.com/revidee/pytorch-pyramid-pooling




class PyramidPooling(nn.Module):
    def __init__(self, levels, mode="max"):
        """
        General Pyramid Pooling class which uses Spatial Pyramid Pooling by default and holds the static methods for both spatial and temporal pooling.
        :param levels defines the different divisions to be made in the width and (spatial) height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n], where  n: sum(filter_amount*level*level) for each level in levels (spatial) or
                                                                    n: sum(filter_amount*level) for each level in levels (temporal)
                                            which is the concentration of multi-level pooling
        """
        super(PyramidPooling, self).__init__()
        self.levels = levels
        self.mode = mode

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out

    @staticmethod
    def spatial_pyramid_pool(previous_conv, levels, mode):
        """
        Static Spatial Pyramid Pooling method, which divides the input Tensor vertically and horizontally
        (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
        :param previous_conv input tensor of the previous convolutional layer
        :param levels defines the different divisions to be made in the width and height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
        for i in range(len(levels)):
            h_kernel = int(math.ceil(previous_conv_size[0] / levels[i]))
            w_kernel = int(math.ceil(previous_conv_size[1] / levels[i]))
            w_pad1 = int(math.floor((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(math.ceil((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            h_pad1 = int(math.floor((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            h_pad2 = int(math.ceil((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * levels[i] - previous_conv_size[1]) and \
                   h_pad1 + h_pad2 == (h_kernel * levels[i] - previous_conv_size[0])

            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                                 mode='constant', value=0)
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)
            if i == 0:
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)

        return spp

    @staticmethod
    def temporal_pyramid_pool(previous_conv, out_pool_size, mode):
        """
        Static Temporal Pyramid Pooling method, which divides the input Tensor horizontally (last dimensions)
        according to each level in the given levels and pools its value according to the given mode.
        In other words: It divides the Input Tensor in "level" horizontal stripes with width of roughly (previous_conv.size(3) / level)
        and the original height and pools the values inside this stripe
        :param previous_conv input tensor of the previous convolutional layer
        :param levels defines the different divisions to be made in the width dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
        for i in range(len(out_pool_size)):
            # print(previous_conv_size)
            #
            h_kernel = previous_conv_size[0]
            w_kernel = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            w_pad1 = int(math.floor((w_kernel * out_pool_size[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(math.ceil((w_kernel * out_pool_size[i] - previous_conv_size[1]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * out_pool_size[i] - previous_conv_size[1])

            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2],
                                 mode='constant', value=0)
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)
            if i == 0:
                tpp = x.view(num_sample, -1)
            else:
                tpp = torch.cat((tpp, x.view(num_sample, -1)), 1)

        return tpp


class SpatialPyramidPooling(PyramidPooling):
    def __init__(self, levels, mode="max"):
        """
                Spatial Pyramid Pooling Module, which divides the input Tensor horizontally and horizontally
                (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
                Can be used as every other pytorch Module and has no learnable parameters since it's a static pooling.
                In other words: It divides the Input Tensor in level*level rectangles width of roughly (previous_conv.size(3) / level)
                and height of roughly (previous_conv.size(2) / level) and pools its value. (pads input to fit)
                :param levels defines the different divisions to be made in the width dimension
                :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
                :returns (forward) a tensor vector with shape [batch x 1 x n],
                                                    where n: sum(filter_amount*level*level) for each level in levels
                                                    which is the concentration of multi-level pooling
                """
        super(SpatialPyramidPooling, self).__init__(levels, mode=mode)

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        """
                Calculates the output shape given a filter_amount: sum(filter_amount*level*level) for each level in levels
                Can be used to x.view(-1, spp.get_output_size(filter_amount)) for the fully-connected layers
                :param filters: the amount of filter of output fed into the spatial pyramid pooling
                :return: sum(filter_amount*level*level)
        """
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out


class TemporalPyramidPooling(PyramidPooling):
    def __init__(self, levels, mode="max"):
        """
        Temporal Pyramid Pooling Module, which divides the input Tensor horizontally (last dimensions)
        according to each level in the given levels and pools its value according to the given mode.
        Can be used as every other pytorch Module and has no learnable parameters since it's a static pooling.
        In other words: It divides the Input Tensor in "level" horizontal stripes with width of roughly (previous_conv.size(3) / level)
        and the original height and pools the values inside this stripe
        :param levels defines the different divisions to be made in the width dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns (forward) a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        super(TemporalPyramidPooling, self).__init__(levels, mode=mode)

    def forward(self, x):
        return self.temporal_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        """
        Calculates the output shape given a filter_amount: sum(filter_amount*level) for each level in levels
        Can be used to x.view(-1, tpp.get_output_size(filter_amount)) for the fully-connected layers
        :param filters: the amount of filter of output fed into the temporal pyramid pooling
        :return: sum(filter_amount*level)
        """
        out = 0
        for level in self.levels:
            out += filters * level
        return out