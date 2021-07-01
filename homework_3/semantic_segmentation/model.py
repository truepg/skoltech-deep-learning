
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import numpy as np

class FirstDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2):
        super(FirstDownBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1, 3, padding=1),
            nn.BatchNorm2d(out_channels1),
            nn.ReLU(),
            nn.Conv2d(out_channels1, out_channels2, 3, padding=1),
            nn.BatchNorm2d(out_channels2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class RegularDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RegularDownBlock, self).__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels):
        super(BottleNeckBlock, self).__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, 
                               output_padding=1)
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, out_channels, 3, stride=2, padding=1, 
                    output_padding=1)
        )

    def forward(self, x):
        return self.block(x)


class LastUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, out_channels3):
        super(LastUpBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(in_channels, out_channels1, 3, padding=1),
            nn.BatchNorm2d(out_channels1),
            nn.ReLU(),
            nn.Conv2d(out_channels1, out_channels2, 3, padding=1),
            nn.BatchNorm2d(out_channels2),
            nn.ReLU(),
            nn.Conv2d(out_channels2, out_channels3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)


def getDownBlockOutChannels(num_down_blocks, max_channels):
    arr = np.array([2 ** i for i in range(num_down_blocks)])
    return np.array(max_channels / arr, dtype=int)[::-1]

class UNet(nn.Module):
    """
    TODO: 8 points

    A standard UNet network (with padding in covs).

    For reference, see the scheme in materials/unet.png
    - Use batch norm between conv and relu
    - Use max pooling for downsampling
    - Use conv transpose with kernel size = 3, stride = 2, padding = 1, and output padding = 1 for upsampling
    - Use 0.5 dropout after concat

    Args:
      - num_classes: number of output classes
      - min_channels: minimum number of channels in conv layers
      - max_channels: number of channels in the bottleneck block
      - num_down_blocks: number of blocks which end with downsampling

    The full architecture includes downsampling blocks, a bottleneck block and upsampling blocks

    You also need to account for inputs which size does not divide 2**num_down_blocks:
    interpolate them before feeding into the blocks to the nearest size which divides 2**num_down_blocks,
    and interpolate output logits back to the original shape
    """
    def __init__(self, 
                 num_classes,
                 min_channels=32,
                 max_channels=512, 
                 num_down_blocks=4):
        super(UNet, self).__init__()
        self.num_classes = num_classes

        # TODO
        downBlockOutChannelsList = getDownBlockOutChannels(num_down_blocks, max_channels)
        self.downBlocksOutputsCopy = [None] * num_down_blocks

        self.downBlocks = [FirstDownBlock(in_channels=3, out_channels1=min_channels, 
                                          out_channels2=downBlockOutChannelsList[0])]

        for i in range(1, num_down_blocks):
          self.downBlocks.append(RegularDownBlock(in_channels=downBlockOutChannelsList[i-1], 
                                                  out_channels=downBlockOutChannelsList[i]))

        self.bottleNeckBlock = BottleNeckBlock(in_channels=max_channels)
        
        upBlockInChannelsList = downBlockOutChannelsList[::-1] * 2

        self.upBlocks = []

        for i in range(num_down_blocks-1):
          self.upBlocks.append(UpBlock(in_channels=upBlockInChannelsList[i], 
                                       out_channels=upBlockInChannelsList[i] // 4))
          
        self.upBlocks.append(LastUpBlock(in_channels=upBlockInChannelsList[-1], 
                                         out_channels1=upBlockInChannelsList[-1] // 2, 
                                         out_channels2=min_channels, 
                                         out_channels3=num_classes))
        
        self.upBlocks = nn.ModuleList(self.upBlocks)
        self.downBlocks = nn.ModuleList(self.downBlocks)
        
        
    def forward(self, inputs):
        # TODO
        batch_size = inputs.shape[2:]
        inputs_ = F.interpolate(input=inputs, size=(256, 256))

        for block_ix, block in enumerate(self.downBlocks):
          inputs_ = block(inputs_)
          self.downBlocksOutputsCopy[block_ix] = inputs_.clone()

        inputs_ = self.bottleNeckBlock(inputs_)

        for block_ix, block in enumerate(self.upBlocks):
          inputs_ = torch.cat([self.downBlocksOutputsCopy[-(block_ix+1)], inputs_], dim=1)
          inputs_ = block(inputs_)
        
        logits = F.interpolate(input=inputs_, size=batch_size)

        assert logits.shape == (inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits

class DeepLab(nn.Module):
    """
    TODO: 6 points

    (simplified) DeepLab segmentation network.
    
    Args:
      - backbone: ['resnet18', 'vgg11_bn', 'mobilenet_v3_small'],
      - aspp: use aspp module
      - num classes: num output classes

    During forward pass:
      - Pass inputs through the backbone to obtain features
      - Apply ASPP (if needed)
      - Apply head
      - Upsample logits back to the shape of the inputs
    """
    def __init__(self, backbone, aspp, num_classes):
        super(DeepLab, self).__init__()
        self.backbone = backbone
        self.init_backbone()
        self.num_classes = num_classes
        self.aspp = aspp

        if aspp:
            self.aspp = ASPP(self.out_features, 256, [12, 24, 36])

        self.head = DeepLabHead(self.out_features, num_classes)

    def init_backbone(self):
        # TODO: initialize an ImageNet-pretrained backbone
        if self.backbone == 'resnet18':
            # TODO: number of output features in the backbone
            self.model = models.resnet18(pretrained=True)
            self.out_features = self.model.layer4[1].bn2.num_features
        elif self.backbone == 'vgg11_bn':
            # TODO
            self.model = models.vgg11_bn(pretrained=True)
            self.out_features = self.model.features[26].num_features

        elif self.backbone == 'mobilenet_v3_small':
            # TODO
            self.model = models.mobilenet_v3_small(pretrained=True)
            self.out_features = self.model.features[12][1].num_features

    def _forward(self, x):
        # TODO: forward pass through the backbone
        if self.backbone == 'resnet18':
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

        elif self.backbone == 'vgg11_bn':
            x = self.model.features(x)

        elif self.backbone == 'mobilenet_v3_small':
            x = self.model.features(x)

        return x

    def forward(self, inputs):
        # TODO

        inputs_ = self._forward(inputs)

        if not self.aspp:
          logits = self.head(inputs_)
        else:
          inputs_ = self.aspp(inputs_)
          logits = self.head(inputs_)

        # logits = F.interpolate(input=logits, size=tuple(inputs.shape[2:]))
        logits = F.interpolate(input=logits, size=inputs.shape[2:])

        assert logits.shape == (inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, 1)
        )


class ASPP(nn.Module):
    """
    TODO: 8 points

    Atrous Spatial Pyramid Pooling module
    with given atrous_rates and out_channels for each head
    Description: https://paperswithcode.com/method/aspp
    
    Detailed scheme: materials/deeplabv3.png
      - "Rates" are defined by atrous_rates
      - "Conv" denotes a Conv-BN-ReLU block
      - "Image pooling" denotes a global average pooling, followed by a 1x1 "conv" block and bilinear upsampling
      - The last layer of ASPP block should be Dropout with p = 0.5

    Args:
      - in_channels: number of input and output channels
      - num_channels: number of output channels in each intermediate "conv" block
      - atrous_rates: a list with dilation values
    """
    def __init__(self, in_channels, num_channels, atrous_rates):
        super(ASPP, self).__init__()
        
        self.convs = [nn.Sequential(
              nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=1),
              nn.BatchNorm2d(num_channels),
              nn.ReLU()
          )]

        for rate in atrous_rates:
          self.convs.append(nn.Sequential(
              nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, dilation=rate, padding=rate),
              nn.BatchNorm2d(num_channels),
              nn.ReLU()
          ))
        self.convs = nn.ModuleList(self.convs)
        
        self.image_pooling = nn.Sequential(
              nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=1),
              # nn.BatchNorm2d(num_channels),
              nn.ReLU()
          )
        
        N = len(atrous_rates) + 2
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=N * num_channels, out_channels=in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout()
        )


    def forward(self, x):
        # TODO: forward pass through the ASPP module
        res_list = []
        # print('x shape: ', x.shape)
        for block_ix, block in enumerate(self.convs):
          # print(block_ix)
          res_list.append(block(x))
          # print(res_list[-1].shape)

        global_pooling = nn.AvgPool2d(kernel_size=x.shape[2:])(x)
        image_pooling = self.image_pooling(global_pooling)
        image_pooling = F.interpolate(image_pooling, size=x.shape[2:])
        res_list.append(image_pooling)

        res = torch.cat(res_list, dim=1)
        res = self.last_layer(res)
        
        assert res.shape[1] == x.shape[1], 'Wrong number of output channels'
        assert res.shape[2] == x.shape[2] and res.shape[3] == x.shape[3], 'Wrong spatial size'
        return res