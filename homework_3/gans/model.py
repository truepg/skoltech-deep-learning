
import torch
from torch import nn
from torch.nn import functional as F
import functools
import math
from torch.nn.utils import spectral_norm



class AdaptiveBatchNorm(nn.BatchNorm2d):
    """
    Adaptive batch normalization layer (4 points)

    Args:
        num_features: number of features in batch normalization layer
        embed_features: number of features in embeddings

    The base layer (BatchNorm2d) is applied to "inputs" with affine = False

    After that, the "embeds" are linearly mapped to "gamma" and "bias"
    
    These "gamma" and "bias" are applied to the outputs like in batch normalization
    with affine = True (see definition of batch normalization for reference)
    """
    def __init__(self, num_features: int, embed_features: int):
        super(AdaptiveBatchNorm, self).__init__(num_features, affine=False)
        # TODO
        self.gamma_linear_map = spectral_norm(nn.Linear(embed_features, num_features))
        self.bias_linear_map = spectral_norm(nn.Linear(embed_features, num_features))
        
    def forward(self, inputs, embeds):
        gamma = self.gamma_linear_map(embeds) # TODO 
        bias = self.bias_linear_map(embeds) # TODO

        assert gamma.shape[0] == inputs.shape[0] and gamma.shape[1] == inputs.shape[1]
        assert bias.shape[0] == inputs.shape[0] and bias.shape[1] == inputs.shape[1]

        outputs = super().forward(inputs) # TODO: apply batchnorm

        return outputs * gamma[..., None, None] + bias[..., None, None]


class PreActResBlock(nn.Module):
    """
    Pre-activation residual block (6 points)

    Paper: https://arxiv.org/pdf/1603.05027.pdf
    Scheme: materials/preactresblock.png
    Review: https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e

    Args:
        in_channels: input number of channels
        out_channels: output number of channels
        batchnorm: this block is with/without adaptive batch normalization
        upsample: use nearest neighbours upsampling at the beginning
        downsample: use average pooling after the end

    in_channels != out_channels:
        - first conv: in_channels -> out_channels
        - second conv: out_channels -> out_channels
        - use 1x1 conv in skip connection

    in_channels == out_channels: skip connection is without a conv
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int = None,
                 batchnorm: bool = False,
                 upsample: bool = False,
                 downsample: bool = False):
        super(PreActResBlock, self).__init__()

        # TODO: define pre-activation residual block
        # TODO: apply spectral normalization to conv layers

        self.skip_connection_flag = False
        self.upsample = upsample
        self.downsample = downsample
        self.batchnorm = batchnorm

        if in_channels != out_channels:
          self.skip_connection_flag = True
          self.skip_connection = nn.Conv2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=1)          

        if batchnorm:
          # self.block = nn.Sequential(
          #     AdaptiveBatchNorm(in_channels, embed_channels),
          #     nn.ReLU(),
          #     spectral_norm(nn.Conv2d(in_channels=in_channels, 
          #                             out_channels=out_channels, 
          #                             kernel_size=3, 
          #                             padding=1)),
          #     AdaptiveBatchNorm(in_channels, embed_channels),
          #     nn.ReLU(),
          #     spectral_norm(nn.Conv2d(in_channels=out_channels, 
          #                             out_channels=out_channels, 
          #                             kernel_size=3, 
          #                             padding=1))
          # )
          self.abn1 = AdaptiveBatchNorm(in_channels, embed_channels)
          self.block1 = nn.Sequential(
              nn.ReLU(),
              spectral_norm(nn.Conv2d(in_channels=in_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=3, 
                                      padding=1)))
          self.abn2 = AdaptiveBatchNorm(out_channels, embed_channels)
          self.block2 = nn.Sequential(
              nn.ReLU(),
              spectral_norm(nn.Conv2d(in_channels=out_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=3, 
                                      padding=1))
          )
        else:
          self.block = nn.Sequential(
              nn.ReLU(),
              spectral_norm(nn.Conv2d(in_channels=in_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=3, 
                                      padding=1)),
              nn.ReLU(),
              spectral_norm(nn.Conv2d(in_channels=out_channels, 
                                      out_channels=out_channels, 
                                      kernel_size=3, 
                                      padding=1))
          )

        
        # Don't forget that activation after residual sum cannot be inplace!

    def forward(self, 
                inputs, # regular features 
                embeds=None): # embeds used in adaptive batch norm
        # TODO

        if self.upsample:
          inputs = F.interpolate(inputs, scale_factor=2)

        if self.skip_connection_flag:
          skip_connection_output = self.skip_connection(inputs)
        else:
          skip_connection_output = inputs.clone()

        if self.batchnorm:
          outputs = self.abn1(inputs, embeds)
          outputs = self.block1(outputs)

          outputs = self.abn2(outputs, embeds)
          outputs = self.block2(outputs)
        else:
          outputs = self.block(inputs)

        outputs += skip_connection_output

        if self.downsample:
          outputs = nn.AvgPool2d(kernel_size=2)(outputs)

        return outputs


class Generator(nn.Module):
    """
    Generator network (8 points)
    
    TODO:

      - Implement an option to condition the synthesis on trainable class embeddings
        (use nn.Embedding module with noise_channels as the size of each embed)

      - Concatenate input noise with class embeddings (if use_class_condition = True) to obtain input embeddings

      - Linearly map input embeddings into input tensor with the following dims: max_channels x 4 x 4

      - Forward an input tensor through a convolutional part, 
        which consists of num_blocks PreActResBlocks and performs upsampling by a factor of 2 in each block

      - Each PreActResBlock is additionally conditioned on the input embeddings (via adaptive batch normalization)

      - At the end of the convolutional part apply regular BN, ReLU and Conv as an image prediction head

      - Apply spectral norm to all conv and linear layers (not the embedding layer)

      - Use Sigmoid at the end to map the outputs into an image

    Notes:

      - The last convolutional layer should map min_channels to 3. With each upsampling you should decrease
        the number of channels by a factor of 2

      - Class embeddings are only used and trained if use_class_condition = True
    """    
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 noise_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_class_condition: bool):
        super(Generator, self).__init__()
        self.output_size = 4 * 2**num_blocks
        # TODO
        self.max_channels = max_channels
        self.use_class_condition = use_class_condition

        if use_class_condition:
          # CHECK EMBEDDING DIMENSION
          self.embedding = nn.Embedding(num_embeddings=num_classes, 
                                        embedding_dim=noise_channels)
          self.linearMap = nn.Linear(2 * noise_channels, max_channels * 4 ** 2)
        else:
          self.linearMap = nn.Linear(noise_channels, max_channels * 4 ** 2)

        preActResBlocksInChannels = [int(max_channels / 2 ** i) for i in range(num_blocks)]
        preActResBlocksOutChannels = [int(in_channels / 2) for in_channels in preActResBlocksInChannels]

        # print('InChannels: ', preActResBlocksInChannels)
        # print('OutChannels: ', preActResBlocksOutChannels)
        
        if use_class_condition:
          # self.preActResBlocks = nn.Sequential(
          #     *[PreActResBlock(in_channels=preActResBlocksInChannels[i],
          #                      out_channels=preActResBlocksOutChannels[i],
          #                      batchnorm=True,
          #                      embed_channels=noise_channels,
          #                      upsample=True) for i in range(num_blocks)]
          # )
          self.preActResBlocks = [PreActResBlock(in_channels=preActResBlocksInChannels[i],
                                                 out_channels=preActResBlocksOutChannels[i],
                                                 batchnorm=True,
                                                 embed_channels=noise_channels,
                                                 upsample=True) for i in range(num_blocks)]
          self.preActResBlocks = nn.ModuleList(self.preActResBlocks)
        else:
          self.preActResBlocks = nn.Sequential(
              *[PreActResBlock(in_channels=preActResBlocksInChannels[i],
                               out_channels=preActResBlocksOutChannels[i],
                               upsample=True) for i in range(num_blocks)]
          )

        self.predictionHead = nn.Sequential(
            nn.BatchNorm2d(preActResBlocksOutChannels[-1]),
            nn.ReLU(),
            spectral_norm(nn.Conv2d(in_channels=preActResBlocksOutChannels[-1],
                                    out_channels=3,
                                    kernel_size=3,
                                    padding=1))
        )

        self.mapOutputsIntoImage = nn.Sigmoid()

    def forward(self, noise, labels):
        # TODO
        batch_size = noise.shape[0]

        if self.use_class_condition:
          embs = self.embedding(labels)
          # print(embs.shape)
          noise = torch.cat([noise, embs], dim=1)
        # print(noise.shape)
        outputs = self.linearMap(noise)
        outputs = outputs.view(batch_size, self.max_channels, 4, 4)

        if self.use_class_condition:
          # outputs = self.preActResBlocks(outputs, embs)
          for layer in self.preActResBlocks:
            outputs = layer(outputs, embs)
        else:
          outputs = self.preActResBlocks(outputs)

        outputs = self.predictionHead(outputs)
        outputs = self.mapOutputsIntoImage(outputs)

        assert outputs.shape == (noise.shape[0], 3, self.output_size, self.output_size)
        return outputs


class Discriminator(nn.Module):
    """
    Discriminator network (8 points)

    TODO:
    
      - Define a convolutional part of the discriminator similarly to
        the generator blocks, but in the inverse order, with downsampling, and
        without batch normalization
    
      - At the end of the convolutional part apply ReLU and sum pooling
    
    TODO: implement projection discriminator head (https://arxiv.org/abs/1802.05637)
    
    Scheme: materials/prgan.png
    
    Notation:
    
      - phi is a convolutional part of the discriminator
    
      - psi is a vector
    
      - y is a class embedding
    
    Class embeddings matrix is similar to the generator, shape: num_classes x max_channels

    Discriminator outputs a B x 1 matrix of realism scores

    Apply spectral norm for all layers (conv, linear, embedding)
    """
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_projection_head: bool):
        super(Discriminator, self).__init__()

        # TODO
        self.use_projection_head = use_projection_head

        if use_projection_head:
          self.embedding = spectral_norm(nn.Embedding(num_embeddings=num_classes, 
                                                      embedding_dim=max_channels))
      
        preActResBlocksOutChannels = [int(max_channels / 2 ** i) for i in range(num_blocks)][::-1]
        preActResBlocksInChannels = [int(out_channels / 2) for out_channels in preActResBlocksOutChannels]

        # print('InChannels: ', preActResBlocksInChannels)
        # print('OutChannels: ', preActResBlocksOutChannels)
        
        self.preActResBlocks = nn.Sequential(
            *[PreActResBlock(in_channels=preActResBlocksInChannels[i],
                            out_channels=preActResBlocksOutChannels[i],
                            downsample=True) for i in range(num_blocks)]
        )

        self.inversePredictionHead = nn.Sequential(
              nn.ReLU(),
              spectral_norm(nn.Conv2d(in_channels=3,
                                      out_channels=preActResBlocksInChannels[0],
                                      kernel_size=3,
                                      padding=1))
          )
        
        self.actAfterConvPart = nn.ReLU()

        self.linearToSingleDigit = spectral_norm(nn.Linear(preActResBlocksOutChannels[-1], 1))



    def forward(self, inputs, labels):
        # TODO
        outputs = self.inversePredictionHead(inputs)
        outputs = self.preActResBlocks(outputs)

        outputs = self.actAfterConvPart(outputs)
        outputs = torch.sum(outputs.view(*outputs.shape[:2], 
                                         outputs.shape[2] * outputs.shape[3]), dim=2)

        if self.use_projection_head:
          embs = self.embedding(labels)
          emb_outputs = (outputs * embs).sum(dim=1)

          outputs = self.linearToSingleDigit(outputs).view(-1)

          scores = outputs + emb_outputs
        else:
          scores = self.linearToSingleDigit(outputs).view(-1)

        assert scores.shape == (inputs.shape[0],)
        return scores