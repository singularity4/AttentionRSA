"""models.py - the U-Nets UNet-Starlight and UNet-Radius-mini, with and without attention, and the classifier and the
radius regressor built from their encoder

AttentionRSA.

Task of the U-Nets
    Segmentation of lensed arcs: for every pixel of a 101 x 101 stamp the network gives the probability that the
    pixel contains light of a lensed source. Two further outputs are read from the bottleneck: the Einstein radius
    (UNet-Radius-mini) and a lens output with two values, non-lens and lens (UNet-Starlight with lens_output=True).
    The networks are trained on simulated images (train_copies.py), where the arc mask, the label and the Einstein
    radius of every image are known.

Architecture (the same layers for all models)
    Encoder: five blocks. inc is a convolutional block; d1-d4 each consist of a 2 x 2 max pooling layer and a
    convolutional block. Max pooling rounds down, so the feature maps have 101, 50, 25, 12 and 6 pixels on a side;
    the 6 x 6 feature maps of d4 form the bottleneck.
    Decoder: four blocks u1-u4. Each enlarges its input by bilinear interpolation to the size of the encoder feature
    maps of the same level, halves the number of feature maps with a 1 x 1 convolutional layer, joins them with the
    encoder feature maps (skip connection) and applies a convolutional block.
    Convolutional block: two 3 x 3 convolutional layers without bias, each followed by group normalisation and a ReLU
    activation. Spatial dropout with probability 0.1 follows the blocks d3, d4 and u1.
    Output: a 1 x 1 convolutional layer with one value (logit) per pixel; the sigmoid of the logit is the arc
    probability. The weights start from the default initialisation of PyTorch.

Readout of the bottleneck
    Without attention, the 16 b bottleneck feature maps are averaged over the 36 positions. With attention
    (AttentionPooling), a 1 x 1 convolutional layer gives each position a score, a softmax over the positions turns
    the scores into weights that are positive and sum to one (the 6 x 6 attention map), and the feature maps are
    averaged with these weights. A linear layer turns the averages into the Einstein radius (one output, arcseconds)
    or into the lens output (two outputs; the softmax of the two is the lens probability).
    The output layers are created after all other layers, and the attention layer last, so that for a given seed a
    network with attention starts from the same weights as the network without attention in every layer they share.

Models
    The models differ in the number of filters b of the first convolutional block, which doubles after every pooling
    layer up to 16 b in the bottleneck, and in their outputs.
        key           name               b    outputs                        parameters (with attention)
        starlight     UNet-Starlight    16    arc mask, lens (lens_output)   1,812,243  (1,812,500)
        radius-mini   UNet-Radius-mini   8    arc mask, Einstein radius        453,898    (454,027)
    The key is the option --model of train_copies.py. With attention the networks are called Attention
    UNet-Starlight and Attention UNet-Radius-mini. Without attention they are the U-Nets of UNet-Starlight
    (https://github.com/singularity4/UNet-Starlight), with the same layers and the same initialisation; here
    UNet-Starlight also has the lens output.

Classifier and radius regressor for the real stamps
    The real stamps have no arc masks, so only the encoder and the readout are trained on them:
        EncoderClassifier   encoder, readout and lens output (a new linear layer if the U-Net has no lens output)
        RadiusRegressor     encoder, readout and Einstein radius layer
    Both start either from the weights of a U-Net trained on simulations (pretrained) or from random weights.

Saved networks
    A saved network stores the constructor arguments of UNet ("architecture": in_channels, base, dropout and, if they
    are used, einstein_radius, lens_output and attention) and the weights, under the names of the network parts (inc,
    d1-d4, u1-u4, out, radius, lens, attention; within the blocks net, reduce, conv and score). These names must not
    change, or saved networks can no longer be loaded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

FILTERS = {"starlight": 16, "radius-mini": 8}            # filters of the first convolutional block
NAMES = {"starlight": "UNet-Starlight", "radius-mini": "UNet-Radius-mini"}
EINSTEIN_RADIUS_MODELS = ("radius-mini",)                 # models with the Einstein radius output
ENCODER_BLOCKS = ("inc", "d1", "d2", "d3", "d4")


# ================================================================================================ 1. network parts

def number_of_groups(feature_maps):
    """Number of groups of the group normalisation: the largest of 8, 4, 2 and 1 that divides the number of feature
    maps."""
    for groups in (8, 4, 2, 1):
        if feature_maps % groups == 0:
            return groups
    return 1


class ConvolutionalBlock(nn.Module):
    """Two 3 x 3 convolutional layers, each followed by group normalisation and a ReLU activation; optionally spatial
    dropout, which sets whole feature maps to zero."""

    def __init__(self, input_maps, output_maps, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(input_maps, output_maps, 3, padding=1, bias=False),
                  nn.GroupNorm(number_of_groups(output_maps), output_maps),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(output_maps, output_maps, 3, padding=1, bias=False),
                  nn.GroupNorm(number_of_groups(output_maps), output_maps),
                  nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features)


class EncoderBlock(nn.Module):
    """Encoder block d1-d4: a 2 x 2 max pooling layer, which halves the size of the feature maps (rounding down),
    followed by a convolutional block."""

    def __init__(self, input_maps, output_maps, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), ConvolutionalBlock(input_maps, output_maps, dropout))

    def forward(self, features):
        return self.net(features)


class DecoderBlock(nn.Module):
    """Decoder block u1-u4: bilinear interpolation to the size of the skip connection, a 1 x 1 convolutional layer
    that halves the number of feature maps, joining with the feature maps of the skip connection, and a convolutional
    block."""

    def __init__(self, input_maps, skip_maps, output_maps, dropout=0.0):
        super().__init__()
        self.reduce = nn.Conv2d(input_maps, output_maps, 1)
        self.conv = ConvolutionalBlock(output_maps + skip_maps, output_maps, dropout)

    def forward(self, features, skip_features):
        features = F.interpolate(features, size=skip_features.shape[-2:], mode="bilinear", align_corners=False)
        features = self.reduce(features)
        features = torch.cat([skip_features, features], dim=1)
        return self.conv(features)


class AttentionPooling(nn.Module):
    """The weighted average of the bottleneck feature maps. A 1 x 1 convolutional layer gives each position a score,
    and a softmax over the positions turns the scores into weights that are positive and sum to one (the attention
    map)."""

    def __init__(self, feature_maps):
        super().__init__()
        self.score = nn.Conv2d(feature_maps, 1, kernel_size=1)

    def forward(self, feature_maps, return_weights=False):
        scores = self.score(feature_maps).flatten(1)                       # (batch, positions)
        weights = torch.softmax(scores, dim=1)
        pooled = (feature_maps.flatten(2) * weights[:, None, :]).sum(dim=2)
        if return_weights:
            return pooled, weights.view(feature_maps.shape[0], *feature_maps.shape[2:])
        return pooled


def pool_bottleneck(bottleneck, attention=None, return_weights=False):
    """The pooled bottleneck: the plain average over the positions, or the attention-weighted average. With
    return_weights=True, also the weights of the positions (1/36 each for the plain average)."""
    if attention is not None:
        return attention(bottleneck, return_weights=return_weights)
    pooled = bottleneck.mean(dim=(2, 3))
    if return_weights:
        positions = bottleneck.shape[2] * bottleneck.shape[3]
        return pooled, torch.full((bottleneck.shape[0], *bottleneck.shape[2:]), 1.0 / positions,
                                  device=bottleneck.device)
    return pooled


# ===================================================================================================== 2. U-Net

class UNet(nn.Module):
    """U-Net with `base` filters in the first convolutional block. Output: one logit per pixel; with
    einstein_radius=True, also the estimated Einstein radius; with lens_output=True, also two lens logits (non-lens,
    lens); with attention=True, both are read from the bottleneck through an attention map."""

    def __init__(self, in_channels=1, base=16, dropout=0.10, einstein_radius=False, lens_output=False,
                 attention=False):
        super().__init__()
        self.bottleneck_maps = 16 * base
        self.inc = ConvolutionalBlock(in_channels, base)
        self.d1 = EncoderBlock(base, 2 * base)
        self.d2 = EncoderBlock(2 * base, 4 * base)
        self.d3 = EncoderBlock(4 * base, 8 * base, dropout=dropout)
        self.d4 = EncoderBlock(8 * base, 16 * base, dropout=dropout)
        self.u1 = DecoderBlock(16 * base, 8 * base, 8 * base, dropout=dropout)
        self.u2 = DecoderBlock(8 * base, 4 * base, 4 * base)
        self.u3 = DecoderBlock(4 * base, 2 * base, 2 * base)
        self.u4 = DecoderBlock(2 * base, base, base)
        self.out = nn.Conv2d(base, 1, 1)
        # The output layers are created after the segmentation layers, and the attention layer last (see the top).
        self.einstein_radius = bool(einstein_radius)
        if self.einstein_radius:
            self.radius = nn.Linear(16 * base, 1)
        self.lens_output = bool(lens_output)
        if self.lens_output:
            self.lens = nn.Linear(16 * base, 2)
        self.attention = AttentionPooling(16 * base) if attention else None

    def forward(self, image, return_einstein_radius=False, return_lens_logits=False, return_attention=False):
        """Logits of the arc probability of every pixel; with return_einstein_radius=True, also the estimated
        Einstein radius (arcseconds) of every image; with return_lens_logits=True, also the lens logits; with
        return_attention=True, also the 6 x 6 weights of the readout. The extra outputs follow in this order."""
        level_0 = self.inc(image)              # 101 x 101
        level_1 = self.d1(level_0)             # 50 x 50
        level_2 = self.d2(level_1)             # 25 x 25
        level_3 = self.d3(level_2)             # 12 x 12
        bottleneck = self.d4(level_3)          # 6 x 6
        features = self.u1(bottleneck, level_3)
        features = self.u2(features, level_2)
        features = self.u3(features, level_1)
        features = self.u4(features, level_0)
        logits = self.out(features)
        if not (return_einstein_radius or return_lens_logits or return_attention):
            return logits
        pooled, weights = pool_bottleneck(bottleneck, self.attention, return_weights=True)
        outputs = [logits]
        if return_einstein_radius:
            if not self.einstein_radius:
                raise ValueError("This U-Net has no Einstein radius output")
            outputs.append(self.radius(pooled).squeeze(1))
        if return_lens_logits:
            if not self.lens_output:
                raise ValueError("This U-Net has no lens output")
            outputs.append(self.lens(pooled))
        if return_attention:
            outputs.append(weights)
        return tuple(outputs)


def build_unet(model, dropout=0.10, attention=False, lens_output=False):
    """A new U-Net with random weights; model is starlight or radius-mini."""
    return UNet(base=FILTERS[model], dropout=dropout, einstein_radius=model in EINSTEIN_RADIUS_MODELS,
                lens_output=lens_output, attention=attention)


def architecture_of(model, dropout=0.10, attention=False, lens_output=False):
    """The constructor arguments of UNet for a model, as stored with a saved network (einstein_radius, lens_output
    and attention only if they are used)."""
    architecture = {"in_channels": 1, "base": FILTERS[model], "dropout": dropout}
    if model in EINSTEIN_RADIUS_MODELS:
        architecture["einstein_radius"] = True
    if lens_output:
        architecture["lens_output"] = True
    if attention:
        architecture["attention"] = True
    return architecture


# ====================================================================== 3. classifier and radius regressor

class EncoderClassifier(nn.Module):
    """The encoder of a U-Net, its readout and a linear layer with two outputs (non-lens, lens). For a U-Net with a
    lens output, that layer is used; otherwise a new one is created."""

    def __init__(self, unet):
        super().__init__()
        self.encoder = nn.ModuleDict({block: getattr(unet, block) for block in ENCODER_BLOCKS})
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.attention = getattr(unet, "attention", None)
        self.classifier = unet.lens if getattr(unet, "lens_output", False) else nn.Linear(unet.bottleneck_maps, 2)

    def features(self, images):
        """The bottleneck feature maps, averaged over the 6 x 6 positions (with attention: weighted by the attention
        map)."""
        feature_maps = images
        for block in ENCODER_BLOCKS:
            feature_maps = self.encoder[block](feature_maps)
        if self.attention is not None:
            return self.attention(feature_maps)
        return self.pooling(feature_maps).flatten(1)

    def forward(self, images):
        return self.classifier(self.features(images))


def build_classifier(saved_network, initialisation):
    """The classifier with the architecture of a saved U-Net. initialisation "pretrained": the encoder, the attention
    layer and the lens output start from the weights of the saved network; "random": from random weights. The
    complete U-Net is built before a new linear layer, so that for a given seed the random weights are the same in
    both cases."""
    if initialisation not in ("pretrained", "random"):
        raise ValueError(f"Unknown initialisation: {initialisation}")
    unet = UNet(**saved_network["architecture"])
    if initialisation == "pretrained":
        unet.load_state_dict(saved_network["model_state_dict"])
    return EncoderClassifier(unet)


class RadiusRegressor(nn.Module):
    """The encoder of UNet-Radius-mini, its readout and its Einstein radius layer."""

    def __init__(self, unet):
        super().__init__()
        self.encoder = nn.ModuleDict({block: getattr(unet, block) for block in ENCODER_BLOCKS})
        self.radius = unet.radius
        self.attention = getattr(unet, "attention", None)

    def forward(self, images):
        feature_maps = images
        for block in ENCODER_BLOCKS:
            feature_maps = self.encoder[block](feature_maps)
        return self.radius(pool_bottleneck(feature_maps, self.attention)).squeeze(1)


def build_regressor(saved_network, pretrained):
    """The radius regressor with the architecture of a saved UNet-Radius-mini. pretrained=True: encoder, attention
    layer and radius layer start from the weights of the saved network; pretrained=False: from random weights."""
    unet = UNet(**saved_network["architecture"])
    if pretrained:
        unet.load_state_dict(saved_network["model_state_dict"])
    return RadiusRegressor(unet)


# ================================================================================================ 4. saved networks

def load_saved_network(path):
    """A network saved by train_copies.py: weights, architecture, input transform and decision rule."""
    return torch.load(path, map_location="cpu")


def model_of_saved_network(saved_network):
    """starlight or radius-mini, from the presence of the Einstein radius output and the number of filters of the
    first convolutional block."""
    architecture = saved_network["architecture"]
    if architecture.get("einstein_radius", False):
        return "radius-mini"
    if int(architecture.get("base", 16)) == FILTERS["starlight"]:
        return "starlight"
    raise ValueError(f"No model has {architecture.get('base')} filters in the first convolutional block")


def choose_device(name="auto"):
    """The processor on which the networks run: cpu, cuda (a GPU), or auto (a GPU if one is available)."""
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("A GPU was requested, but none is available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
