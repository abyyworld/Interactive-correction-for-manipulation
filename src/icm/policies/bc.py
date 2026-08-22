"""Behaviour-cloning policy: encoder(s) -> MLP -> action chunk.

Three choices that matter more than the architecture
----------------------------------------------------
**Action chunking.** The policy predicts ``chunk`` future actions from one
observation and they are blended at execution time. Single-step behaviour
cloning compounds its own error: a small mistake moves the robot slightly
off-distribution, the next prediction is worse, and the trajectory diverges.
Predicting a short horizon and temporally ensembling the overlapping
predictions is the cheapest known fix and costs one extra output layer.

**L1 loss, not L2.** Demonstration actions are multi-modal - there are several
equally good ways to approach a cube. L2 regresses to the mean of the modes,
which for manipulation is often an action that does none of them. L1 is far more
tolerant of that.

**uint8 images normalised on device.** Keeping images as bytes until the last
moment is a 4x memory saving through the loader, which is what makes a
multi-gigabyte image dataset trainable on an 8 GB card.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class PolicyConfig:
    state_dim: int = 24
    action_dim: int = 7
    chunk: int = 8
    hidden: tuple[int, ...] = (512, 512)
    image_keys: tuple[str, ...] = ()
    image_size: int = 84
    backbone: str = "resnet18"
    pretrained: bool = False  # offline-safe default; set True on a networked box
    feature_dim: int = 256
    dropout: float = 0.1
    #: Vocabulary for templated language conditioning. Empty disables it.
    vocab: tuple[str, ...] = ()
    film: bool = True

    @property
    def uses_images(self) -> bool:
        return bool(self.image_keys)

    @property
    def uses_language(self) -> bool:
        return bool(self.vocab)


class SpatialSoftmax(nn.Module):
    """Collapse a feature map to expected 2D keypoint locations.

    For manipulation this is a much better inductive bias than global average
    pooling: what matters is *where* the cube is, and average pooling discards
    exactly that.
    """

    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        self.channels = channels
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, width), np.linspace(-1.0, 1.0, height))
        self.register_buffer("pos_x", torch.from_numpy(pos_x.reshape(-1)).float())
        self.register_buffer("pos_y", torch.from_numpy(pos_y.reshape(-1)).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = x.reshape(b * c, h * w)
        attn = F.softmax(flat, dim=-1)
        exp_x = (attn * self.pos_x).sum(dim=1)
        exp_y = (attn * self.pos_y).sum(dim=1)
        return torch.stack([exp_x, exp_y], dim=1).reshape(b, c * 2)

    @property
    def out_dim(self) -> int:
        return self.channels * 2


def build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return ``(feature_extractor, channels)`` keeping the spatial map intact.

    torchvision is imported lazily and only for the ResNet variants, so the
    compact "small" backbone - used by the tests and by CPU-only runs - has no
    vision dependency at all.
    """
    if name == "small":
        # Compact CNN for smoke tests and CPU-only runs.
        net = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        return net, 128

    import torchvision

    if name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        layers = nn.Sequential(*list(net.children())[:-2])  # keep the spatial map
        return layers, 512
    if name == "resnet34":
        weights = torchvision.models.ResNet34_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet34(weights=weights)
        return nn.Sequential(*list(net.children())[:-2]), 512
    raise ValueError(f"unknown backbone {name!r}")


def _feature_shape(net: nn.Module, image_size: int) -> tuple[int, int, int]:
    """Channels, height, width of ``net``'s output for a square input."""
    was_training = net.training
    net.eval()
    with torch.no_grad():
        out = net(torch.zeros(1, 3, image_size, image_size))
    net.train(was_training)
    return int(out.shape[1]), int(out.shape[2]), int(out.shape[3])


class BCPolicy(nn.Module):
    """Predicts a chunk of future actions from proprioception, images and language."""

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        feat_dims: list[int] = []

        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.feature_dim, config.feature_dim),
        )
        feat_dims.append(config.feature_dim)

        self.backbones = nn.ModuleDict()
        self.pools = nn.ModuleDict()
        if config.uses_images:
            for key in config.image_keys:
                net, ch = build_backbone(config.backbone, config.pretrained)
                self.backbones[key] = net
                # Probe the real output shape rather than deriving it from the
                # stride arithmetic. ResNet-18 on an 84 px input yields 3x3, not
                # 84//32 = 2, because of the padding in its stem - a mismatch that
                # only surfaces as a broadcast error at the first forward pass.
                c, h, w = _feature_shape(net, config.image_size)
                pool = SpatialSoftmax(c, h, w)
                self.pools[key] = pool
                feat_dims.append(pool.out_dim)

        self.lang_embed = None
        self.film = None
        if config.uses_language:
            self.lang_embed = nn.Sequential(
                nn.Linear(len(config.vocab), config.feature_dim), nn.ReLU(inplace=True)
            )
            if config.film:
                # FiLM conditions the trunk multiplicatively, which lets the
                # instruction gate visual features rather than merely being
                # concatenated and ignored.
                self.film = nn.Linear(config.feature_dim, 2 * sum(feat_dims))
            else:
                feat_dims.append(config.feature_dim)

        in_dim = sum(feat_dims)
        layers: list[nn.Module] = []
        for h in config.hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True), nn.Dropout(config.dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, config.chunk * config.action_dim))
        self.head = nn.Sequential(*layers)

        self.register_buffer("img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        # Proprioceptive/state inputs are raw physical quantities on wildly
        # different scales - joint angles near 3 rad beside object heights near
        # 0.02 m - and several dimensions are near-constant. Feeding that to an
        # MLP unnormalised conditions the problem badly enough that the policy
        # does not learn the task at all. Registered as buffers so the statistics
        # travel with the checkpoint and inference cannot silently disagree with
        # training.
        self.register_buffer("state_mean", torch.zeros(config.state_dim))
        self.register_buffer("state_std", torch.ones(config.state_dim))

    def set_normalization(self, mean, std, min_std: float = 1e-2) -> None:
        """Install input statistics. ``min_std`` prevents amplifying constant dims.

        Ten of the twenty-seven state dimensions here barely vary (a fixed goal
        position, near-identity object quaternions). Dividing those by their true
        standard deviation turns numerical noise into a large input, so the
        deviation is floored.
        """
        mean = torch.as_tensor(np.asarray(mean), dtype=torch.float32)
        std = torch.as_tensor(np.asarray(std), dtype=torch.float32).clamp(min=min_std)
        self.state_mean.copy_(mean)
        self.state_std.copy_(std)

    def encode_image(self, key: str, image: torch.Tensor) -> torch.Tensor:
        if image.dtype == torch.uint8:
            image = image.float().div_(255.0)
        if image.ndim == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        image = (image - self.img_mean) / self.img_std
        return self.pools[key](self.backbones[key](image))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        state = (batch["state"] - self.state_mean) / self.state_std
        feats = [self.state_encoder(state)]
        for key in self.config.image_keys:
            feats.append(self.encode_image(key, batch[key]))
        x = torch.cat(feats, dim=-1)

        if self.lang_embed is not None:
            lang = self.lang_embed(batch["instruction"])
            if self.film is not None:
                gamma, beta = self.film(lang).chunk(2, dim=-1)
                x = x * (1.0 + gamma) + beta
            else:
                x = torch.cat([x, lang], dim=-1)

        out = self.head(x)
        return out.reshape(-1, self.config.chunk, self.config.action_dim)

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        pred = self(batch)
        target = batch["action"]
        valid = batch["action_valid"].unsqueeze(-1)
        weight = batch.get("weight")
        per_elem = F.l1_loss(pred, target, reduction="none") * valid
        # Divide by the number of summed *elements*, not the number of valid
        # timesteps: `valid` is (B, T, 1) while `per_elem` is (B, T, A), so using
        # valid.sum() alone reports action_dim times the true mean absolute
        # error. That does not change the gradient direction, but it makes the
        # logged loss incomparable to any baseline - it read 0.55 when the true
        # error was 0.078, which looked like a policy that had learned nothing.
        n_elem = valid.sum(dim=(1, 2)).clamp(min=1.0) * pred.shape[-1]
        per_sample = per_elem.sum(dim=(1, 2)) / n_elem
        if weight is not None:
            loss = (per_sample * weight).sum() / weight.sum().clamp(min=1e-6)
        else:
            loss = per_sample.mean()
        with torch.no_grad():
            gripper_err = (pred[:, 0, 6] - target[:, 0, 6]).abs().mean()
            pos_err = (pred[:, 0, :3] - target[:, 0, :3]).abs().mean()
        return loss, {
            "loss": float(loss.detach()),
            "pos_l1": float(pos_err),
            "gripper_l1": float(gripper_err),
        }

    @torch.no_grad()
    def predict(self, batch: dict[str, torch.Tensor]) -> np.ndarray:
        self.eval()
        return self(batch)[0].cpu().numpy()


class TemporalEnsemble:
    """Blends overlapping chunk predictions at execution time.

    Each timestep is covered by up to ``chunk`` predictions made at different
    moments. Averaging with an exponential preference for the most recent
    prediction smooths the trajectory without adding the lag a plain moving
    average would.
    """

    #: Action indices excluded from blending. The gripper command is effectively
    #: binary (measured std 0.999 on expert data, 24% of actions saturated), so
    #: averaging eight predictions of it produces a half-closed gripper that
    #: never actually grasps. Continuous position and yaw terms benefit from
    #: smoothing; a discrete open/close decision does not.
    DISCRETE_DIMS = (6,)

    def __init__(
        self,
        chunk: int,
        action_dim: int = 7,
        decay: float = 0.35,
        discrete_dims: tuple[int, ...] | None = None,
    ):
        self.chunk = chunk
        self.action_dim = action_dim
        self.decay = decay
        self.discrete_dims = self.DISCRETE_DIMS if discrete_dims is None else discrete_dims
        self.reset()

    def reset(self) -> None:
        self._buffer: list[tuple[int, np.ndarray]] = []
        self._t = 0

    def step(self, prediction: np.ndarray) -> np.ndarray:
        self._buffer.append((self._t, np.asarray(prediction, dtype=np.float64)))
        self._buffer = [(t0, p) for t0, p in self._buffer if self._t - t0 < self.chunk]
        num = np.zeros(self.action_dim)
        den = 0.0
        for t0, pred in self._buffer:
            age = self._t - t0
            w = float(np.exp(-self.decay * age))
            num += w * pred[age]
            den += w
        blended = num / max(den, 1e-8)
        # Discrete dimensions take the newest prediction rather than an average.
        newest = self._buffer[-1][1][0]
        for d in self.discrete_dims:
            if d < self.action_dim:
                blended[d] = newest[d]
        self._t += 1
        return blended
