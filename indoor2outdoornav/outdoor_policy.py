import gym.spaces
import torch
import torch.nn.functional as F
from habitat.core import spaces
from habitat.tasks.nav.nav import IntegratedPointGoalGPSAndCompassSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.rl.ppo.policy import PointNavBaselineNet, Policy
from torch import nn
from torch.distributions import Normal

PointNavBaselinePolicy = baseline_registry.get_policy("PointNavBaselinePolicy")


@baseline_registry.register_policy
class OutdoorPolicy(PointNavBaselinePolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        action_distribution_type: str = "gaussian",
        **kwargs,
    ):
        Policy.__init__(
            self,
            OutdoorPolicyNet(  # type: ignore
                observation_space=observation_space,
                hidden_size=hidden_size,
                **kwargs,
            ),
            action_space.n,
            action_distribution_type=action_distribution_type,
        )


class OutdoorPolicyNet(PointNavBaselineNet):
    def __init__(self, observation_space: spaces.Dict, hidden_size: int):
        super().__init__(observation_space, hidden_size)
        self.real_visual_features = None
        self.visual_features = None
        self.reality_visual_encoder = StochasticCNN(observation_space, hidden_size)
        self.reality_visual_encoder.using_one_camera = True
        self.visual_encoder = StochasticCNN(observation_space, hidden_size)
        self.reality_visual_decoder = Decoder(hidden_size)
        self.sim_visual_decoder = Decoder(hidden_size)
        self.discriminator = Discriminator(hidden_size)

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        self.visual_features = self.visual_encoder(observations)
        x = [Normal(*self.visual_features).sample()]
        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            x.append(self.tgt_encoder(goal_observations))

        x_out = torch.cat(x, dim=1)
        x_out, rnn_hidden_states = self.state_encoder(x_out, rnn_hidden_states, masks)
        return x_out, rnn_hidden_states


class Upsample(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, x):
        return F.interpolate(x, size=self.output_size, mode="bilinear")


class Decoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1024)  # reshape to 32x32
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, (3, 3), (1, 1)),  # 34x34
            Upsample((64, 64)),
            nn.ConvTranspose2d(32, 64, (3, 3), (1, 1)),  # 66x66
            Upsample((128, 128)),
            nn.ConvTranspose2d(64, 32, (3, 3), (1, 1)),  # 130x130
            Upsample((254, 254)),
            nn.ConvTranspose2d(32, 1, (3, 3), (1, 1)),  # 256x256
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(-1, 1, 32, 32)  # NCHW
        x = self.decoder(x)
        return x.permute(0, 2, 3, 1)  # NCHW --> NHWC


class StochasticCNN(SimpleCNN):
    def __init__(self, observation_space, output_size):
        super().__init__(observation_space, output_size * 2)
        self.output_size = output_size

    def forward(self, observations):
        x = super().forward(observations)
        mean, std = torch.split(x, [self.output_size, self.output_size], dim=1)
        std = torch.clamp(std, min=1e-6, max=1)
        return mean, std


class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, visual_feats):
        return self.discriminator(visual_feats)
