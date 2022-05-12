import gym.spaces
import torch
import torch.nn.functional as F
from habitat import Config
from habitat.core import spaces
from habitat.tasks.nav.nav import IntegratedPointGoalGPSAndCompassSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.simple_cnn import SimpleCNN
from habitat_baselines.rl.ppo.policy import PointNavBaselineNet, Policy
from torch import nn
from torch.distributions import Normal, kl_divergence

PointNavBaselinePolicy = baseline_registry.get_policy("PointNavBaselinePolicy")


@baseline_registry.register_policy
class OutdoorPolicy(PointNavBaselinePolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        action_distribution_type: str = "gaussian",
        kl_coeff: float = 0.125,
        disc_coeff: float = 8.0,
        enc_gen_coeff: float = 1.0,
        recon_coeff: float = 1.0,
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
        self.sim_recon_loss = None
        self.real_recon_loss = None
        self.kl_loss = None
        self.disc_loss = None
        self.kl_coeff = kl_coeff
        self.disc_coeff = disc_coeff
        self.enc_gen_coeff = enc_gen_coeff
        self.recon_coeff = recon_coeff
        self.reconstruction_criterion = nn.L1Loss(reduction="mean")
        self.discriminator_criterion = nn.BCELoss(reduction="mean")

    @classmethod
    def from_config(cls, config: Config, observation_space: spaces.Dict, action_space):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            kl_coeff=config.RL.OUTDOOR.kl_coeff,
            disc_coeff=config.RL.OUTDOOR.disc_coeff,
            enc_gen_coeff=config.RL.OUTDOOR.enc_gen_coeff,
            recon_coeff=config.RL.OUTDOOR.recon_coeff,
        )

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        sim_img,
        real_img,
    ):
        (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
        ) = super().evaluate_actions(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            action,
        )

        (kl_loss, disc_loss, enc_gen_loss,) = self.calculate_losses(
            sim_img,
            real_img,
        )

        return (
            value,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            kl_loss,
            disc_loss,
            enc_gen_loss,
        )

    def after_step(self, sim_img, sim_obs, real_img):
        sim_recon_loss = self.sim_reconstruction_loss(sim_img, sim_obs)
        real_recon_loss = self.reality_reconstruction_loss(real_img)
        return sim_recon_loss, real_recon_loss

    @staticmethod
    def _sample_vis_feats(mu_std):
        dist = Normal(*mu_std)
        return dist.rsample()

    def sim_reconstruction_loss(self, sim_img, sim_obs):
        sim_vis_feats = self.net.visual_encoder(sim_obs)
        sampled_sim_vis_feats = self._sample_vis_feats(sim_vis_feats)

        sim_depth_pred = self.net.sim_visual_decoder(sampled_sim_vis_feats)
        return self.reconstruction_criterion(sim_depth_pred, sim_img)

    def reality_reconstruction_loss(self, real_img):
        real_obs = {"depth": real_img}
        real_vis_feats = self.net.reality_visual_encoder(real_obs)
        sampled_real_vis_feats = self._sample_vis_feats(real_vis_feats)

        real_depth_pred = self.net.reality_visual_decoder(sampled_real_vis_feats)
        return self.reconstruction_criterion(real_depth_pred, real_img)

    def kl_divergence_loss(self, sim_vis_feats, real_vis_feats):
        sim_encoder_dist = Normal(*sim_vis_feats)
        real_encoder_dist = Normal(*real_vis_feats)
        sim_unit_normal = Normal(
            torch.zeros_like(sim_vis_feats[0]),
            torch.ones_like(sim_vis_feats[0]),
        )

        real_unit_normal = Normal(
            torch.zeros_like(real_vis_feats[0]),
            torch.ones_like(real_vis_feats[0]),
        )

        return (
            kl_divergence(sim_encoder_dist, sim_unit_normal).mean()
            + kl_divergence(real_encoder_dist, real_unit_normal).mean()
        )

    def discriminator_loss(self, sampled_sim_vis_feats, sampled_real_vis_feats):
        pred_real = self.net.discriminator(sampled_real_vis_feats)
        labels_real = torch.ones_like(pred_real)
        loss_D_real = self.discriminator_criterion(pred_real, labels_real)

        pred_sim = self.net.discriminator(sampled_sim_vis_feats)
        labels_sim = torch.zeros_like(pred_sim)
        loss_D_sim = self.discriminator_criterion(pred_sim, labels_sim)

        return loss_D_real + loss_D_sim

    def encoder_generator_loss(self, sim_obs):
        sim_vis_feats = self.net.visual_encoder(sim_obs)
        sampled_sim_vis_feats = self._sample_vis_feats(sim_vis_feats)

        preds = self.net.discriminator(sampled_sim_vis_feats)
        # labels are real for generator
        labels = torch.ones_like(preds)
        enc_gen_loss = self.discriminator_criterion(preds, labels)
        return enc_gen_loss

    def calculate_losses(
        self,
        sim_img,
        real_img,
    ):
        width = sim_img.shape[2]
        sim_right_depth, sim_left_depth = torch.split(sim_img, int(width / 2), 2)
        sim_obs = {
            "spot_right_depth": sim_right_depth,
            "spot_left_depth": sim_left_depth,
        }
        real_obs = {"depth": real_img}
        #
        # sim_recon_loss = self.recon_coeff * self.sim_reconstruction_loss(
        #     sim_img, sim_obs
        # )
        # real_recon_loss = self.recon_coeff * self.reality_reconstruction_loss(real_img)

        sim_vis_feats = self.net.visual_encoder(sim_obs)
        real_vis_feats = self.net.reality_visual_encoder(real_obs)

        sampled_sim_vis_feats = self._sample_vis_feats(sim_vis_feats)
        sampled_real_vis_feats = self._sample_vis_feats(real_vis_feats)

        kl_loss = self.kl_coeff * self.kl_divergence_loss(sim_vis_feats, real_vis_feats)
        disc_loss = self.disc_coeff * self.discriminator_loss(
            sampled_sim_vis_feats, sampled_real_vis_feats
        )
        enc_gen_loss = self.enc_gen_coeff * self.encoder_generator_loss(sim_obs)
        return kl_loss, disc_loss, enc_gen_loss


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
        return F.interpolate(
            x, size=self.output_size, mode="bilinear", align_corners=False
        )


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
