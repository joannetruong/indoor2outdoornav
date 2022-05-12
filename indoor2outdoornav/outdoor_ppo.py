import torch
from habitat_baselines.rl.ddppo.algo.ddppo import \
    OutdoorDecentralizedDistributedMixin
from habitat_baselines.rl.ppo import PPO
from torch import nn, optim
from torch.distributions import Normal, kl_divergence

from indoor2outdoornav.dataloader import get_dataloader

"""
2x reconstruction losses (sim, real)
KL divergence loss for VAE
discriminator loss

"""


class OutdoorPPO(PPO):
    def __init__(
        self,
        dir_names,
        batch_size,
        lr=None,
        eps=None,
        use_second_optimizer=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            lr=lr, eps=eps, use_second_optimizer=use_second_optimizer, *args, **kwargs
        )
        self.observations = None
        self.real_dataloader = get_dataloader(dir_names, batch_size)
        self.losses = {}
        self.decoder_losses = {}
        self.use_second_optimizer = use_second_optimizer

    def after_step(self):
        width = self.sim_img.shape[2]
        sim_right_depth, sim_left_depth = torch.split(self.sim_img, int(width / 2), 2)
        sim_obs = {
            "spot_right_depth": sim_right_depth,
            "spot_left_depth": sim_left_depth,
        }
        sim_recon_loss = self.actor_critic.sim_reconstruction_loss(
            self.sim_img, sim_obs
        )
        real_recon_loss = self.actor_critic.reality_reconstruction_loss(self.real_img)
        self.decoder_losses["sim_recon_loss"] = sim_recon_loss
        self.decoder_losses["real_recon_loss"] = real_recon_loss
        # pass

    def _evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        self.observations = observations
        real_img = next(iter(self.real_dataloader)).to(
            device=self.actor_critic.net.visual_features[0].device
        )

        sim_img = torch.cat(
            [
                # Spot is cross-eyed; right is on the left on the FOV
                observations["spot_right_depth"],
                observations["spot_left_depth"],
            ],
            dim=2,
        )  # NHWC
        self.sim_img = sim_img
        self.real_img = real_img

        (
            values,
            action_log_probs,
            dist_entropy,
            rnn_hidden_states,
            kl_loss,
            disc_loss,
            enc_disc_loss,
        ) = self.actor_critic.evaluate_actions(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            action,
            sim_img,
            real_img,
        )

        self.losses["kl_loss"] = kl_loss
        self.losses["disc_loss"] = disc_loss
        self.losses["enc_disc_loss"] = enc_disc_loss

        return values, action_log_probs, dist_entropy, rnn_hidden_states


class OutdoorDDPPO(OutdoorDecentralizedDistributedMixin, OutdoorPPO):
    def _evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        self.observations = observations
        real_img = next(iter(self.real_dataloader)).to(
            device=self.actor_critic.net.visual_features[0].device
        )

        sim_img = torch.cat(
            [
                # Spot is cross-eyed; right is on the left on the FOV
                observations["spot_right_depth"],
                observations["spot_left_depth"],
            ],
            dim=2,
        )  # NHWC
        self.sim_img = sim_img
        self.real_img = real_img

        (
            values,
            action_log_probs,
            dist_entropy,
            rnn_hidden_states,
            kl_loss,
            disc_loss,
            enc_disc_loss,
        ) = self._evaluate_actions_wrapper.ddp(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            action,
            sim_img,
            real_img,
        )

        self.losses["kl_loss"] = kl_loss
        self.losses["disc_loss"] = disc_loss
        self.losses["enc_gen_loss"] = enc_disc_loss

        return values, action_log_probs, dist_entropy, rnn_hidden_states

    def after_step(self):
        width = self.sim_img.shape[2]
        sim_right_depth, sim_left_depth = torch.split(self.sim_img, int(width / 2), 2)
        sim_obs = {
            "spot_right_depth": sim_right_depth,
            "spot_left_depth": sim_left_depth,
        }
        sim_recon_loss, real_recon_loss = self._after_step_wrapper.ddp(
            self.sim_img, sim_obs, self.real_img
        )

        self.decoder_losses["sim_recon_loss"] = sim_recon_loss
        self.decoder_losses["real_recon_loss"] = real_recon_loss
