import torch
from habitat_baselines.rl.ddppo.algo.ddppo import DecentralizedDistributedMixin
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
        kl_coeff,
        disc_coeff,
        recon_coeff,
        lr=None,
        eps=None,
        *args,
        **kwargs,
    ):
        super().__init__(lr=lr, eps=eps, *args, **kwargs)
        self.observations = None
        self.reconstruction_criterion = nn.L1Loss(reduction="mean")
        self.discriminator_criterion = nn.BCELoss(reduction="mean")
        self.sampled_real_vis_feats = None
        self.sampled_sim_vis_feats = None
        self.real_vis_feats = None
        self.real_dataloader = get_dataloader(dir_names, batch_size)
        self.kl_coeff = kl_coeff
        self.disc_coeff = disc_coeff
        self.recon_coeff = recon_coeff
        self.losses = {}

        self.encoder_optimizer = optim.Adam(
            list(
                filter(
                    lambda p: p.requires_grad,
                    self.actor_critic.net.visual_encoder.parameters(),
                )
            ),
            lr=lr,
            eps=eps,
        )

    @staticmethod
    def _sample_vis_feats(mu_std):
        dist = Normal(*mu_std)
        return dist.rsample()

    def sim_reconstruction_loss(self, sim_img, sampled_sim_vis_feats):
        sim_depth_pred = self.actor_critic.net.sim_visual_decoder(sampled_sim_vis_feats)
        return self.reconstruction_criterion(sim_depth_pred, sim_img)

    def reality_reconstruction_loss(self, real_img, sampled_real_vis_feats):
        real_depth_pred = self.actor_critic.net.reality_visual_decoder(
            sampled_real_vis_feats
        )
        return self.reconstruction_criterion(real_depth_pred, real_img)

    def kl_divergence_loss(self, sim_vis_feats, real_vis_feats):
        sim_encoder_dist = Normal(*sim_vis_feats)
        real_encoder_dist = Normal(*real_vis_feats)
        unit_normal = Normal(
            torch.zeros_like(real_vis_feats[0]),
            torch.ones_like(real_vis_feats[0]),
        )

        return (
            kl_divergence(sim_encoder_dist, unit_normal).mean()
            + kl_divergence(real_encoder_dist, unit_normal).mean()
        )

    def discriminator_loss(self, sampled_sim_vis_feats, sampled_real_vis_feats):
        vis_feats = torch.cat([sampled_sim_vis_feats, sampled_real_vis_feats], dim=0)
        preds = self.actor_critic.net.discriminator(vis_feats)

        num_samples = sampled_sim_vis_feats.shape[0]
        labels = torch.ones_like(preds)
        labels[:num_samples, :] = 0
        return self.discriminator_criterion(preds, labels)

    def before_backward(self, loss):
        self.losses["sim_recon_loss"] = (
            self.recon_coeff * self.sim_reconstruction_loss()
        )
        self.losses[
            "real_recon_loss"
        ] = self.recon_coeff * self.reality_reconstruction_loss(
            self.real_img, self.sampled_real_vis_feats
        )
        self.losses["kl_loss"] = self.kl_coeff * self.kl_divergence_loss(
            self.sim_vis_feats, self.real_vis_feats
        )
        self.disc_loss = self.disc_coeff * self.discriminator_loss(
            self.sampled_sim_vis_feats, self.sampled_real_vis_feats
        )
        self.losses["disc_loss"] = self.disc_loss

        for v in self.losses.values():
            loss += v

    def after_backward(self, loss):
        # # want to encourage generating sim encodings similar to real encodings
        # vis_feats = torch.cat([sampled_sim_vis_feats, sampled_real_vis_feats], dim=0)
        # preds = self.actor_critic.net.discriminator(vis_feats)
        #
        # -self.disc_loss.backwards()
        #
        # self.optimizer.step()
        #
        # # freeze decoder
        # ## gradient ascent on visual encoder
        pass

    def _evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        self.observations = observations
        real_img = next(iter(self.real_dataloader)).to(
            device=self.actor_critic.net.visual_features[0].device
        )

        (
            values,
            action_log_probs,
            dist_entropy,
            hx,
            self.sim_img,
            self.real_img,
            self.sim_vis_feats,
            self.real_vis_feats,
            self.sampled_sim_vis_feats,
            self.sampled_real_vis_feats,
        ) = self.actor_critic.evaluate_actions(
            observations, rnn_hidden_states, prev_actions, masks, action, real_img
        )
        return values, action_log_probs, dist_entropy, hx


class OutdoorDDPPO(DecentralizedDistributedMixin, OutdoorPPO):
    def _evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        self.observations = observations
        real_img = next(iter(self.real_dataloader)).to(
            device=self.net.visual_features[0].device
        )
        real_obs = {"depth": real_img}

        (
            values,
            action_log_probs,
            dist_entropy,
            hx,
            self.sim_img,
            self.real_img,
            self.sim_vis_feats,
            self.real_vis_feats,
            self.sampled_sim_vis_feats,
            self.sampled_real_vis_feats,
        ) = self._evaluate_actions_wrapper.ddp(
            observations, rnn_hidden_states, prev_actions, masks, action, real_obs
        )
        return values, action_log_probs, dist_entropy, hx
