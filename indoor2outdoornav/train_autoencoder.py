import os
import cv2
import tqdm
import torch
import numpy as np
import argparse
import time

from torch import optim as optim
import torch.nn.functional as F
from indoor2outdoornav.outdoor_policy import *
from indoor2outdoornav.dataloader import get_dataloader
from torch.distributions import Normal, kl_divergence
from gym import spaces
from gym.spaces import Dict as SpaceDict

from torch.utils.tensorboard import SummaryWriter

MODE = 'train'
TY = 'sim'

VAR = True
DEVICE = 'cuda'
BATCH_SIZE = 32
LR = 2.5e-4
EPS = 1e-5
EPOCHS = 100000
NUM_BATCHES_PER_EPOCH = 1000
NUM_TEST_BATCHES_PER_EPOCH = 50
LOSS_COEFF = 0.00001
SAVE_FREQ = 1000

class VAE(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__() 

        observation_space = SpaceDict(
            {
                "depth": spaces.Box(
                    low=0.0, high=1.0, shape=(256, 256, 1), dtype=np.float32
                ),
            }
        )

        if VAR:
            # self.encoder = StochasticCNN(observation_space, hidden_size)
            self.encoder = StochasticCNNv2(observation_space, hidden_size)
        else:
            self.encoder = SimpleCNN(observation_space, hidden_size)

        self.encoder.using_one_camera = True
        self.decoder = Decoder(hidden_size)

def _sample_vis_feats(mu_std):
    dist = Normal(*mu_std)
    return dist.rsample()

def evaluate_model(vae, val_loader, out_img_dir, total_num_steps, writer, save=False):
    vae.eval()
    total_val_loss = 0
    n_its = 0
    with torch.no_grad():
        data_iter = iter(val_loader)
        for _ in range(NUM_TEST_BATCHES_PER_EPOCH):
            data = next(data_iter).to(DEVICE)
            n_its += 1
            obs = {"depth": data}
            vis_feats = vae.encoder(obs)
            if VAR:
                samp_vis_feats =  _sample_vis_feats(vis_feats)
                pred = vae.decoder(samp_vis_feats)
            else:
                pred = vae.decoder(vis_feats)

            # losses
            # recon_loss = F.l1_loss(pred, obs['depth'], reduction="none")
            recon_loss = calc_recon_loss(pred, obs["depth"])

            kl_loss = 0
            if VAR:
                kl_loss = calc_kl_loss(vis_feats, samp_vis_feats)

            # total_loss = (l1_loss + kl_loss).mean()
            total_loss = (kl_loss - recon_loss).mean() * LOSS_COEFF
            total_val_loss += total_loss.item()


            if save:
                save_gt = data.detach().cpu().numpy().squeeze() * 255
                save_pred = pred.detach().cpu().numpy().squeeze() * 255
                pred_save_pth = os.path.join(
                    f"{out_img_dir}/depth_pred_{n_its:06}.png"
                )
                cv2.imwrite(pred_save_pth, save_pred)

                # gt_save_pth = os.path.join(
                #     f"{OUT_IMG_DIR}/depth_gt_{n_its:06}.png"
                # )
                # cv2.imwrite(gt_save_pth, save_gt)

    total_val_loss /= n_its
    print("total val loss: ", total_val_loss)
    writer.add_scalar("val_loss", total_val_loss, total_num_steps)

    return total_val_loss

def calc_kl_loss(mu_std, samp_vis_feats):
    p = Normal(
                torch.zeros_like(mu_std[0]),
                torch.ones_like(mu_std[1]),
    )
    q = Normal(*mu_std)
    log_qzx = q.log_prob(samp_vis_feats)
    log_pz = p.log_prob(samp_vis_feats)

    kl_loss = (log_qzx - log_pz)
    
    # kl_loss = kl_divergence(enc_dist, unit_normal).mean()

    return kl_loss.sum(-1)

def calc_recon_loss(pred, gt):
    log_scale = nn.Parameter(torch.Tensor([0.0])).to(DEVICE)
    scale = torch.exp(log_scale)
    dist = Normal(pred, scale)
    log_pxz = dist.log_prob(gt)
    # recon_loss = F.l1_loss(pred, obs['depth'], reduction="none")

    return log_pxz.sum(dim=(1, 2, 3))

def train_model(vae, train_loader, optimizer, total_num_steps, ckpt_dir, writer):
    vae.train()
    data_iter = iter(train_loader)
    for _ in range(NUM_BATCHES_PER_EPOCH):
        data = next(data_iter).to(DEVICE)
        obs = {"depth": data}
        vis_feats = vae.encoder(obs)
        if VAR:
            samp_vis_feats =  _sample_vis_feats(vis_feats)
            pred = vae.decoder(samp_vis_feats)
        else:
            pred = vae.decoder(vis_feats)

        ## RECONSTRUCTION LOSS
        recon_loss = calc_recon_loss(pred, obs["depth"])

        ### KL LOSS
        kl_loss = 0
        if VAR:
            kl_loss = calc_kl_loss(vis_feats, samp_vis_feats)
        total_loss = (kl_loss - recon_loss).mean() * LOSS_COEFF
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        total_num_steps +=1 
        print("total loss: ", total_loss, total_num_steps)
        if total_num_steps % SAVE_FREQ == 0:
            writer.add_scalar("train_loss", total_loss.item(), total_num_steps)
            checkpoint = {
                    "state_dict": {
                        "actor_critic." + k: v
                        for k, v in vae.state_dict().items()
                    },
                }
            save_pth = os.path.join(ckpt_dir, f"ckpt.{total_num_steps}.pth")
            torch.save(
                    checkpoint,
                    save_pth,
                )
            print('saved checkpoint to: ', save_pth)

    return total_num_steps


def main(args):
    if args.ty == 'real':
        train_dirs = [ "/coc/testnvme/jtruong33/data/outdoor_imgs/bay_trail/filtered",
                      "/coc/testnvme/jtruong33/data/outdoor_imgs/outdoor_inspection_route/filtered",
                      "/coc/testnvme/jtruong33/data/outdoor_imgs/2022-04-19-14-17-58/filtered"]
        val_dirs = ["/coc/testnvme/jtruong33/data/outdoor_imgs/2022-04-22-16-56-00/filtered"]
    elif args.ty == 'sim':
        train_dirs = [ "/coc/testnvme/jtruong33/data/outdoor_imgs/sim_hm3d_eval_imgs"]
        val_dirs = ["/coc/testnvme/jtruong33/data/outdoor_imgs/sim_hm3d_eval_imgs"]

    vae = VAE(hidden_size=256)
    vae.to(DEVICE)

    if args.ext != '':
        args.ext += '_'

    timestr = time.strftime("%Y%m%d-%H%M%S")
    CKPT_DIR = os.path.join("checkpoints", args.ty + '_' + args.ext + timestr)

    OUT_IMG_DIR = os.path.join("results", "pred_imgs_" + args.ty + '_' + args.ext + timestr)
    total_num_steps = 0

    writer = SummaryWriter(os.path.join(CKPT_DIR, "tb"), flush_secs=5)


    if args.eval:
        os.makedirs(OUT_IMG_DIR, exist_ok=True)

        checkpoint = torch.load(args.ckpt, map_location=DEVICE)
        val_loader = get_dataloader(val_dirs, 1)
        vae.load_state_dict(
            {k[len("actor_critic.") :]: v for k, v in checkpoint["state_dict"].items()},
            strict=False,
        )

        total_val_loss = evaluate_model(
                vae,
                val_loader,
                OUT_IMG_DIR,
                total_num_steps,
                writer,
                save=True
            )
        print("saved images to: ", os.path.abspath(OUT_IMG_DIR))
    else:
        os.makedirs(CKPT_DIR, exist_ok=True)
        train_loader = get_dataloader(train_dirs, BATCH_SIZE)
        val_loader = get_dataloader(val_dirs, BATCH_SIZE)
        params = list(vae.parameters())
        optimizer = optim.Adam(params, lr=LR, eps=EPS)

        for epoch in tqdm.tqdm(range(0, EPOCHS + 1)):
            total_num_steps = train_model(
                vae,
                train_loader,
                optimizer,
                total_num_steps,
                CKPT_DIR,
                writer
            )
            total_val_loss = evaluate_model(
                vae,
                val_loader,
                OUT_IMG_DIR,
                total_num_steps,
                writer,
                save=False
            )
        writer.flush()
        writer.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--ty", default='sim')
    parser.add_argument("--ext", default='')
    parser.add_argument("-e", "--eval", default=False, action="store_true")
    parser.add_argument("-cpt", "--ckpt", default=None)
    args = parser.parse_args()
    main(args)
 