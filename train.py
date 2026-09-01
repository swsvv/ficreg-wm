import os
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import MLP, ARPredictor, Embedder, InverseDynamicsModel, SIGReg
from utils import ModelObjectCallBack, get_column_normalizer, get_img_preprocessor


@contextmanager
def frozen(module):
    """Temporarily set requires_grad=False on all params of `module`.

    Gradient still flows through the module's computation to its inputs,
    but IDM weights do not accumulate gradient during this forward pass.
    Used to apply IDM as a fixed judge when regularizing the forward model.
    """
    states = [(p, p.requires_grad) for p in module.parameters()]
    for p, _ in states:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, was in states:
            p.requires_grad_(was)


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    idm_weight = cfg.loss.idm.weight
    fwd_via_idm_weight = cfg.loss.fwd_via_idm.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    tgt_emb = emb[:, n_preds:]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # L_pred: next-state prediction loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    # L_sigreg: Gaussian regularizer (fp32 for numerical stability under bf16)
    output["sigreg_loss"] = self.sigreg(emb.float().transpose(0, 1))

    # L_idm: IDM trained on ground-truth transitions with detached encoder outputs
    tgt_action = batch["action"][:, :ctx_len]
    idm_pred_true = self.model.predict_action(ctx_emb.detach(), tgt_emb.detach())
    output["idm_loss"] = (idm_pred_true - tgt_action).pow(2).mean()

    # L_fwd: frozen IDM judges predicted transitions — gradients flow into predictor only
    with frozen(self.model.idm):
        fwd_action = self.model.predict_action(ctx_emb.detach(), pred_emb)
    output["fwd_via_idm_loss"] = (fwd_action - tgt_action).pow(2).mean()

    output["loss"] = (
        output["pred_loss"]
        + lambd * output["sigreg_loss"]
        + idm_weight * output["idm_loss"]
        + fwd_via_idm_weight * output["fwd_via_idm_loss"]
    )

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    with torch.no_grad():
        tgt_f = tgt_action.float()
        tgt_var = tgt_f.var(dim=(0, 1)).mean().clamp_min(1e-8)
        tgt_std = tgt_f.std(dim=(0, 1)).mean()

        def _action_stats(pred, prefix):
            pred = pred.float()
            resid = pred - tgt_f
            explained_var = 1.0 - resid.var(dim=(0, 1)).mean() / tgt_var
            return {
                f"{prefix}/mae": resid.abs().mean(),
                f"{prefix}/explained_var": explained_var,
                f"{prefix}/pred_std": pred.std(dim=(0, 1)).mean(),
                f"{prefix}/pred_mean_abs": pred.mean(dim=(0, 1)).abs().mean(),
            }

        idm_dict = {"idm/tgt_std": tgt_std}
        idm_dict.update(_action_stats(idm_pred_true, "idm/true"))
        idm_dict.update(_action_stats(fwd_action, "idm/fwd"))

        pred_flat = pred_emb.float().flatten(0, 1)
        tgt_flat = tgt_emb.float().flatten(0, 1)
        ctx_flat = ctx_emb.float().flatten(0, 1)

        idm_dict.update(
            {
                "idm/pred_tgt_cos": F.cosine_similarity(
                    pred_flat, tgt_flat, dim=-1
                ).mean(),
                "idm/pred_norm_over_ctx_norm": (
                    pred_flat.norm(dim=-1).mean()
                    / ctx_flat.norm(dim=-1).mean().clamp_min(1e-8)
                ),
            }
        )

        self.log_dict(idm_dict, on_step=True, sync_dist=True)

    with torch.no_grad():
        T = self.model.predictor.transformer
        c_proj = T.cond_proj(act_emb[:, :ctx_len])
        adaln_dict = {}

        for i, blk in enumerate(T.layers):
            mod = blk.adaLN_modulation(c_proj)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
                6, dim=-1
            )

            adaln_dict.update(
                {
                    # attention path
                    f"adaln/layer{i}/shift_msa_mean": shift_msa.mean(),
                    f"adaln/layer{i}/shift_msa_std": shift_msa.std(),
                    f"adaln/layer{i}/scale_msa_mean": scale_msa.mean(),
                    f"adaln/layer{i}/scale_msa_std": scale_msa.std(),
                    f"adaln/layer{i}/gate_msa_mean": gate_msa.mean(),
                    f"adaln/layer{i}/gate_msa_std": gate_msa.std(),
                    # mlp path
                    f"adaln/layer{i}/shift_mlp_mean": shift_mlp.mean(),
                    f"adaln/layer{i}/shift_mlp_std": shift_mlp.std(),
                    f"adaln/layer{i}/scale_mlp_mean": scale_mlp.mean(),
                    f"adaln/layer{i}/scale_mlp_std": scale_mlp.std(),
                    f"adaln/layer{i}/gate_mlp_mean": gate_mlp.mean(),
                    f"adaln/layer{i}/gate_mlp_std": gate_mlp.std(),
                    # overall norm
                    f"adaln/layer{i}/mod_norm": mod.norm(dim=-1).mean(),
                }
            )

        self.log_dict(adaln_dict, on_step=True, sync_dist=True)

    return output


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(
        train_set, **cfg.loader, shuffle=True, drop_last=True, generator=rnd_gen
    )
    val = torch.utils.data.DataLoader(
        val_set, **cfg.loader, shuffle=False, drop_last=False
    )

    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    idm = InverseDynamicsModel(
        emb_dim=embed_dim,
        action_dim=effective_act_dim,
        **cfg.idm,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
        idm=idm,
    )

    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    data_choice = HydraConfig.get().runtime.choices.data
    run_dir = Path(
        swm.data.utils.get_cache_dir(),
        f"{data_choice}_{cfg.output_model_name}",
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir,
        filename=cfg.output_model_name,
        epoch_interval=1,
        stop_epoch=cfg.get("stop_epoch"),
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
