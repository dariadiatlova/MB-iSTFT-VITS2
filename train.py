import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import (DistributedBucketSampler, TextAudioCollate,
                        TextAudioSpeakerLoader, TextAudioSpeakerCollate)
from losses import (discriminator_loss, feature_loss, generator_loss, kl_loss,
                    subband_stft_loss)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from models import (AVAILABLE_FLOW_TYPES, DurationDiscriminator,
                    MultiPeriodDiscriminator, SynthesizerTrn, DurationDiscriminator2)
from pqmf import PQMF
from text.symbols import symbols
from utils import find_free_port

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
global_step = 0
log_original_audio = True


# - base vits2 : Aug 29, 2023
def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    hps = utils.get_hparams()
    mp.set_start_method("spawn")
    n_gpus = torch.cuda.device_count()
    free_port = find_free_port()
    dist_url = f"tcp://127.0.0.1:{free_port}"
    mp.spawn(
        run,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
            dist_url,
        ),
    )


def run(rank, n_gpus, hps, dist_url):
    global global_step
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=n_gpus,
        rank=rank,
    )

    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        wandb.init(
            project=hps.wandb.project,
            dir=hps.wandb.dir,
            resume=hps.wandb.resume,
            id=hps.wandb.id,
        )

    torch.distributed.barrier()
    torch.cuda.set_device(rank)
    torch.manual_seed(hps.train.seed)

    if (
        "use_mel_posterior_encoder" in hps.model.keys()
        and hps.model.use_mel_posterior_encoder == True
    ):  # P.incoder for vits2
        print("Using mel posterior encoder for VITS2")
        posterior_channels = 80  # vits2
        hps.data.use_mel_posterior_encoder = True
    else:
        print("Using lin posterior encoder for VITS1")
        posterior_channels = hps.data.filter_length // 2 + 1
        hps.data.use_mel_posterior_encoder = False

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )

    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=32,
            num_workers=0,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
        )
    # some of these flags are not being used in the code and directly set in hps json file.
    # they are kept here for reference and prototyping.

    if (
        "use_transformer_flows" in hps.model.keys()
        and hps.model.use_transformer_flows == True
    ):
        use_transformer_flows = True
        transformer_flow_type = hps.model.transformer_flow_type
        print(f"Using transformer flows {transformer_flow_type} for VITS2")
        assert (
            transformer_flow_type in AVAILABLE_FLOW_TYPES
        ), f"transformer_flow_type must be one of {AVAILABLE_FLOW_TYPES}"
    else:
        print("Using normal flows for VITS1")
        use_transformer_flows = False

    if (
        "use_spk_conditioned_encoder" in hps.model.keys()
        and hps.model.use_spk_conditioned_encoder == True
    ):
        if hps.data.n_speakers == 0:
            use_spk_conditioned_encoder = False
            print("Warning: use_spk_conditioned_encoder is True but n_speakers is 0")
            print(
                "Setting use_spk_conditioned_encoder to False as model is a single speaker model"
            )
        use_spk_conditioned_encoder = True
    else:
        print("Using normal encoder for VITS1 (cuz it's single speaker after all)")
        use_spk_conditioned_encoder = False

    if (
        "use_noise_scaled_mas" in hps.model.keys()
        and hps.model.use_noise_scaled_mas == True
    ):
        print("Using noise scaled MAS for VITS2")
        use_noise_scaled_mas = True
        mas_noise_scale_initial = 0.01
        noise_scale_delta = 2e-6
    else:
        print("Using normal MAS for VITS1")
        use_noise_scaled_mas = False
        mas_noise_scale_initial = 0.0
        noise_scale_delta = 0.0

    if (
        "use_duration_discriminator" in hps.model.keys()
        and hps.model.use_duration_discriminator == True
    ):
        print("Using duration discriminator for VITS2")
        use_duration_discriminator = True
        if hps.model.duration_discriminator_type == "dur_disc_1":
            net_dur_disc = DurationDiscriminator(
                hps.model.hidden_channels,
                hps.model.hidden_channels,
                3,
                0.1,
                gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
            ).cuda(rank)
        elif hps.model.duration_discriminator_type == "dur_disc_2":
            net_dur_disc = DurationDiscriminator2(
                hps.model.hidden_channels,
                hps.model.hidden_channels,
                3,
                0.1,
                gin_channels=hps.model.gin_channels if hps.data.n_speakers != 0 else 0,
            ).cuda(rank)
        else:
            assert False, (f"Tried to use duration discriminator, but duration discriminator type "
                           f"{hps.model.duration_discriminator_type} is unknown!")
    else:
        print("NOT using any duration discriminator like VITS1")
        net_dur_disc = None
        use_duration_discriminator = False

    net_g = SynthesizerTrn(
        len(symbols),
        posterior_channels,
        hps.train.segment_size // hps.data.hop_length,
        mas_noise_scale_initial=mas_noise_scale_initial,
        noise_scale_delta=noise_scale_delta,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    if net_dur_disc is not None:
        optim_dur_disc = torch.optim.AdamW(
            net_dur_disc.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
    else:
        optim_dur_disc = None

    net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)

    if net_dur_disc is not None:  # 2의 경우
        net_dur_disc = DDP(net_dur_disc, device_ids=[rank], find_unused_parameters=True)

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g, loading_generator=True
        )  # utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        if net_dur_disc is not None:  # 2의 경우
            _, _, _, epoch_str = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
                net_dur_disc,
                optim_dur_disc,
            )
        previous_train_loader_len = 1277 # hard coded for finetuning case as finetune train loader is much smaller
        global_step = (epoch_str - 1) * previous_train_loader_len #len(train_loader)
        print(f"Successfully initialized from global step: {global_step}!")
    except Exception as e:
        print(e)
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )  # epoch_str - 2
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    if net_dur_disc is not None:  # 2의 경우
        scheduler_dur_disc = torch.optim.lr_scheduler.ExponentialLR(
            optim_dur_disc, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
    else:
        scheduler_dur_disc = None

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc],
                [optim_g, optim_d, optim_dur_disc],
                [scheduler_g, scheduler_d, scheduler_dur_disc],
                scaler,
                [train_loader, eval_loader],
                logger,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d, net_dur_disc],
                [optim_g, optim_d, optim_dur_disc],
                [scheduler_g, scheduler_d, scheduler_dur_disc],
                scaler,
                [train_loader, None],
                None,
            )
        scheduler_g.step()
        scheduler_d.step()
        if net_dur_disc is not None:
            scheduler_dur_disc.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger
):
    net_g, net_d, net_dur_disc = nets
    optim_g, optim_d, optim_dur_disc = optims
    train_loader, eval_loader = loaders

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    if net_dur_disc is not None:  # vits2
        net_dur_disc.train()

    if rank == 0:
        loader = tqdm.tqdm(train_loader, desc="Loading training data", leave=True, total=len(train_loader))
    else:
        loader = train_loader

    for batch_idx, (
        x,
        x_lengths,
        spec,
        spec_lengths,
        y,
        y_lengths,
        speaker_ids,
    ) in enumerate(loader):
        if net_g.module.use_noise_scaled_mas:
            current_mas_noise_scale = (
                net_g.module.mas_noise_scale_initial
                - net_g.module.noise_scale_delta * global_step
            )
            net_g.module.current_mas_noise_scale = max(current_mas_noise_scale, 0.0)
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(
            rank, non_blocking=True
        )
        spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(
            rank, non_blocking=True
        )
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(
            rank, non_blocking=True
        )
        speaker_ids = speaker_ids.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                y_hat_mb,
                l_length,
                attn,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
                (hidden_x, logw, logw_),
            ) = net_g(x, x_lengths, spec, spec_lengths, sid=speaker_ids)

            if (
                hps.model.use_mel_posterior_encoder
                or hps.data.use_mel_posterior_encoder
            ):
                mel = spec
            else:
                mel = spec_to_mel_torch(
                    spec,
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )

            y = commons.slice_segments(
                y, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
                loss_disc_all = loss_disc

            # Duration Discriminator
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(
                    hidden_x.detach(), x_mask.detach(), logw_.detach(), logw.detach()
                )  # logw is predicted duration, logw_ is real duration
                with autocast(enabled=False):
                    # TODO: I think need to mean using the mask, but for now, just mean all
                    loss_dur_disc, losses_dur_disc_r, losses_dur_disc_g = (
                        discriminator_loss(y_dur_hat_r, y_dur_hat_g)
                    )
                    loss_dur_disc_all = loss_dur_disc
                optim_dur_disc.zero_grad()
                scaler.scale(loss_dur_disc_all).backward()
                scaler.unscale_(optim_dur_disc)
                grad_norm_dur_disc = commons.clip_grad_value_(
                    net_dur_disc.parameters(), None
                )
                scaler.step(optim_dur_disc)

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            if net_dur_disc is not None:
                y_dur_hat_r, y_dur_hat_g = net_dur_disc(hidden_x, x_mask, logw_, logw)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)

                if hps.model.mb_istft_vits == True:
                    pqmf = PQMF(y.device)
                    y_mb = pqmf.analysis(y)
                    loss_subband = subband_stft_loss(hps, y_mb, y_hat_mb)
                else:
                    loss_subband = torch.tensor(0.0)

                loss_gen_all = (
                    loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_subband
                )
                if net_dur_disc is not None:
                    loss_dur_gen, losses_dur_gen = generator_loss(y_dur_hat_g)
                    loss_gen_all += loss_dur_gen

        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]

                losses = [
                    loss_disc,
                    loss_gen,
                    loss_fm,
                    loss_mel,
                    loss_dur,
                    loss_kl,
                    loss_subband,
                ]

                # logger.info(
                #     "Train Epoch: {} [{:.0f}%] [!n]".format(
                #         epoch, 100.0 * batch_idx / len(train_loader)
                #     )
                # )
                # logger.info([x.item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "train_loss/g_total": loss_gen_all.item(),
                    "train_loss/d_total": loss_disc_all.item(),
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }

                if net_dur_disc is not None:  # 2인 경우
                    scalar_dict.update(
                        {
                            "train_loss/dur_disc_total": loss_dur_disc_all.item(),
                            "grad_norm_dur_disc": grad_norm_dur_disc,
                        }
                    )
                scalar_dict.update(
                    {
                        "train_loss/g_fm": loss_fm.item(),
                        "train_loss/g_mel": loss_mel.item(),
                        "train_loss/g_dur": loss_dur.item(),
                        "train_loss/g_kl": loss_kl.item(),
                        "train_loss/g_subband": loss_subband.item(),
                    }
                )
                wandb.log(scalar_dict)

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader)
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
                )
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
                )
                if net_dur_disc is not None:
                    utils.save_checkpoint(
                        net_dur_disc,
                        optim_dur_disc,
                        hps.train.learning_rate,
                        epoch,
                        os.path.join(hps.model_dir, "DUR_{}.pth".format(global_step)),
                    )
        global_step += 1

    # if rank == 0:
    #     logger.info("====> Epoch: {}".format(epoch))


def evaluate(hps, generator, eval_loader):
    global log_original_audio
    generator.eval()
    original_audio_dict, generated_audio_dict = {}, {}
    with torch.no_grad():
        for batch_idx, (
            x,
            x_lengths,
            spec,
            spec_lengths,
            y,
            y_lengths,
            speaker_ids,
        ) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            y, y_lengths = y.cuda(0), y_lengths.cuda(0)
            speaker_ids = speaker_ids.cuda(0)

            y_hat, y_hat_mb, attn, mask, *_ = generator.module.infer(
                x, x_lengths, max_len=128, sid=speaker_ids
            )
            y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

        if log_original_audio:
            for n in range(y.shape[0]):
                speaker_id = speaker_ids.detach().cpu()[n]
                original_audio_dict.update(
                    {
                        f"original_audio/speaker_{speaker_id}_batch_{batch_idx}_sample_{n}": wandb.Audio(
                            y[n, :, : y_lengths[n]].squeeze(0).detach().cpu().numpy(),
                            sample_rate=hps.data.sampling_rate,
                        )
                    },
                )
        for n in range(y_hat.shape[0]):
            speaker_id = speaker_ids.detach().cpu()[n]
            generated_audio_dict.update(
                {
                    f"generated_audio/speaker_{speaker_id}_batch_{batch_idx}_sample_{n}": wandb.Audio(
                        y_hat[n, :, : y_hat_lengths[n]]
                        .squeeze(0)
                        .detach()
                        .cpu()
                        .numpy(),
                        sample_rate=hps.data.sampling_rate,
                    )
                },
            )
    if log_original_audio:
        wandb.log(original_audio_dict)
        log_original_audio = False
    wandb.log(generated_audio_dict)

    torch.cuda.empty_cache()
    generator.train()


if __name__ == "__main__":
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    main()
