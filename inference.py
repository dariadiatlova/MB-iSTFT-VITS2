import langdetect
import torch
import wandb
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols

"""
from phonemizer.backend.espeak.wrapper import EspeakWrapper
_ESPEAK_LIBRARY = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
EspeakWrapper.set_library(_ESPEAK_LIBRARY)
"""

# - paths
path_to_config = "/app/configs/mb_istft_vits2_base.json"  # path to .json
path_to_model = "/app/data/train_log/G_312000.pth"  # path to G_xxxx.pth

# check device
if torch.cuda.is_available() is True:
    device = "cuda:6"
else:
    device = "cpu"

hps = utils.get_hparams_from_file(path_to_config)

if (
    "use_mel_posterior_encoder" in hps.model.keys()
    and hps.model.use_mel_posterior_encoder == True
):
    print("Using mel posterior encoder for VITS2")
    posterior_channels = 80  # vits2
    hps.data.use_mel_posterior_encoder = True
else:
    print("Using lin posterior encoder for VITS1")
    posterior_channels = hps.data.filter_length // 2 + 1
    hps.data.use_mel_posterior_encoder = False

net_g = SynthesizerTrn(
    len(symbols),
    posterior_channels,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,  # - >0 for multi speaker
    **hps.model,
).to(device)
_ = net_g.eval()

_ = utils.load_checkpoint(path_to_model, net_g, None)


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def langdetector(text):  # from PolyLangVITS
    try:
        lang = langdetect.detect(text).lower()
        if lang == "ko":
            return f"[KO]{text}[KO]"
        elif lang == "ja":
            return f"[JA]{text}[JA]"
        elif lang == "en":
            return f"[EN]{text}[EN]"
        elif lang == "zh-cn":
            return f"[ZH]{text}[ZH]"
        else:
            return text
    except Exception as e:
        return text


@torch.no_grad()
def vctk_inference(generator):
    print(f"Loaded speakers: {generator.n_speakers}")
    collate_fn = TextAudioSpeakerCollate()
    test_dataset = TextAudioSpeakerLoader(hps.data.test_files, hps.data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=0,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
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
        ) in tqdm(enumerate(test_loader), total=len(test_loader), leave=True, desc="Test loader..."):
            x, x_lengths = x.to(device), x_lengths.to(device)
            speaker_ids = speaker_ids.to(device)

            y_hat, y_hat_mb, attn, mask, *_ = generator.infer(
                x, x_lengths, max_len=512, sid=speaker_ids
            )
            y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

            for n in range(y.shape[0]):
                speaker_id = speaker_ids.detach().cpu()[n]
                original_audio_dict.update(
                    {
                        f"test_audio_original/speaker_{speaker_id}_batch_{batch_idx}_sample_{n}": wandb.Audio(
                            y[n, :, : y_lengths[n]].squeeze(0).detach().cpu().numpy(),
                            sample_rate=hps.data.sampling_rate,
                        )
                    },
                )

            for n in range(y_hat.shape[0]):
                speaker_id = speaker_ids.detach().cpu()[n]
                generated_audio_dict.update(
                    {
                        f"test_audio_generated/speaker_{speaker_id}_batch_{batch_idx}_sample_{n}": wandb.Audio(
                            y_hat[n, :, : y_hat_lengths[n]]
                            .squeeze(0)
                            .detach()
                            .cpu()
                            .numpy(),
                            sample_rate=hps.data.sampling_rate,
                        )
                    },
                )
    wandb.log(original_audio_dict)
    wandb.log(generated_audio_dict)


wandb.init(project=hps.wandb.project, resume=False, id=None)
vctk_inference(net_g)
