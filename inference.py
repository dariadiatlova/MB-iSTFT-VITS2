import re

import langdetect
# import matplotlib.pyplot as plt
import torch
import wandb
from scipy.io.wavfile import write
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import (TextAudioCollate, TextAudioSpeakerCollate, TextAudioSpeakerLoader)
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
path_to_model = "/app/data/train_logs/G_62000.pth"  # path to G_xxxx.pth

# - text input
input = "I try to get the waiter's attention by blinking in morse code"

# check device
if torch.cuda.is_available() is True:
    device = "cuda:5"
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


# speed = 1
# sid = 0
# output_dir = "output"
# os.makedirs(output_dir, exist_ok=True)
# speakers = [name for sid, name in enumerate(hps.speakers) if name != "None"]


# def vcss(inputstr):  # single
#     fltstr = re.sub(r"[\[\]\(\)\{\}]", "", inputstr)
#     # fltstr = langdetector(fltstr) #- optional for cjke/cjks type cleaners
#     stn_tst = get_text(fltstr, hps)
#
#     with torch.no_grad():
#         x_tst = stn_tst.to(device).unsqueeze(0)
#         x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
#         audio = (
#             net_g.infer(
#                 x_tst,
#                 x_tst_lengths,
#                 noise_scale=0.667,
#                 noise_scale_w=0.8,
#                 length_scale=1 / speed,
#             )[0][0, 0]
#             .data.cpu()
#             .float()
#             .numpy()
#         )
#     write(f"./{output_dir}/output_{sid}.wav", hps.data.sampling_rate, audio)
#     print(f"./{output_dir}/output_{sid}.wav Generated!")


# def vcms(inputstr, sid):  # multi
#     fltstr = re.sub(r"[\[\]\(\)\{\}]", "", inputstr)
#     # fltstr = langdetector(fltstr) #- optional for cjke/cjks type cleaners
#     stn_tst = get_text(fltstr, hps)
#
#     for idx, speaker in enumerate(speakers):
#         sid = torch.LongTensor([idx]).to(device)
#         with torch.no_grad():
#             x_tst = stn_tst.to(device).unsqueeze(0)
#             x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
#             audio = (
#                 net_g.infer(
#                     x_tst,
#                     x_tst_lengths,
#                     sid=sid,
#                     noise_scale=0.667,
#                     noise_scale_w=0.8,
#                     length_scale=1 / speed,
#                 )[0][0, 0]
#                 .data.cpu()
#                 .float()
#                 .numpy()
#             )
#         write(f"{output_dir}/{speaker}.wav", hps.data.sampling_rate, audio)
#         print(f"{output_dir}/{speaker}.wav Generated!")


def ex_voice_conversion(sid_tgt):  # dummy - TODO : further work
    # import IPython.display as ipd
    output_dir = "ex_output"
    dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    loader = DataLoader(
        dataset,
        num_workers=0,
        shuffle=False,
        batch_size=1,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    data_list = list(loader)
    # print(data_list)

    with torch.no_grad():
        x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [
            x.to(device) for x in data_list[0]
        ]
        """
        sid_tgt1 = torch.LongTensor([1]).to(device)
        sid_tgt2 = torch.LongTensor([2]).to(device)
        sid_tgt3 = torch.LongTensor([4]).to(device)
        """
        audio = (
            net_g.voice_conversion(
                spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        """
        audio1 = net_g.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[0][0, 0].data.cpu().float().numpy()
        audio2 = net_g.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt2)[0][0, 0].data.cpu().float().numpy()
        audio3 = net_g.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt3)[0][0, 0].data.cpu().float().numpy()
        """

    write(
        f"./{output_dir}/output_{sid_src}-{sid_tgt}.wav", hps.data.sampling_rate, audio
    )
    print(f"./{output_dir}/output_{sid_src}-{sid_tgt}.wav Generated!")

    """
    print("Original SID: %d" % sid_src.item())
    ipd.display(ipd.Audio(y[0].cpu().numpy(), rate=hps.data.sampling_rate, normalize=False))
    print("Converted SID: %d" % sid_tgt1.item())
    ipd.display(ipd.Audio(audio1, rate=hps.data.sampling_rate, normalize=False))
    print("Converted SID: %d" % sid_tgt2.item())
    ipd.display(ipd.Audio(audio2, rate=hps.data.sampling_rate, normalize=False))
    print("Converted SID: %d" % sid_tgt3.item())
    ipd.display(ipd.Audio(audio3, rate=hps.data.sampling_rate, normalize=False))
    """


@torch.no_grad()
def vctk_inference(generator):
    print(f"Loaded speakers: {generator.n_speakers}")
    collate_fn = TextAudioCollate(return_ids=True)
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
        ) in enumerate(test_loader):
            x, x_lengths = x.to(device), x_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)
            speaker_ids = speaker_ids.to(device)

            y_hat, y_hat_mb, attn, mask, *_ = generator.infer(
                x, x_lengths, max_len=128, sid=speaker_ids
            )
            y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length

            y_hat, y_hat_mb, attn, mask, *_ = generator.infer(
                x, x_lengths, max_len=128, sid=speaker_ids
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


# vcss(input)
wandb.init(project=hps.wandb.project, resume=False, id=None)
vctk_inference(net_g)
