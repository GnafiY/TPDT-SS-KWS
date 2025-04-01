import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torchaudio
import json
from pathlib import Path
from loguru import logger

# eps secures log and division
EPS = 1e-10
# Rate of the sources in LibriSpeech
RATE = 16000

src_data_dir = "your_path"

parser = argparse.ArgumentParser()
parser.add_argument('--target_num', type=int, required=True, default=100, 
                    help="Num of mixed wav file(s)")
# attention that target_num may be greater than len(os.listdir(librispeech_dir))
parser.add_argument('--hs_subset_name', type=str, required=True,
                    help="Hey-Snips metafile, choices: dev, test, train")
parser.add_argument('--ls_subset_name', type=str, required=True,
                    help="Librispeech metafile, choices: dev-clean, test-clean, train-clean-100, train-clean-360, train-other-500")
parser.add_argument('--ns_subset_name', type=str, required=True,
                    help="Wham metafile, choices: tr, cv, tt")
parser.add_argument('--output_dir', type=str, required=True,
                    help="root path of output csv file")
parser.add_argument('--snr', type=float, required=True,
                    help="SNR of spk1& spk2")
parser.add_argument('--modes', type=str, required=True, default="max min",
                    help="mix mode")

# json -> uid& path
def get_hey_snips_info(args_subset_name):
    logger.info(f"Getting informantion about `Hey-Snips` dataset, using `{args_subset_name}`.")
    hey_snips_base_dir = f"{src_data_dir}/hey_snips"
    data = []
    for file in os.listdir(hey_snips_base_dir):
        # if file.endswith(".json"):
        if file == f"{args_subset_name}.json":
            with open(os.path.join(hey_snips_base_dir, file)) as f:
                hey_snips_json_data = json.load(f)
                for wav_data in hey_snips_json_data:
                    audio_file_path, uid = wav_data['audio_file_path'], wav_data['id']
                    is_hotword = wav_data['is_hotword']
                    abs_path = os.path.join(hey_snips_base_dir, audio_file_path)
                    if is_hotword == 1:
                        data.append([uid, abs_path])
    logger.info("Informantion about `Hey-Snips` dataset collected.")
    return data

# csv -> uid& path
def get_librispeech_info(subset_name):
    logger.info(f"Getting informantion about `Librispeech` dataset, using `{subset_name}`.")
    data = []
    scp_dir = f"{src_data_dir}/metadata/LibriSpeech/scps/{subset_name}.scp"
    with open(scp_dir) as scp_file:
        for line in scp_file.readlines():
            data.append(line.split())
    logger.info("Informantion about `Librispeech` dataset collected.")
    return data

# wavs -> absolute path
def get_noise_info(subset_name):
    logger.info(f"Getting informantion about `Wham noise` dataset, using `{subset_name}`.")
    noise_dir = f"{src_data_dir}/wham_noise/{subset_name}"
    sick_file_list = ["421o0310_2.2765_050c010a_-2.2765.wav"]
    all_file_list = [os.path.join(noise_dir, noise_wav) for noise_wav in os.listdir(noise_dir)]
    for sick_file in sick_file_list:
        if sick_file in all_file_list:
            all_file_list.remove(sick_file)
    data = []
    for file in all_file_list:
        uid = file.split('/')[-1].split('.')[0]
        data.append([uid, file])
    logger.info("Informantion about `Wham noise` dataset collected.")
    return data

def get_mix_list(hs_info, ls_info, ns_info, target_num):
    hs_info_l, ls_info_l, ns_info_l = len(hs_info), len(ls_info), len(ns_info)

    logger.info("Getting mix index list.")
    mix_list = []
    produced = set()
    while mix_list.__len__() < target_num:
        hs_idx, ls_idx, ns_idx = np.random.randint(0, hs_info_l), np.random.randint(0, ls_info_l), np.random.randint(0, ns_info_l)
        
        while hs_idx == ls_idx:
            hs_idx, ls_idx = np.random.randint(0, hs_info_l), np.random.randint(0, ls_info_l)
        
        mix_idx = [hs_idx, ls_idx, ns_idx]
        uid = f"{hs_info[hs_idx][0]}_{ls_info[ls_idx][0]}"
        if uid in produced:
            continue
        else:
            mix_list.append(mix_idx)
            produced.add(uid)
    logger.info("Mix index list collected.")
    return mix_list


def main(args):
    modes = args.modes.split()
    target_num = args.target_num
    hs_info = get_hey_snips_info(args.hs_subset_name)
    ls_info = get_librispeech_info(args.ls_subset_name)
    ns_info = get_noise_info(args.ns_subset_name)
    
    mix_list = get_mix_list(ls_info, ls_info, ns_info, target_num)
    snr = float(args.snr)
    output_dir = args.output_dir

    # create_libri_hs_mix(hs_info, ls_info, ns_info, mix_list, output_dir, snr, modes)
    create_libri_hs_mix(ls_info, ls_info, ns_info, mix_list, output_dir, snr, modes)

def create_libri_hs_mix(hs_info, ls_info, ns_info, mix_list, output_dir, snr, modes):
    output_dir = Path(output_dir)
    for mode in modes:
        logger.info(f"Creating wav file at SNR `{snr}` with `{mode}` mode for `{str(output_dir).split('/')[-1]}`.")
        output_base_dir = output_dir / "wav16k" / mode
        for folder in ["mixed", "s1", "s2", "noise", "scps"]:
            if not os.path.exists(output_base_dir / folder):
                (output_base_dir / folder).mkdir(parents=True)

        scp_lines = []

        for mix_idx in mix_list:
            hs_idx, ls_idx, ns_idx = mix_idx[0], mix_idx[1], mix_idx[2]
            hs_wav, _ = torchaudio.load(hs_info[hs_idx][1])
            ls_wav, _ = torchaudio.load(ls_info[ls_idx][1])
            ns_wav, _ = torchaudio.load(ns_info[ns_idx][1])
            if ns_wav.shape[0] >= 2:
                ns_wav = ns_wav[0].unsqueeze(0)

            hs_wav, ls_wav, ns_wav = process_wav_length(hs_wav, ls_wav, ns_wav, mode)
            ls_alpha = calculate_librispeech_gain(ls_wav, ns_wav, 0)
            hs_alpha = calculate_librispeech_gain(hs_wav, ns_wav + ls_alpha * ls_wav, snr)

            mixed_file_id = f"{hs_info[hs_idx][0]}_{ls_info[ls_idx][0]}"
            scp_lines.append(mixed_file_id)

            mixed_wav = hs_alpha * hs_wav + ls_alpha * ls_wav + ns_wav

            torchaudio.save(output_base_dir / "mixed" / f"{mixed_file_id}.wav", mixed_wav, RATE)
            torchaudio.save(output_base_dir / "s1" / f"{mixed_file_id}.wav", hs_alpha * hs_wav, RATE)
            torchaudio.save(output_base_dir / "s2" / f"{mixed_file_id}.wav", ls_alpha * ls_wav, RATE)
            torchaudio.save(output_base_dir / "noise" / f"{mixed_file_id}.wav", ns_wav, RATE)
        
        
            with open(output_base_dir / "scps" / "spk1.scp", 'w') as spk1_scp:
                with open(output_base_dir / "scps" / "spk2.scp", 'w') as spk2_scp:
                    with open(output_base_dir / "scps" / "wav.scp", 'w') as wav_scp:
                        with open(output_base_dir / "scps" / "noise.scp", 'w') as noise_scp:
                            for scp_line in scp_lines:
                                wav_scp.write(f"{scp_line} {output_base_dir}/mixed/{scp_line}.wav\n")
                                spk1_scp.write(f"{scp_line} {output_base_dir}/s1/{scp_line}.wav\n")
                                spk2_scp.write(f"{scp_line} {output_base_dir}/s2/{scp_line}.wav\n")
                                noise_scp.write(f"{scp_line} {output_base_dir}/noise/{scp_line}.wav\n")

def process_wav_length(hs_wav: torch.Tensor, ls_wav: torch.Tensor, ns_wav: torch.Tensor, mode):
    # max -> fill
    if mode == 'max':
        ret_wav_length = max(hs_wav.shape[1], max(ls_wav.shape[1], ns_wav.shape[1]))
        pad_hs_wav = torch.cat((hs_wav, torch.zeros(1, ret_wav_length - hs_wav.shape[1])), 1)
        pad_ls_wav = torch.cat((ls_wav, torch.zeros(1, ret_wav_length - ls_wav.shape[1])), 1)
        pad_ns_wav = torch.cat((ns_wav, torch.zeros(1, ret_wav_length - ns_wav.shape[1])), 1)
        return pad_hs_wav, pad_ls_wav, pad_ns_wav
    # max -> fill
    elif mode == 'min':
        ret_wav_length = min(hs_wav.shape[1], min(ls_wav.shape[1], ns_wav.shape[1]))
        return hs_wav[:, :ret_wav_length], ls_wav[:, :ret_wav_length], ns_wav[:, :ret_wav_length]
    else:
        logger.error(f"Unexpected mixture mode `{mode}`!")
        import pdb; pdb.set_trace()

def si_snr_xy(x, y, eps=1e-8):
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    y_zm = y - torch.mean(y, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * y_zm, dim=-1,
        keepdim=True) * y_zm / (l2norm(y_zm, keepdim=True) ** 2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

def snr_xy(x: np.array, y: np.array):
    return 10 * np.log10(np.mean(x ** 2) / (np.mean(y ** 2) + EPS) + EPS)

def calculate_librispeech_gain(wav: torch.Tensor, ns_wav: torch.Tensor, t):
    l, r = EPS, 10000
    # wav, ns_wav = wav.numpy(), ns_wav.numpy()
    def f(alpha):
        mixed_wav = ns_wav + alpha * wav
        return si_snr_xy(alpha * wav, mixed_wav) - t
    while r - l >= EPS:
        mid = (l + r) / 2
        if f(mid) * f(l) > 0:
            l = mid
        else:
            r = mid
    return l

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
