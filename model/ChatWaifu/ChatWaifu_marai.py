import hashlib
import logging
import os
import random
import re
import sys

from scipy.io.wavfile import write  # wav
from torch import no_grad, LongTensor

import config
import plugins.ChatWaifu.config as waifu_config
from model.ChatWaifu import commons
from model.ChatWaifu import utils
from model.ChatWaifu.mel_processing import spectrogram_torch
from model.ChatWaifu.models import SynthesizerTrn
from model.ChatWaifu.text import text_to_sequence

####################################
# CHATGPT INITIALIZE

idmessage = ""

# 读取配置文件中设置的语言模型
try:
    speakerID = waifu_config.waifu_voice  # 语音ID，对应上方的角色
    if speakerID > 4:
        speakerID = 4
        logging.warning("不存在您设置的语言模型,将自动切换到随机的语言模型")
except Exception:
    try:
        speakerID = config.waifu_voice
    except Exception:
        logging.error("错误：未定义waifu_voice选项")
        speakerID = 4


def get_input():
    # prompt for input
    print("You:")
    user_input = input()
    return user_input


def get_token():
    """向用户获取ChatGPT的cookie下的token"""
    token = 'eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..5aX47D4DxuM_sHIR.XNcghwU89T8vwGXiAq0bYa_SexoHQMcB7SOnhrBRa7mZm3KfiDD2dAZwAT5TGu79onjA42mzCoQvCaJ016G9Z5CG1yXZdpXeJsSwg4PvEJ1EFuBVvxR4VB9FfmbmIMqNt7QnMtQfPc2hupN3ZnXtl9WFYkbiNHcU2Nz-lmmvdh__ltu3CANLylRA2d6fL5vsyK1YBm94996bH9b9byhvHWnPZy3uPTxhhLsSPXMoZCmJ_JIlLfkI1HppHhH0wOn0xIp1cXPPBIbLjNj1bYgUKRyzUdkgH27BETaYDi5_Z_9I4o7FhPhfLygurygvu3nAJqIQ2nsDWHghdtJG7gWN-k87qHHyjC5ygO4-wUmdsDyDfC3A5G1js8YYSrYEUXBMwTbMBN4wrqQKrorPe8U71W9xkhdhg5-3FblHbwtEPEwPYGi7wH-s-N9dAWkh0tQLSKz3x6Mb8YiIl3kXyBH3022SYe7n1ms7F8P_lt5SLp1pKqePX9Doe2TvC8kFdiXu4SYUI-lerDkX4abR_g_NSvhTrHmaXc7SnmXsAiz8ZayMTbK5ox5Weqoa6OSm2i6SuUboA-qLC-iYLZGXPILncE06FS1tN5gAx5fTX8dM2St-Xfm3soh-PiihxZDh69UM7opodrdOGDCYo6LFF59_teosDvtH8ziCgtrvP3BtZ3J8l-qV2OMITj1CMTTJZtgs46CoWkinJHGPNcktSsrti1sYfVtkan6kNPIAsRrY7bzliIHJ4BdlLr_dRAFcEYqhUb9aBw7H99asCcXszG0647jU9NZkWZFtLdM5KI5Ryz6bn4yy59v0eTLOHg9o0p8Z0M6PmqnHxKyytBUd2KFCAJla8P0lrQ0lEdxujug1Yj2zD52xIdrwYl02yQrIPtse5KWE7qvX92Q2pskOge8pZRdRmyuU5yMduiI3ydcdRPu72yGmYMVQ0MjR1FeTTbvlIeJBYhWVsX--yD3-yKDFP5FqwoU-mUjRi0fHrNB5_AOXJa-EQ2ExJ0PZUMa7q9ReDQiQpztZhXbu--4egh1Y5HrttKDiL5eFGc4it3Xb1ffwT3jTzZuby0YcnP8NGh2r19-m5Ff_j7-msis_5Y0epODpDFtEub8OPkrsILsCKl4OTls4heL4vH7D6zHnPLVLGR95plNfeIaq12QclGMcSnTk5tz8BxII5lYAAbcohGZ5A281Cy3FaRPFiyfkEPqrqrIBZSyB8c7WazSBu2KqjN7PGIP3KvAWEzNL3tae-w8aZFJiMkjmZ423lgOlibdTU7jdBcx8sVXL1MrG10lw8502Vy0sX_BfkqCMWuQl02QicR-I_8xz3lexgkP7UaKcB86s1zg_-0PTRjoKAmPoQ3t3KKcrrdPeUVlq8SBz8zH3gSGuDxJ3Dm1uNhVhya-kuYADnkn4Hk7gAYOBSrNnB8LfUARPPOtN_7X_b7Hotz3gZWIzHcr5G2Kx-vScgscQoCRrODtduRMNqLDoP1tIhVJvL3H-X0bNxStAgi1S7VscaIcDJxqeOyM0rGQ5ZyPn0bmK9YQgN_isxNktptsZtQ6Fiqvh89mGgSllFG_2xJ_ZsjvKtO1SWkrJFnNeyatyOZfS-NHz6dbVb9-dw51OHjoGHfk0dH36QkilrfNKQmMSWyp64V5miHcHY9Key5z1Z-XZvSz0k9KSlboYgJom6Ht78TnrbKpgypzDQzLxIAn7R5GE13ZwHsia3eAmVnTE_u_v96_l7ZKutuTk34XYSRUDzJqyCGAB1P0MwpRpTfnmiC9yumhP8RRmD4W1UikQwdsXwZ7yz4FHH9LuJ655dAjg4ep1XPUIU_E-Ti8LJBG3O6OSpxdS_UBaSFtejjZBcVOGPjTgEhi3yMJSNVuvEgqXTUN_z9Y9JofyNfCtdzerMv01W-t7WfR5E5cUDNMmHTJKFvv0EEYJ7L0PPfJVpuxX5BJj1tqZp-0ZJQ3-6Ug3SgtG0uvKsUqIgoiFNg6pMW08wOhf3dF6HWVCYXyzZDWn25Jlv4cbwTKnIcUOzFowfInOlUN4dsuOT8LVAwDp0wjJcZnn5N0S2CZgsJPZpF9GXcYFkqN85VzEFcV2jqBrgHlFmXk_zD5BiVM0Exe9zQ8ImS38qcLxWtIOm8b2r9WVDqw0l8WiBrBDEdhjxjqmJ9iIWmCOY6Egh7Nb5rPa5y1eAJzFvXNPdcUSAW2vEmIR3RuKju3sOvRtwGmCWfE8W5FZ2AKtW3JdxznyjMc5PnjO_wk6Y_dYoSJ-zkxIQlAySpC5h52h36fkdlQuDa_lxob9GEMb03f_YLDc1mA9HbyV_pKXozvy2fjamj68L5KqEIYIUQGYzL4kB3_cYDnyhJlBLqz_TE-vGdJhz7mk7I5R.qOWRZKLDeYbTHfYDy-JHCg'
    return token


################################################


logging.getLogger('numba').setLevel(logging.WARNING)


def ex_print(text, escape=False):
    if escape:
        print(text.encode('unicode_escape').decode())
    else:
        print(text)


def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def ask_if_continue():
    while True:
        answer = input('Continue? (y/n): ')
        if answer == 'y':
            break
        elif answer == 'n':
            sys.exit(0)


def print_speakers(speakers, escape=False):
    if len(speakers) > 100:
        return
    print('ID\tSpeaker')
    for id, name in enumerate(speakers):
        ex_print(str(id) + '\t' + name, escape)


def get_speaker_id(message):
    speaker_id = input(message)
    try:
        speaker_id = int(speaker_id)
    except:
        print(str(speaker_id) + ' is not a valid ID!')
        sys.exit(1)
    return speaker_id


def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text


def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text


def generateSound(inputString, language, old_msg, sender_id):
    if '--escape' in sys.argv:
        escape = True
    else:
        escape = False

    # model = input('Path of a VITS model: ')
    # config = input('Path of a config file: ')
    import os
    this_file = os.getcwd()
    config = ''
    if language == "ch":
        model = this_file + r"\model\ChatWaifu\model\CN\model.pth"
        config = this_file + r"\model\ChatWaifu\model\CN\config.json"
    elif language == "jp":
        model = this_file + r"\model\ChatWaifu\model\H_excluded.pth"
        config = this_file + r"\model\ChatWaifu\model\config.json"
    elif language == "multi":
        model = this_file + r"\model\ChatWaifu\model\Multi\multi.pth"
        config = this_file + r"\model\ChatWaifu\model\Multi\config.json"

    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
    use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False
    emotion_embedding = hps_ms.data.emotion_embedding if 'emotion_embedding' in hps_ms.data.keys() else False

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        emotion_embedding=emotion_embedding,
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils.load_checkpoint(model, net_g_ms)

    def voice_conversion():
        audio_path = input('Path of an audio file to convert:\n')
        print_speakers(speakers)
        audio = utils.load_audio_to_torch(
            audio_path, hps_ms.data.sampling_rate)

        originnal_id = get_speaker_id('Original speaker ID: ')
        target_id = get_speaker_id('Target speaker ID: ')
        out_path = input('Path to save: ')

        y = audio.unsqueeze(0)

        spec = spectrogram_torch(y, hps_ms.data.filter_length,
                                 hps_ms.data.sampling_rate, hps_ms.data.hop_length, hps_ms.data.win_length,
                                 center=False)
        spec_lengths = LongTensor([spec.size(-1)])
        sid_src = LongTensor([originnal_id])

        with no_grad():
            sid_tgt = LongTensor([target_id])
            audio = net_g_ms.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[
                0][0, 0].data.cpu().float().numpy()
        return audio, out_path

    if n_symbols != 0:
        if not emotion_embedding:
            # while True:
            if (1 == 1):
                # choice = input('TTS or VC? (t/v):')
                choice = 't'
                if choice == 't':
                    # text = input('Text to read: ')
                    text = inputString
                    if text == '[ADVANCED]':
                        # text = input('Raw text:')
                        text = "我不会说"
                        # print('Cleaned text is:')
                        # ex_print(_clean_text(
                        #    text, hps_ms.data.text_cleaners), escape)
                        # continue

                    length_scale, text = get_label_value(
                        text, 'LENGTH', 1, 'length scale')
                    noise_scale, text = get_label_value(
                        text, 'NOISE', 0.667, 'noise scale')
                    noise_scale_w, text = get_label_value(
                        text, 'NOISEW', 0.8, 'deviation of noise')
                    cleaned, text = get_label(text, 'CLEANED')

                    stn_tst = get_text(text, hps_ms, cleaned=cleaned)

                    # print_speakers(speakers, escape)
                    # speaker_id = get_speaker_id('Speaker ID: ')
                    speaker_id = speakerID
                    if speaker_id == 4:  # 随机语音
                        speaker_id = random.randint(0, 3)
                    # out_path = input('Path to save: ')
                    # md5不重复语音命名
                    hash_key = hashlib.md5((str(sender_id) + old_msg).encode()).hexdigest()
                    out_filename = os.getcwd() + '\\voice-file\\voice_' + hash_key + '.wav'
                    logging.info(f"[ChatWaifu]: file name is: voice_{hash_key}.wav")
                    out_path = out_filename

                    with no_grad():
                        x_tst = stn_tst.unsqueeze(0)
                        x_tst_lengths = LongTensor([stn_tst.size(0)])
                        sid = LongTensor([speaker_id])
                        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                               noise_scale_w=noise_scale_w, length_scale=length_scale)[0][
                            0, 0].data.cpu().float().numpy()

                elif choice == 'v':
                    audio, out_path = voice_conversion()

                write(out_path, hps_ms.data.sampling_rate, audio)
                print('[model.ChatWaifu]Successfully saved!')
                # ask_if_continue()
        else:
            import os
            import librosa
            import numpy as np
            from torch import FloatTensor
            import audonnx
            w2v2_folder = input('Path of a w2v2 dimensional emotion model: ')
            w2v2_model = audonnx.load(os.path.dirname(w2v2_folder))
            # while True:
            if (1 == 1):
                # choice = input('TTS or VC? (t/v):')
                choice = 't'
                if choice == 't':
                    # text = input('Text to read: ')
                    text = inputString
                    if text == '[ADVANCED]':
                        # text = input('Raw text:')
                        text = "我不会说"
                        # print('Cleaned text is:')
                        # ex_print(_clean_text(
                        #    text, hps_ms.data.text_cleaners), escape)
                        # continue

                    length_scale, text = get_label_value(
                        text, 'LENGTH', 1, 'length scale')
                    noise_scale, text = get_label_value(
                        text, 'NOISE', 0.667, 'noise scale')
                    noise_scale_w, text = get_label_value(
                        text, 'NOISEW', 0.8, 'deviation of noise')
                    cleaned, text = get_label(text, 'CLEANED')

                    stn_tst = get_text(text, hps_ms, cleaned=cleaned)

                    # print_speakers(speakers, escape)
                    # speaker_id = get_speaker_id('Speaker ID: ')
                    speaker_id = speakerID
                    if speaker_id == 4:
                        speaker_id = random.randint(0, 3)

                    emotion_reference = input('Path of an emotion reference: ')
                    if emotion_reference.endswith('.npy'):
                        emotion = np.load(emotion_reference)
                        emotion = FloatTensor(emotion).unsqueeze(0)
                    else:
                        audio16000, sampling_rate = librosa.load(
                            emotion_reference, sr=16000, mono=True)
                        emotion = w2v2_model(audio16000, sampling_rate)[
                            'hidden_states']
                        emotion_reference = re.sub(
                            r'\..*$', '', emotion_reference)
                        np.save(emotion_reference, emotion.squeeze(0))
                        emotion = FloatTensor(emotion)

                    # out_path = input('Path to save: ')
                    # md5加密语音
                    hash_key = hashlib.md5((str(sender_id) + old_msg).encode()).hexdigest()
                    out_filename = os.getcwd() + '\\voice-file\\voice_' + hash_key + '.wav'
                    logging.info(f"[ChatWaifu]: file name is: voice_{hash_key}.wav")
                    out_path = out_filename

                    with no_grad():
                        x_tst = stn_tst.unsqueeze(0)
                        x_tst_lengths = LongTensor([stn_tst.size(0)])
                        sid = LongTensor([speaker_id])
                        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                               noise_scale_w=noise_scale_w,
                                               length_scale=length_scale, emotion_embedding=emotion)[0][
                            0, 0].data.cpu().float().numpy()

                elif choice == 'v':
                    audio, out_path = voice_conversion()

                write(out_path, hps_ms.data.sampling_rate, audio)
                print('Successfully saved!')
                # ask_if_continue()
    else:
        model = input('Path of a hubert-soft model: ')
        from hubert_model import hubert_soft
        hubert = hubert_soft(model)

        while True:
            audio_path = input('Path of an audio file to convert:\n')

            if audio_path != '[VC]':
                import librosa
                if use_f0:
                    audio, sampling_rate = librosa.load(
                        audio_path, sr=hps_ms.data.sampling_rate, mono=True)
                    audio16000 = librosa.resample(
                        audio, orig_sr=sampling_rate, target_sr=16000)
                else:
                    audio16000, sampling_rate = librosa.load(
                        audio_path, sr=16000, mono=True)

                # print_speakers(speakers, escape)
                target_id = get_speaker_id('Target speaker ID: ')
                out_path = input('Path to save: ')
                length_scale, out_path = get_label_value(
                    out_path, 'LENGTH', 1, 'length scale')
                noise_scale, out_path = get_label_value(
                    out_path, 'NOISE', 0.1, 'noise scale')
                noise_scale_w, out_path = get_label_value(
                    out_path, 'NOISEW', 0.1, 'deviation of noise')

                from torch import inference_mode, FloatTensor
                import numpy as np
                with inference_mode():
                    units = hubert.units(FloatTensor(audio16000).unsqueeze(
                        0).unsqueeze(0)).squeeze(0).numpy()
                    if use_f0:
                        f0_scale, out_path = get_label_value(
                            out_path, 'F0', 1, 'f0 scale')
                        f0 = librosa.pyin(audio, sr=sampling_rate,
                                          fmin=librosa.note_to_hz('C0'),
                                          fmax=librosa.note_to_hz('C7'),
                                          frame_length=1780)[0]
                        target_length = len(units[:, 0])
                        f0 = np.nan_to_num(np.interp(np.arange(0, len(f0) * target_length, len(f0)) / target_length,
                                                     np.arange(0, len(f0)), f0)) * f0_scale
                        units[:, 0] = f0 / 10

                stn_tst = FloatTensor(units)
                with no_grad():
                    x_tst = stn_tst.unsqueeze(0)
                    x_tst_lengths = LongTensor([stn_tst.size(0)])
                    sid = LongTensor([target_id])
                    audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                           noise_scale_w=noise_scale_w, length_scale=length_scale)[0][
                        0, 0].data.float().numpy()

            else:
                audio, out_path = voice_conversion()

            write(out_path, hps_ms.data.sampling_rate, audio)
            print('[model.ChatWaifu]Successfully saved!')
            # ask_if_continue()
