import pandas as pd
import os
from joblib import Parallel, delayed
from tqdm import trange
import torch

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

tr_model_path = None
tr_config_path = None
tr_vocoder_path = None
tr_vocoder_config_path = None
ts_model_path = None
ts_config_path = None
ts_vocoder_path = None
ts_vocoder_config_path = None
speakers_file_path = None
language_ids_file_path = None
encoder_path = None
encoder_config_path = None

## EDIT THE LINE BELOW
python_bin_path = "/root/miniconda3/envs/temp_env/bin/python"


json_path = "../../lib/python3.8/site-packages/TTS/.models.json"
path = os.path.join(python_bin_path, json_path)
manager = ModelManager(path)


tr_model_name = "tts_models/en/vctk/vits"
ts_model_name = "tts_models/en/vctk/fast_pitch"

tr_model_path, tr_config_path, tr_model_item = manager.download_model(tr_model_name)
ts_model_path, ts_config_path, ts_model_item = manager.download_model(ts_model_name)
tr_vocoder_name = tr_model_item["default_vocoder"]
ts_vocoder_name = ts_model_item["default_vocoder"]

if tr_vocoder_name is not None:
    tr_vocoder_path, tr_vocoder_config_path, _ = manager.download_model(tr_vocoder_name)
if ts_vocoder_name is not None:
    ts_vocoder_path, ts_vocoder_config_path, _ = manager.download_model(ts_vocoder_name)




@torch.inference_mode()
def get_audio(text, speaker_id, out_path, split, model):
    wav = model.tts(
        text,
        speaker_id,
        "en",
        None,
        reference_wav=None,
        style_wav=None,
        style_text=None,
        reference_speaker_name=None,
    )
    model.save_wav(wav, out_path)

def process_row(i, dataset, prompt, sequence, speaker, out_path_prompt, out_path_seq, split, model):
    get_audio(prompt, speaker, out_path_prompt, split, model)
    get_audio(sequence, speaker, out_path_seq, split, model)

def inference(dataset):
    tr_synthesizer = Synthesizer(
        tr_model_path,
        tr_config_path,
        speakers_file_path,
        language_ids_file_path,
        tr_vocoder_path,
        tr_vocoder_config_path,
        encoder_path,
        encoder_config_path,
        True,
    )

    df = pd.read_csv(f"./final_splits/{dataset}/train.csv")
    os.makedirs(f"./dump/{dataset}/", exist_ok=True)    

    Parallel(n_jobs=6, prefer="threads")(
        delayed(process_row)(
            i, 
            dataset, 
            df["Denormalized Prompt"][i],
            df["Denormalized Sequence"][i],
            df["Speaker ID"][i],
            df["prompt_path_audio"][i],
            df["seq_path_audio"][i],
            df["split"][i],
            tr_synthesizer
        )
        for i in trange(len(df))
    )
    
    ts_synthesizer = Synthesizer(
        ts_model_path,
        ts_config_path,
        speakers_file_path,
        language_ids_file_path,
        ts_vocoder_path,
        ts_vocoder_config_path,
        encoder_path,
        encoder_config_path,
        True,
    )

    df = pd.read_csv(f"./final_splits/{dataset}/test.csv")

    Parallel(n_jobs=6, prefer="threads")(
        delayed(process_row)(
            i, 
            dataset, 
            df["Denormalized Prompt"][i],
            df["Denormalized Sequence"][i],
            df["Speaker ID"][i],
            df["prompt_path_audio"][i],
            df["seq_path_audio"][i],
            df["split"][i],
            ts_synthesizer
        )
        for i in trange(len(df))
    )
inference("numeric")
inference("alphanumeric")
# inference("alpha")
