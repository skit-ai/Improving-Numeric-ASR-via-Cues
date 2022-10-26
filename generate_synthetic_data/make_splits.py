import pandas as pd
import os
from sklearn.model_selection import train_test_split

os.makedirs("./final_splits/", exist_ok=True)

def get_feature_paths(dataset, i):
    return os.path.join(f"dump/{dataset}_features/{i}_prompt.pth"), os.path.join(f"dump/{dataset}_features/{i}_seq.pth")

def get_audio_paths(dataset, i):
    return os.path.join(f"dump/{dataset}/{i}_prompt.wav"), os.path.join(f"dump/{dataset}/{i}_seq.wav")

def make_splits(dataset):
    os.makedirs(f"final_splits/{dataset}", exist_ok=True)

    df = pd.read_csv(f"./dump/{dataset}_seqs.csv", dtype={'Prompt': str, "Normalized Sequence": str})
    ppaths, spaths = [], []
    ppaths_audio, spaths_audio = [], []
    for i in range(len(df)):
        p, s = get_feature_paths(dataset, i)
        spaths.append(s)
        ppaths.append(p)
        p, s = get_audio_paths(dataset, i)
        spaths_audio.append(s)
        ppaths_audio.append(p)
        
    df["prompt_path"] = ppaths
    df["seq_path"] = spaths
    df["prompt_path_audio"] = ppaths_audio
    df["seq_path_audio"] = spaths_audio

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    train_df.to_csv(f"./final_splits/{dataset}/train.csv", index=False)    
    test_df.to_csv(f"./final_splits/{dataset}/test.csv", index=False)    

    print(dataset)
    print("train len", len(train_df))
    print("test len", len(test_df))
    
make_splits("numeric")
make_splits("alphanumeric")
# make_splits("alpha")