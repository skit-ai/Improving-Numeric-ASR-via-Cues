speakers = {'p225': 1, 'p226': 2, 'p227': 3, 'p228': 4, 'p229': 5, 'p230': 6, 'p231': 7, 'p232': 8, 'p233': 9, 'p234': 10, 'p236': 11, 'p237': 12, 'p238': 13, 'p239': 14, 'p240': 15, 'p241': 16, 'p243': 17, 'p244': 18, 'p245': 19, 'p246': 20, 'p247': 21, 'p248': 22, 'p249': 23, 'p250': 24, 'p251': 25, 'p252': 26, 'p253': 27, 'p254': 28, 'p255': 29, 'p256': 30, 'p257': 31, 'p258': 32, 'p259': 33, 'p260': 34, 'p261': 35, 'p262': 36, 'p263': 37, 'p264': 38, 'p265': 39, 'p266': 40, 'p267': 41, 'p268': 42, 'p269': 43, 'p270': 44, 'p271': 45, 'p272': 46, 'p273': 47, 'p274': 48, 'p275': 49, 'p276': 50, 'p277': 51, 'p278': 52, 'p279': 53, 'p280': 54, 'p281': 55, 'p282': 56, 'p283': 57, 'p284': 58, 'p285': 59, 'p286': 60, 'p287': 61, 'p288': 62, 'p292': 63, 'p293': 64, 'p294': 65, 'p295': 66, 'p297': 67, 'p298': 68, 'p299': 69, 'p300': 70, 'p301': 71, 'p302': 72, 'p303': 73, 'p304': 74, 'p305': 75, 'p306': 76, 'p307': 77, 'p308': 78, 'p310': 79, 'p311': 80, 'p312': 81, 'p313': 82, 'p314': 83, 'p316': 84, 'p317': 85, 'p318': 86, 'p323': 87, 'p326': 88, 'p329': 89, 'p330': 90, 'p333': 91, 'p334': 92, 'p335': 93, 'p336': 94, 'p339': 95, 'p340': 96, 'p341': 97, 'p343': 98, 'p345': 99, 'p347': 100, 'p351': 101, 'p360': 102, 'p361': 103, 'p362': 104, 'p363': 105, 'p364': 106, 'p374': 107, 'p376': 108}
speakers = [k for k in speakers]
speakers_train = speakers[len(speakers) // 5:]
speakers_test = [f"VCTK_{item}" for item in speakers[:len(speakers)//5]]
# speakers_test = [item for item in speakers[:len(speakers)//5]]


import random
import pronunciation
import os
import numpy as np
import pandas as pd
random.seed(123)

seq_len = 6
num_examples = 10000

num_prompts = 100

letters = [c for c in 'abcdefghijklmnopqrstuvwxyz']
numbers = [c for c in "0123456789"]

def gen_seq(alpha, numeric, seq_length):
    if alpha and numeric:
        sample_set = letters + numbers
    elif alpha and not numeric:
        sample_set = letters
    elif numeric and not alpha:
        sample_set = numbers
    return "".join(np.random.choice(sample_set, size=seq_length))

def denormalize_seqs(prompts, seqs, dataset):
    denormed = []
    denormed_prompt = []
    p_types = []
    objs = pronunciation.get_pron_objs(dataset, len(prompts))
    for p, s, o in zip(prompts, seqs, objs):
        p_obj = o
        p_type = o.__class__.__name__

        denormed.append(p_obj.denormalize(s))
        denormed_prompt.append(p_obj.denormalize(p))
        p_types.append(p_type)
    return p_types, denormed, denormed_prompt

an_prompts = [
    "AA00",
]

an_prompts = [item.lower() for item in an_prompts]
temp = []
for _ in range(num_prompts-len(an_prompts)):
    a, n = random.choice([(True, True), (True, False), (False, True)])
    temp.append(gen_seq(a, n, random.choice([2, 3, 4])))
an_prompts = an_prompts + temp

n_prompts = []
for _ in range(num_prompts):
    n_prompts.append(gen_seq(False, True, random.choice([2, 3, 4])))

a_prompts = []
for _ in range(num_prompts):
    a_prompts.append(gen_seq(True, False, random.choice([2, 3, 4])))

numeric_seqs = [gen_seq(False, True, seq_len) for _ in range(num_examples)]
alpha_seqs = [gen_seq(True, False, seq_len) for _ in range(num_examples)]
alphanumeric_seqs = [gen_seq(True, True, seq_len) for _ in range(num_examples)]

os.makedirs("./dump", exist_ok=True)

#dump numeric sequences
speaker_ids = np.random.choice(speakers_train+speakers_test, size=num_examples)
prompts_ = np.random.choice(n_prompts, size=num_examples)
p_types, denorm, prompts_denorm = denormalize_seqs(prompts_, numeric_seqs, "numeric")
df = pd.DataFrame(
    zip(
        prompts_,
        prompts_denorm,
        numeric_seqs,
        denorm,
        p_types,
        speaker_ids
    ),
    columns=["Prompt", "Denormalized Prompt", "Normalized Sequence", "Denormalized Sequence", "Pronunication Class", "Speaker ID"]
)
df["split"] = df["Speaker ID"].apply(lambda x: "test" if x in speakers_test else "train")
print(df)
df.to_csv("./dump/numeric_seqs.csv", index=False)

#dump alpha sequences
speaker_ids = np.random.choice(speakers_train+speakers_test, size=num_examples)
prompts_ = np.random.choice(a_prompts, size=num_examples)
p_types, denorm, prompts_denorm = denormalize_seqs(prompts_, alpha_seqs, "alpha")
df = pd.DataFrame(
    zip(
        prompts_,
        prompts_denorm,
        alpha_seqs,
        denorm,
        p_types,
        speaker_ids
    ),
    columns=["Prompt", "Denormalized Prompt", "Normalized Sequence", "Denormalized Sequence", "Pronunication Class", "Speaker ID"]
)
df["split"] = df["Speaker ID"].apply(lambda x: "test" if x in speakers_test else "train")
print(df)
df.to_csv("./dump/alpha_seqs.csv", index=False)

# dump alphanumeric sequences
speaker_ids = np.random.choice(speakers_train+speakers_test, size=num_examples)
prompts_ = np.random.choice(an_prompts, size=num_examples)
p_types, denorm, prompts_denorm = denormalize_seqs(prompts_, alphanumeric_seqs, "alphanumeric")
df = pd.DataFrame(
    zip(
        prompts_,
        prompts_denorm,
        alphanumeric_seqs,
        denorm,
        p_types,
        speaker_ids
    ),
    columns=["Prompt", "Denormalized Prompt", "Normalized Sequence", "Denormalized Sequence", "Pronunication Class", "Speaker ID"]
)
df["split"] = df["Speaker ID"].apply(lambda x: "test" if x in speakers_test else "train")
print(df)
df.to_csv("./dump/alphanumeric_seqs.csv", index=False)
