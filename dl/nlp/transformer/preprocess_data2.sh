!git clone https://furiousavocados19:password1234@bitbucket.org/furiousavocados19/wmt-14-preprocessed-data.git
!cd wmt-14-preprocessed-data/ && tar -zxvf wmt14-preprocessed.tar.gz
!mv wmt-14-preprocessed-data/wmt14-preprocessed/ .
!rm -rf wmt-14-preprocessed-data/
!rm -rf wmt14-preprocessed/sentencepiece/

import numpy as np

with open("wmt14-preprocessed/train.en.ids", "r") as file_obj:
    english = np.array([
        np.array(
                line.split()[:max_sequence_length],
        dtype="uint16")
        for line in file_obj.read().splitlines()
    ])
with open("wmt14-preprocessed/train.de.ids", "r") as file_obj:
    german = np.array([
        np.array(
                line.split()[:max_sequence_length],
        dtype="uint16")
        for line in file_obj.read().splitlines()
    ])
np.random.seed(42)
p = np.random.permutation(len(german))
german = german[p]
english = english[p]

!rm -rf wmt14-preprocessed/
!mkdir attention-train-input

def save_train_data(filepath, data):
    with open(filepath, "w") as file_obj:
        for line in data:
            file_obj.write(" ".join(line.astype('str')) + "\n")

save_train_data("attention-train-input/train.en", english)
save_train_data("attention-train-input/train.de", german)
np.savetxt("attention-train-input/train.permutation", p)

# Push to remote repo
!cd attention-train-input && \
    git init && \
    git remote add origin https://furiousavocados19:password1234@bitbucket.org/furiousavocados19/transformer-train-input.git && \
    git config --global user.email "michael_liu2@yahoo.com" && \
    git config --global user.name "Michael Liu" && \
    git add -A && \
    git commit -m 'First commit' && \
    git push -f origin master;
