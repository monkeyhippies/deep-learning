# Much of this was copied from https://github.com/OpenNMT/OpenNMT-tf

# Download 3 datasets
!wget --trust-server-names http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
!tar zxvf training-parallel-commoncrawl.tgz
!ls | grep -v 'commoncrawl.de-en.[de,en]' | xargs rm

!wget --trust-server-names http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
!tar zxvf training-parallel-europarl-v7.tgz
!cd training && ls | grep -v 'europarl-v7.de-en.[de,en]' | xargs rm
!rm training-parallel-europarl-v7.tgz

!wget --trust-server-names http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz
!tar zxvf training-parallel-nc-v11.tgz
!cd training-parallel-nc-v11 && ls | grep -v 'news-commentary-v11.de-en.[de,en]' | xargs rm
!rm training-parallel-nc-v11.tgz

# Concat into train.de and train.en files and remove extra files
!mv *.de training/
!mv *.en training/
!mv training-parallel-nc-v11/* training/
!rmdir training-parallel-nc-v11/

!for i in $(ls training/*.de); do cat $i >> train.de; done;
!for i in $(ls training/*.en); do cat $i >> train.en; done;

!rm -rf training/

# Install sentencepiece
!apt-get install -y cmake build-essential pkg-config libgoogle-perftools-dev
!git clone https://github.com/google/sentencepiece.git
!cd sentencepiece && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j $(nproc) && \
    make install && \
    ldconfig -v

# Train bpe encoding
!cat train.de >> train.txt
!cat train.en >> train.txt
!spm_train --input=train.txt \
    --model_prefix=wmt\
    --vocab_size=37000\
    --character_coverage=1.0\
    --model_type=bpe\
    --pad_id=0 --eos_id=1 --bos_id=2 --unk_id=3

!rm train.txt

#Encode training set
!spm_encode --model=wmt.model \
    --extra_options=eos \
    --output_format=id \
    < train.en > train.en.ids

!spm_encode --model=wmt.model \
    --extra_options=eos \
    --output_format=id \
    < train.de > train.de.ids

# Example decode
#!spm_decode --model=wmt.model \
#    --input_format=id \
#    < train.en.ids > train.en.ids.decoded

# tar and zip output
!mkdir ../wmt14-preprocessed
!mv * ../wmt14-preprocessed
!mv ../wmt14-preprocessed .
!rm -rf ../wmt14-preprocessed/sentencepiece
!tar -zcvf wmt14-preprocessed.tar.gz wmt14-preprocessed
!rm -rf wmt14-preprocessed

# Push to remote repo
!git init;
!git remote add origin https://furiousavocados19:password1234@bitbucket.org/furiousavocados19/wmt-14-preprocessed-data.git
!git config --global user.email "michael_liu2@yahoo.com"
!git config --global user.name "Michael Liu"
!git add -A;
!git commit -m 'First commit';
!git push -f origin master;
