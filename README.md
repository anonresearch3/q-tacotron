# q-Tacotron

PyTorch implementation of [Q-Tacotron: Improving Text-to-Speech by Conditioning on High-Level Hand-Crafted Features]().  

The approach use hand-crafted observed features at a word level (q-features) which allow to make a controllable text-to-speech.
At inference time, we incorporate language information into the q-Tacotron using BERT-like model.

Audio samples using our published [q-Tacotron]() and [HiFi-GAN](https://github.com/jik876/hifi-gan) models.

## Setup
1. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/)
2. Clone this repo: `git clone `
3. CD into this repo: `cd qtacotron`
4. Initialize submodule: `git submodule init; git submodule update`
5. `docker pull nvcr.io/nvidia/nemo:22.07`
6. `docker run -itd --ipc=host -v /home:/home --restart unless-stopped --name nemo_nemo_20_07 --shm-size=16g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/nemo:22.07`
7. `pip install transformers pyreaper NumPy==1.20 --user`

## Preprocess
1. Run `sh ./extract_ljspeech_phonemes_and_durs.sh` to extracts the phonemes and alignments for the LJSpeech dataset via the Montreal Forced Aligner (MFA) library, and computes durations per phoneme. It created based on [NeMo script]( https://github.com/NVIDIA/NeMo/blob/583aa6adf5ba8f20363687f479f6e2a8f3840c91/scripts/dataset_processing/ljspeech/extract_ljspeech_phonemes_and_durs.sh).
2. Run `python ./preprocess.py` 

## Training
1. `python train.py --output_directory=outdir`

## Inference demo
See [Colab](qtacotron.ipynb) or
[cli](inference.py)

## Related repos
[HiFi-GAN](https://github.com/jik876/hifi-gan) 

## Acknowledgements
This implementation based on NVIDIA [Tacotron 2](https://github.com/NVIDIA/tacotron2) repository