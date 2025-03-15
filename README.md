# Multi-label Musical Instrument Recognition Classification

Project the two of us ([huytruong1998](https://github.com/huytruong1998) ; [Bretan-Cezar](https://github.com/Bretan-Cezar)) 
worked on for a course on Advanced Audio Processing,
where we investigated the performance of Convolutional Neural Networks, 
given the task of tagging audio files containing fully-mixed musical pieces, based on 
the incorporated instruments.

The tags should be applied based on the instrument-specific 
sounds that were detected at any point within an audio clip.

Contains code for:
    - Extracting features from .wav files as h5py dumps before training;
    - The model training itself;
    - Inference and generating results in a reader-friendly format;
As well as documentation on the data used, results and observations.

Citations:

\[1\] Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley.
"Panns: Large-scale pretrained audio neural networks for audio pattern recognition." IEEE/ACM
Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.

\[2\] Eric J. Humphrey, Simon Durand, Brian McFee. OpenMIC-2018