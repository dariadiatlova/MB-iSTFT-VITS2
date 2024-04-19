# MB-iSTFT-VITS2

Based on MB-iSTFT-VITS implementation 


### Architecture

![Alt text](resources/image6.png)

[//]: # (A... [vits2_pytorch]&#40;https://github.com/p0p4k/vits2_pytorch&#41; and [MB-iSTFT-VITS]&#40;https://github.com/MasayaKawamura/MB-iSTFT-VITS&#41; hybrid... Gods, an abomination! Who created this atrocity?)

[//]: # ()
[//]: # (This is an experimental build. Does not guarantee performance, therefore. )

[//]: # ()
[//]: # (According to [shigabeev]&#40;https://github.com/shigabeev&#41;'s [experiment]&#40;https://github.com/FENRlR/MB-iSTFT-VITS2/issues/2&#41;, it can now dare claim the word SOTA for its performance &#40;at least for Russian&#41;.)
 
## How to run 

### Docker 

1. Build docker image `docker build -t istft-vits2 . `
2. Run docker image
```commandline
docker run -it --rm --net=host --ipc=host --gpus "all" -v <path_to_data_folder>:/app/data -v $PWD:/app istft-vits2 
```
3. Build Monotonic Alignment Search.
```commandline
cd monotonic_align
mkdir monotonic_align
python setup.py build_ext --inplace
```
4. Login to wandb account (type `wandb login` and copy-paste the authorization key from the browser).

5. Download data
```commandline
cd /app/data
gdown --id 1_AbWa07zKk678AUFyEtKjQo4wRWbv_HB # download checkpoints 
gdown --id 1u2vg-4zyCMgUEyRYRgO78FGX2yMyMgdT # download audio files
tar -xvzf DUMMY2.tar.gz DUMMY2
tar -xvzf train_logs.tar.gz train_logs
```
6. Run [train.py](train.py)
```commandline
python -m train -c /app/data/configs/mb_istft_vits2_finetune.json -m /app/data/train_logs
```
### Colab notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iXgYMI5fTWXI4XbY13OXrm1E9PmsnOlI?usp=sharing)


Follow instructions in [training_colab.ipynb](training_colab.ipynb)

## Homework

The main tasks are written in regular font, while _italic_ font indicates additional tasks **for extra credit**.

1. Finetune MB-iSTFT-VITS2 on 5 speakers for 50 epochs. Find the file lists in the [finetune](filelists/mipht/finetune) folder.

- Download the trained model weights and finetune model with new speaker embeddings.
- _Initialize new speaker embeddings from the weights of the old speaker embeddings._
- _Finetune the model with all parameters frozen except for the gender embedding._
- _Train the model from scratch on the finetune dataset._
- _Dive into the code and propose your conditioning mechanism, then compare the results._
   
2. Evaluate your results on test files:
- Implement at least one metric for TTS assessment and evaluate audios generated at different training stages.
- Log results to Weights & Biases (modify the [inference.py](inference.py) script).
- _Modify the [train.py](train.py) script to evaluate validation files during training._
- _Implement several metrics and evaluate the generated audio results._
  
3. Create a report using Weights & Biases (wandb) by following this guide: [Create a Report with wandb](https://docs.wandb.ai/guides/reports/create-a-report). Log loss charts, audio samples, and metrics in the report.


[//]: # (8. Edit [configurations]&#40;configs&#41; based on files and cleaners you used.)

[//]: # (## Setting json file in [configs]&#40;configs&#41;)

[//]: # (| Model | How to set up json file in [configs]&#40;configs&#41; | Sample of json file configuration|)

[//]: # (| :---: | :---: | :---: |)

[//]: # (| iSTFT-VITS2 | ```"istft_vits": true, ```<br>``` "upsample_rates": [8,8], ``` | istft_vits2_base.json |)

[//]: # (| MB-iSTFT-VITS2 | ```"subbands": 4,```<br>```"mb_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ``` | mb_istft_vits2_base.json |)

[//]: # (| MS-iSTFT-VITS2 | ```"subbands": 4,```<br>```"ms_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ``` | ms_istft_vits2_base.json |)

[//]: # (| Mini-iSTFT-VITS2 | ```"istft_vits": true, ```<br>``` "upsample_rates": [8,8], ```<br>```"hidden_channels": 96, ```<br>```"n_layers": 3,``` | mini_istft_vits2_base.json |)

[//]: # (| Mini-MB-iSTFT-VITS2 | ```"subbands": 4,```<br>```"mb_istft_vits": true, ```<br>``` "upsample_rates": [4,4], ```<br>```"hidden_channels": 96, ```<br>```"n_layers": 3,```<br>```"upsample_initial_channel": 256,``` | mini_mb_istft_vits2_base.json |)

### Training Example
```sh
python train.py -c configs/mb_istft_vits2_base.json -m <path_to_logs_and_ckpt_directory>
```

### Evaluation Example
```sh
# modify parameters in inference.py: path_to_config, path_to_model
python -m inference
```

## Credits
- [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
- [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
- [MasayaKawamura/MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS)
- [ORI-Muchim/PolyLangVITS](https://github.com/ORI-Muchim/PolyLangVITS)
- [tonnetonne814/MB-iSTFT-VITS-44100-Ja](https://github.com/tonnetonne814/MB-iSTFT-VITS-44100-Ja)
- [misakiudon/MB-iSTFT-VITS-multilingual](https://github.com/misakiudon/MB-iSTFT-VITS-multilingual)
