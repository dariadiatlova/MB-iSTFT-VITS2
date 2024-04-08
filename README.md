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
### Colab notebook
Follow instructions in [training_colab.ipynb](training_colab.ipynb)

## Homework

1. Finetune MB-iSTFT-VITS2 on 5 speakers. Find filelists in [finetune](filelists/mipht/finetune) folder.

    - train new speaker embeddings for new speakers
    - fine-tune old speaker-embeddings with new-speaker voices*
    - fine-tune the whole network and partly frozen, compare the results*
    - dive into the code and propose your conditioning mechanism, compare the results**
   
2. Implement at least one metric for TTS generation assessment:

    - finetune the model for 50 epoch, evaluate generation for each 10th checkpoint;
    - log results to wandb (modify [inference.py](inference.py) script)
    - modify [train.py](train.py) and evaluate train / evaluation files during training. 
3. Create the report using wandb, attach loss charts, audio samples and metrics on different epochs.  
      

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
python inference.py 
```

## Credits
- [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
- [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
- [MasayaKawamura/MB-iSTFT-VITS](https://github.com/MasayaKawamura/MB-iSTFT-VITS)
- [ORI-Muchim/PolyLangVITS](https://github.com/ORI-Muchim/PolyLangVITS)
- [tonnetonne814/MB-iSTFT-VITS-44100-Ja](https://github.com/tonnetonne814/MB-iSTFT-VITS-44100-Ja)
- [misakiudon/MB-iSTFT-VITS-multilingual](https://github.com/misakiudon/MB-iSTFT-VITS-multilingual)
