# IQE-CLIP: Instance-aware Query Embedding for Zero/Few-shot Anomaly Detection in Medical Domain

Official implementation of "IQE-CLIP: Instance-aware Query Embedding for Zero/Few-shot Anomaly Detection in Medical Domain" is comming soon.

[Paper Link]()



<center><img src="assets/IQE_CLIP.png "width="100%"></center>

**Abstract**:  Recently, the rapid advancements of vision-language models, such as CLIP, leads to significant progress in zero-/few-shot anomaly detection (ZFSAD) tasks. However, most existing CLIP-based ZFSAD methods commonly assume prior knowledge of categories and rely on carefully crafted prompts tailored to specific scenarios. While such meticulously designed text prompts effectively capture semantic information in the textual space, they fall short of distinguishing normal and anomalous instances within the joint embedding space. Moreover, these ZFSAD methods are predominantly explored in industrial scenarios, with few efforts conducted to medical tasks. 
To this end, we propose an innovative framework for ZFSAD tasks in medical domain, denoted as IQE-CLIP. We reveal that query embeddings, which incorporate both textual and instance-aware visual information, are better indicators for abnormalities. Specifically, we first introduce class-based prompting tokens and learnable prompting tokens for better adaptation of CLIP to the medical domain. Then, we design an instance-aware query module (IQM) to extract region-level contextual information from both text prompts and visual features, enabling the generation of query embeddings that are more sensitive to anomalies. Extensive experiments conducted on six medical datasets demonstrate that IQE-CLIP achieves state-of-the-art performance on both zero-shot and few-shot tasks.


## 🛠️  Get Started

### 🔧 Installation
To set up the IQE-CLIP environment, follow one of the methods below:

- Clone the repository::
  ```shell
  git clone https://github.com/caoyunkang/AdaCLIP.git && cd AdaCLIP
  ```
- Create a conda environment and install dependencies:   
  ```shell
  conda create -n IQECLIP python=3.9.5 -y
  conda activate IQECLIP
  conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
  pip install requirements.txt -r 
  ```


### 📦 Pretrained model
- Download CLIP pretrained weights:
[ViT-L-14-336px.pt](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt)


### 📁 Data Preparation

1. Please follow the [BMAD](https://github.com/DorisBao/BMAD) to apply for permission to download the relevant dataset. 
2. Or use the pre-processed benchmark by [MVFAD](https://github.com/MediaBrain-SJTU/MVFA-AD). Please download the following dataset.

    * [Liver](https://drive.google.com/file/d/1xriF0uiwrgoPh01N6GlzE5zPi_OIJG1I/view?usp=sharing)
    * [Brain](https://drive.google.com/file/d/1YxcjcQqsPdkDO0rqIVHR5IJbqS9EIyoK/view?usp=sharing)
    * [HIS](https://drive.google.com/file/d/1hueVJZCFIZFHBLHFlv1OhqF8SFjUVHk6/view?usp=sharing)
    * [RESC](https://drive.google.com/file/d/1BqDbK-7OP5fUha5zvS2XIQl-_t8jhTpX/view?usp=sharing)
    * [OCT17](https://drive.google.com/file/d/1GqT0V3_3ivXPAuTn4WbMM6B9i0JQcSnM/view?usp=sharing)
    * [ChestXray](https://drive.google.com/file/d/15DhnBAk-h6TGLTUbNLxP8jCCDtwOHAjb/view?usp=sharing)

3. Place it within the master directory `data` and unzip the dataset.

    ```
    tar -xvf Liver.tar.gz
    tar -xvf Brain.tar.gz
    tar -xvf Histopathology_AD.tar.gz
    tar -xvf Retina_RESC.tar.gz
    tar -xvf Retina_OCT2017.tar.gz
    tar -xvf Chest.tar.gz
    ```
4. The data file structure is as followed:
```
data/
├── Brain_AD/
│   ├── valid/
│   └── test/
├── ...
├── Retina_RESC_AD/
│   ├── valid/
│   └── test/
...
dataset/
├── fewshot_seed/
│   ├── Brain/
│   ├── ...
│   └── Retina_RESC/
├── medical_few.py
└── medical_zero.py
```


### 🚀 Train
Run the following command to train the model in zero-shot mode on a specific dataset (e.g., ``` Brain``` ).

``` 
python train_zero.py \
  --obj Brain \
  --batch_size 16 \
  --epoch 50 \
  --features_list [6,12,18,24] \
  --log_path 'Brain_zero.log' \
  --save_dir './ckpt' \
  --prompt_len 2 \
  --deep_prompt_len 1 \
  --use_global \
  --total_d_layer_len 11

```
Use the following command to train the model in few-shot mode.
This setting assumes a small number of anomaly examples are available (e.g., ``` K=4``` ),


``` 
python train_few.py \
  --obj Brain \
  --shot 4 \
  --batch_size 16 \
  --epoch 50 \
  --features_list [6,12,18,24] \
  --log_path 'Brain_few.log' \
  --save_dir './ckpt' \
  --prompt_len 2 \
  --deep_prompt_len 1 \
  --use_global \
  --total_d_layer_len 11

```

### 🔍 Test

After training, run the following command to evaluate the model in zero-shot mode.
Make sure to replace ``` $YOUR_CHECKPOINT_PATH```  with the path to your trained model checkpoint.
``` 
python test_zero.py \
  --obj Brain \
  --batch_size 16 \
  --features_list [6,12,18,24] \
  --log_path 'Test_Brain_zero.log' \
  --ckpt_path $YOUR_CHECKPOINT_PATH \
  --prompt_len 2 \
  --deep_prompt_len 1 \
  --use_global \
  --total_d_layer_len 11

```
Use the following command to evaluate the model in few-shot mode.
``` 
python test_few.py \
  --obj Brain \
  --shot 4 \
  --batch_size 16 \
  --features_list [6,12,18,24] \
  --log_path 'Test_Brain_few.log' \
  --ckpt_path $YOUR_CHECKPOINT_PATH \
  --prompt_len 2 \
  --deep_prompt_len 1 \
  --use_global \
  --total_d_layer_len 11

```

## Acknowledgement
Our work is largely inspired by the following projects. Thanks for their admiring contribution.
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [April-GAN](https://github.com/ByChelsea/VAND-APRIL-GAN)
- [AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP)
- [CoCoOp](https://github.com/KaiyangZhou/CoOp)
- [BMAD](https://github.com/DorisBao/BMAD)
- [MVFAD](https://github.com/MediaBrain-SJTU/MVFA-AD)
- [VCPCLIP](https://github.com/xiaozhen228/VCP-CLIP)
-[AdaCLIP](https://github.com/caoyunkang/AdaCLIP)

## Citation

If you find this project helpful for your research, please consider citing the following BibTeX entry.

```BibTex


```