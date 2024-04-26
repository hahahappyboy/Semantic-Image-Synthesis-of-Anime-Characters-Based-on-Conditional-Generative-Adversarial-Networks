# Semantic lmage Synthesis of Anime Characters Based on Generative Adversarial Networks

**Abstract**: The goal of semantic image synthesis is to generate realistic images from semantic label images. However, current state-of-the-art approaches for generating anime characters from semantic images still face challenges, especially in the inability to control the network to generate a specific anime character and in the resulting confusion of texture details in the generated anime characters. To address this, we introduce a Generative Adversarial Network for Semantic Image Synthesis of Anime Characters, featuring a conditional generator based on character identity for precise character generation and high-quality colors, as well as discriminators leveraging semantic segmentation and edge detection to enhance texture realism. In our experiments, we constructed two new datasets collected from popular anime works, including annotations for semantic images. Experimental results show the superiority of our proposed method in generating specific and realistic anime characters compared to existing methods.

<p align="center">
<img src="figs/result_1.png" >
</p>

# Datasets

The Quintuplets and Zero Two Datasets can be downloaded [here]() as zip files. Copy them into the checkpoints folder (the default is `./datasets`, create it if it doesn't yet exist) and unzip them. The folder structure should be

```
datasets
├── Quintuplets                   
└── ZeroTwo
```

# Pretrained models

The checkpoints for the pre-trained models are available [here](https://pan.baidu.com/s/1_oV5gn33nUtZZtwGVR0yGA?pwd=1234) as zip files. Copy them into the checkpoints folder (the default is `./checkpoints`, create it if it doesn't yet exist) and unzip them. The folder structure should be

```
checkpoints
├── quintuplets                 
└── zero_two
```

You can generate images with a pre-trained checkpoint via `test.py`. 

The example of Quintuplets:

```
python test.py --class_num 9  --name quintuplets --class_dir ./datasets/Quintuplets/class.txt \
--ckpt_iter 140000 --dataset_mode custom --dataroot ./datasets/Quintuplets  --batch_size 1 --gpu_ids 0
```

The example of Zero Two :

```
python test.py --class_num 15  --name zero_two --class_dir ./datasets/ZeroTwo/class.txt \
--ckpt_iter 140000 --dataset_mode custom --dataroot ./datasets/ZeroTwo  --batch_size 1 --gpu_ids 0
```

# Train the model

To train on the Quintuplets dataset, for example:

```
python train.py --name quintuplets --class_dir datasets/Quintuplets/class.txt --class_num 9 --dataset_mode custom \
--dataroot ./datasets/Quintuplets --gpu_ids 0 --num_epochs 400 --batch_size 1 --freq_print 1000 --freq_save_latest 10000
```

To train on the Zero Two dataset, for example:

```
python train.py --name zero_two --class_dir datasets/ZeroTwo/class.txt --class_num 15 --dataset_mode custom \
--dataroot ./datasets/ZeroTwo --gpu_ids 0 --num_epochs 400 --batch_size 1 --freq_print 1000 --freq_save_latest 10000
```

The `--class_dir` is the path for storing the character ID tag txt. The `--class_num`  is the number of anime characters contained in the dataset.

# Test the model

To test on the Quintuplets dataset, for example:

```
python test.py --class_num 9  --name quintuplets --class_dir ./datasets/Quintuplets/class.txt \
--ckpt_iter 140000 --dataset_mode custom --dataroot ./datasets/Quintuplets  --batch_size 1 --gpu_ids 0
```

To test on the Zero Two dataset, for example:

```
python test.py --class_num 15  --name zero_two --class_dir ./datasets/ZeroTwo/class.txt \
--ckpt_iter 140000 --dataset_mode custom --dataroot ./datasets/ZeroTwo  --batch_size 1 --gpu_ids 0
```

# Result
Our method generates images with higher quality colors and more natural textures compared to previous methods
<p align="center">
<img src="figs/rusult_2.png" >
</p>

<p align="center">
<img src="figs/result_4.png" >
</p>

Our method has better results on FID and KID compared to previous methods, and has second best results on LPIPS

<p align="center">
<img src="figs/result_3.png" >
</p>

# Contact

If you have any questions, please create an issue on this repository or contact us at 

# Citation

If you use this work please cite

# Acknowledgement

This code is based on [https://github.com/boschresearch/OASIS](OASIS)

# License

This project is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).
