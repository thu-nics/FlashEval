# FlashEval: Towards Fast and Accurate Evaluation of Text-to-image Diffusion Generative Models

This repository is the official implementation of the paper:

[**FlashEval: Towards Fast and Accurate Evaluation of Text-to-image Diffusion Generative Models**](https://arxiv.org/abs/2403.16379)
[*Lin Zhao*](),
[*Tianchen Zhao*](),
[*Zinan Lin*](),
[*Xuefei Ning*](),
[*Guohao Dai*](),
[*Huazhong Yang*](),
[*Yu Wang*]()
CVPR, 2024. [**[Project Page]**](https://telling-left-from-right.github.io) [**[Paper]**](https://arxiv.org/abs/2403.16379)

## Usage

1. [Setup](#setup)
2. [One-time generation](#one-time-generation)
3. [Preprocessing](#preprocessing)
4. [Search algorithm](#search-algorithm)
5. [Condensed subset](#condensed-subset)

## Setup

To install the required dependencies, use the following commands:

```bash
conda create -n flasheval python=3.9
conda activate flasheval
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
cd FlashEval
pip install -e .
```


## One-time generation
We process the [*COCO*](https://cocodataset.org/#home) and [*diffusionDB*](https://poloclub.github.io/diffusiondb/) datasets used in the paper as an example here, which you can easily change to your own dataset. Besides, if you use the same model settings as in the paper, we give all the processed data of One-time Generation for your convenience in [all_metrics](Image_score/all_metrics/).

### Textual dataset
We give the prompts of the two datasets processed according to Section 5.1: [COCO_40504.json](data/COCO_40504.json) and [diffusionDB_5000.json](data/diffusionDB_5000.json).

### Image Generation
For full precision model, running the following command. For quant models, please use [q-diffusion](https://github.com/Xiuyu-Li/q-diffusion) to generate images.
```bash
python Image_generation/generate.py --model_name <choose model> --scheduler <choose scheduler> --gpu_id <GPU ID to use for CUDA> --seed <seed> --step <step> --save_dir <save_path>
```

### Image Evaluation
The metrics considered in this paper are: 
1) [CLIP](https://arxiv.org/abs/2103.00020), which measures the text-image alignment
2) [Aesthetic](https://github.com/christophschuhmann/improved-aesthetic-predictor), which measures how good-looking an image is
3) [ImageReward](https://github.com/THUDM/ImageReward), which measures the human preference of an image
4) [HPS](https://tgxs002.github.io/align_sd_web/), which measures the human preference of an image
5) [FID](https://arxiv.org/abs/1706.08500), accelerating FID evaluation with a batch-based approach

For HPS, you need to download the pre-trained model first [HPS](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155172150_link_cuhk_edu_hk/EWDmzdoqa1tEgFIGgR5E7gYBTaQktJcxoOYRoTHWzwzNcw?e=b7rgYW); For other metrics, running the evaluation comment will automatically download the models. 

1. For single-image evaluation metrics, running the following command:
```bash
python Evaluation/test_benchmark.py --config configs/evaluation.yaml
```
2. For FID, please download the ground truth images from [*COCO*](https://cocodataset.org/#home) and calculate the score.

## Preprocessing
1. Organize model scores and divide them into training models and testing models. All the processed data of One-time Generation are in [all_metrics](Image_score/all_metrics/).
```bash
python preprocess/get_model_score.py --config configs/get_score.yaml
```


## Search algorithm
```bash
python search_algorithm.py --config configs/search.yaml
```
Note: Since constructing random subsets involves randomness, the results may vary each time the process is run. As iteration increases, the effect gets better.

## Condensed subset
coming soon

## Citation
If you find our work useful, please cite:

```BiBTeX
@article{zhao2024flasheval,
  title={FlashEval: Towards Fast and Accurate Evaluation of Text-to-image Diffusion Generative Models},
  author={Zhao, Lin and Zhao, Tianchen and Lin, Zinan and Ning, Xuefei and Dai, Guohao and Yang, Huazhong and Wang, Yu},
  journal={arXiv preprint arXiv:2403.16379},
  year={2024}
}
```