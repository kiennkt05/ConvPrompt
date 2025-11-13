## Convolutional Prompting meets Language Models for Continual Learning [[Paper]](https://arxiv.org/pdf/2403.20317.pdf) [[Website]](https://cvir.github.io/projects/convprompt)

This repository contains the implementation details of our Convolutional Prompting meets Language Models for Continual Learning (ConvPrompt) approach for continual learning with transformer backbone.

Anurag Roy, Riddhiman Moulick, Vinay K. Verma, Saptarshi Ghosh, Abir Das, "Convolutional Prompting meets Language Models for Continual Learning"
 

If you use the codes and models from this repo, please cite our work. Thanks!

```
@InProceedings{roy_2024_CVPR,
    author    = {Roy, Anurag and Moulick, Riddhiman and Verma, Vinay and Ghosh, Saptarshi and Das, Abir},
    title     = {Convolutional Prompting meets Language Models for Continual Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024}
}
```

## Requirements
The code is written for python `3.8.18`, but should work for other version with some modifications.
```
pip install -r requirements.txt
```
## Data preparation
If you already have CIFAR-100/ImageNet-R/CUB-200, pass your dataset path to the `--data-path` argument during execution
(If they aren't ready they will automatically get downloaded to the data-path mentioned when the `download` argument is kept True in `datasets.py`).

## Descriptor generation
The descriptors for the datasets that have been used have been stored in the format `./descriptors/descriptors_{dataset_name}`. For the generation of new descriptors, one can refer to the code from <a href="https://github.com/sachit-menon/classify_by_description_release">Visual Classification via Description from Large Language Models</a>. <br>
<i>(Note: The generation of any such task-similarity metric such as descriptors is a one-time process)</i>



## Training
### ConvPrompt / Legacy Methods
```
export TOKENIZERS_PARALLELISM=false
python -m main <cifar100_convprompt | imr_convprompt | cub_convprompt> \
    --num_tasks 10 \
    --data-path /local_datasets/ \
    --output_dir ./output
```

### RainbowPrompt (Continual Prompt Evolution)
```
# Split CUB-200
python main.py --config configs/rainbow_cubs.yaml \
    --data_path ./data/cubs \
    --output_dir ./outputs/rainbow_cubs

# Split CIFAR-100
python main.py --config configs/rainbow_cifar100.yaml \
    --data_path ./data/cifar100 \
    --output_dir ./outputs/rainbow_cifar100

# Split ImageNet-R
python main.py --config configs/rainbow_imagenet_r.yaml \
    --data_path ./data/imagenet_r \
    --output_dir ./outputs/rainbow_imagenet_r
```
Optional overrides such as `--epochs`, `--batch_size`, or `--device` can be appended to each command. The YAML file under `configs/` controls optimizer, scheduler, and RainbowPrompt-specific hyperparameters.

#### Selecting the RainbowPrompt variant
Set `rainbow.mode` in the config to choose between the implementation in this repo (`enhanced`, default) and a reproduction closer to the original paper (`paper`):

```yaml
rainbow:
  mode: paper      # or "enhanced"
  proj_dim: 96
  align_hidden_dim: 56
  ...
```


## Acknowledgement

This repo is heavily based on the  [DualPrompt](https://github.com/JH-LEE-KR/dualprompt-pytorch) Implementation.






