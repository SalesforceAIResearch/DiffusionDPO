# Intro

This is the training code for [Diffusion-DPO](https://arxiv.org/abs/2311.12908). The script is adapted from [the diffusers library](https://github.com/huggingface/diffusers/tree/v0.20.0-release/examples/text_to_image).


# Model Checkpoints

The below are initialized with StableDiffusion models and trained as described in the paper (replicable with [launchers/](launchers/) scripts assuming 16 GPUs, scale gradient accumulation accordingly).

[StableDiffusion1.5](https://huggingface.co/mhdang/dpo-sd1.5-text2image-v1)

[StableDiffusion-XL-1.0](https://huggingface.co/mhdang/dpo-sdxl-text2image-v1?text=Test)

Use this [notebook](quick_samples.ipynb) to compare generations. It also has a sample of automatic quantative evaluation using PickScore.


# Setup

`pip install -r requirements.txt`

# Structure

- `launchers/` is examples of running SD1.5 or SDXL training
- `utils/` has the scoring models for evaluation or AI feedback (PickScore, HPS, Aesthetics, CLIP)
- `quick_samples.ipynb` is visualizations from a pretrained model vs baseline
- `requirements.txt` Basic pip requirements
- `train.py` Main script, this is pretty bulky at >1000 lines, training loop starts at ~L1000 at this commit (`ctrl-F` "for epoch").
- `upload_model_to_hub.py` Uploads a model checkpoint to HF (simple utility, current values are placeholder)

# Running the training

Example SD1.5 launch

```bash
# from launchers/sd15.sh
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="yuvalkirstain/pickapic_v2"

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

accelerate launch --mixed_precision="fp16"  train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="/export/share/datasets/vision_language/pick_a_pic_v2/" \
  --checkpointing_steps 500 \
  --beta_dpo 5000 \
   --output_dir="tmp-sd15"
```

## Important Args

### General

- `--pretrained_model_name_or_path` what model to train/initalize from
- `--output_dir` where to save/log to
- `--seed` training seed (not set by default)
- `--sdxl` run SDXL training
- `--sft` run SFT instead of DPO

### DPO

- `--beta_dpo` KL-divergence parameter beta for DPO
- `--choice_model` Model for AI feedback (Aesthetics, CLIP, PickScore, HPS)

### Optimizers/learning rates

- `--max_train_steps` How many train steps to take
- `--gradient_accumulation_steps`
- `--train_batch_size` see above notes in script for actual BS
- `--checkpointing_steps` how often to save model
  
- `--gradient_checkpointing` turned on automatically for SDXL


- `--learning_rate`
- `--scale_lr` Found this to be very helpful but isn't default in code
- `--lr_scheduler` Type of LR warmup/decay. Default is linear warmup to constant
- `--lr_warmup_steps` number of scheduler warmup steps
- `--use_adafactor` Adafactor over Adam (lower memory, default for SDXL)

### Data
- `--dataset_name` if you want to switch from Pick-a-Pic
- `--cache_dir` where dataset is cached locally **(users will want to change this to fit their file system)**
- `--resolution` defaults to 512 for non-SDXL, 1024 for SDXL.
- `--random_crop` and `--no_hflip` changes data aug
- `--dataloader_num_workers` number of total dataloader workers

# Citation

```
@misc{wallace2023diffusion,
      title={Diffusion Model Alignment Using Direct Preference Optimization}, 
      author={Bram Wallace and Meihua Dang and Rafael Rafailov and Linqi Zhou and Aaron Lou and Senthil Purushwalkam and Stefano Ermon and Caiming Xiong and Shafiq Joty and Nikhil Naik},
      year={2023},
      eprint={2311.12908},
      archivePrefix={arXiv},
      primaryClass={cs.CV}

```

# Ethical Considerations

This release is for research purposes only in support of an academic paper. Our models, datasets, and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before deploying this model. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people’s lives, rights, or safety. For further guidance on use cases, refer to our AUP and AI AUP.

