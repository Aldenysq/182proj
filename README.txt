export MODEL_ID="timbrooks/instruct-pix2pix"
export DATASET_ID="aldenn13l/182-fine-tune"
export OUTPUT_DIR="cartoonization-finetuned"

accelerate launch --mixed_precision="fp16" finetune_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --dataset_name=$DATASET_ID \
  --use_ema \
  --resolution=256 --random_flip \
  --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=15000 \
  --checkpointing_steps=5000 --checkpoints_total_limit=1 \
  --learning_rate=5e-05 --lr_warmup_steps=0 \
  --mixed_precision=fp16 \
  --val_image_url="https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png" \
  --validation_prompt="Generate a cartoonized version of the natural image" \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --report_to=tensorboard \
  --push_to_hub