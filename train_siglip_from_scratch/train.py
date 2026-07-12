from transformers import TrainingArguments, Trainer, default_data_collator
from model import SiglipModel, SiglipConfig
from dataset import SiglipDataset, MyDataCollator
from transformers import AutoTokenizer, AutoProcessor
from transformers import ViTImageProcessor, ViTForImageClassification

# =============================================================================
# 训练入口:用 HF Trainer 把 SiglipModel 在 MUGE 上微调。
#
# 数据 shape 流(一个训练 step):
#   数据集 __getitem__ → {input_ids list[64], attention_mask list[64], pixel_values [1,3,224,224]}
#        │ MyDataCollator 拼 batch
#        ▼
#   {input_ids [B,64], attention_mask [B,64], pixel_values [B,3,224,224]}
#        │ model.forward
#        ▼
#   pooler_output: text [B,768], vision [B,768]
#        │ L2 norm
#        ▼
#   logits_per_text [B,B], logits_per_image [B,B] → loss [] 标量 → 反传
#
# 关键超参说明:
#   per_device_train_batch_size=32, gradient_accumulation_steps=8
#     → 有效 batch = 32 * 8 = 256。SigLIP 损失逐格独立,对小 batch 鲁棒,但大有效 batch 仍有好处。
#   learning_rate=1e-4, fp16=True → 混合精度省显存、加速。
#   num_train_epochs=40, save_steps=2000, save_total_limit=5 → 最多保留 5 个 checkpoint。
#   resume_from_checkpoint=True    → 断点续训:优先从 output_dir 里最新 checkpoint 恢复。
# =============================================================================
def train():

    config = SiglipConfig(vision_model_name_or_path='/home/user/wyf/train_siglip_from_scratch/vit-base-patch16-224',
                          text_model_name_or_path='/home/user/wyf/chinese-roberta-wwm-ext')

    model = SiglipModel(config)   # 双塔:ViT + chinese-RoBERTa,两个塔都加载预训练权重
    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name_or_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_name_or_path)

    args = TrainingArguments(
        output_dir='./outputs',
        do_train=True,
        per_device_train_batch_size=32,
        learning_rate=1e-4,
        num_train_epochs=40,
        save_steps=2000,
        save_total_limit=5,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,           # 每 100 步记一次 loss(可在 data_process.ipynb 里画曲线)
        report_to='none',            # 不上报到 wandb/tensorboard
        dataloader_pin_memory=True,  # pin memory 加速 host→GPU 拷贝
        dataloader_num_workers=1,
    )
    dataset = SiglipDataset(text_data_path='/home/user/wyf/train_siglip_from_scratch/MUGE/all_texts.jsonl',
                            image_data_path='/home/user/wyf/train_siglip_from_scratch/MUGE/all_imgs.tsv',
                            tokenizer=tokenizer,
                            processor=processor,
                            max_seq_length=64)

    # Trainer 会自动:
    #   - 把 batch dict([B,64]/[B,64]/[B,3,224,224]) 喂给 model.forward,取返回的 SiglipOutput.loss([] 标量)反传;
    #   - 用 MyDataCollator 把多个 __getitem__ 结果拼成 batch tensor。
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=MyDataCollator(tokenizer)
    )
    trainer.train(resume_from_checkpoint=True)  # 断点续训
    trainer.save_model()   # 保存到 ./outputs(可被 AutoModel.from_pretrained 加载,见 test.ipynb)
    trainer.save_state()   # 保存 optimizer/调度器等训练状态

if __name__ == '__main__':
    train()
