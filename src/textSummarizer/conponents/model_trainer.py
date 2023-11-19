import os
import torch
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    def train(self):
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)


        def convert_examples_to_features(example_batch):
            input_encodings = tokenizer(example_batch['dialogue'],
                                        max_length=64, #1024
                                        truncation = True )

            with tokenizer.as_target_tokenizer():
                target_encodings = tokenizer(
                    example_batch['summary'],
                    max_length=64, # 128
                    truncation=True
                )

            return{
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
                'labels': target_encodings['input_ids']
            }

        #loading data 
        dataset_samsum_pt = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum_pt.map(convert_examples_to_features, batched = True)


        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, num_train_epochs=1, warmup_steps=500,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            weight_decay=0.01, logging_steps=10,
            evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
            gradient_accumulation_steps=16
        ) 

        trainer = Trainer(model=model_pegasus, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=dataset_samsum_pt["train"], 
                  eval_dataset=dataset_samsum_pt["validation"])
        
        trainer.train()

        ## Save model
        model_pegasus.save_pretrained(os.path.join(self.config.root_dir,"pegasus-samsum-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))