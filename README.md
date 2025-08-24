# Efficient Fine-Tuning with LoRA and Model Quantization

This repository demonstrates how to combine Low-Rank Adaptation (LoRA) with model quantization for efficient fine-tuning of large language models. This workflow dramatically reduces the memory needs, cost, and training time—making advanced AI accessible even on modest hardware.

---

## Table of Contents

- [Introduction](#introduction)
- [What is LoRA?](#what-is-lora)
- [What is Model Quantization?](#what-is-model-quantization)
- [Why Use Both?](#why-use-both)
- [Quickstart: Code Example](#quickstart-code-example)
- [Results](#results)
- [Conclusion](#conclusion)

---

## Introduction

Fine-tuning powerful neural models once required expensive compute and huge storage. With **LoRA** and **quantization**, you can:

- Fine-tune only small, trainable adapter layers (LoRA)
- Shrink models and boost training/inference speed by using lower-precision weights (quantization)

This project shows you how to do both, as in the `gemma` experiments.

---

## What is LoRA?

LoRA (Low-Rank Adaptation) adds small "adapters" to specific model weights and only updates those during training.

**Benefits**:
- Reduces memory footprint while fine-tuning
- Speeds up training
- Lets you keep/adapt multiple tasks by swapping adapters

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)
print(model.print_trainable_parameters())

```
---

## What is Model Quantization?

Quantization converts 32-bit (or 16-bit) weights to 8-bit or 4-bit. The result: smaller, faster models.

**Benefits**:
- Models fit in GPU/CPU RAM
- Speed up training/inference

**Example: Load a Quantized Model**

```python

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
"your-model-id",
quantization_config=bnb_config,
device_map="auto"
)

```
---

## Why Use Both?

- **Fits big models on small hardware**: Train large LLMs or Transformers on a single GPU
- **Trains faster/cheaper**: Less compute needed 
- **Tasks are modular**: Mix & match LoRA adapters for different jobs

---

## Quickstart: Code Example

Here’s how to combine LoRA and quantization for a fast, light fine-tune:

Step 1: Load quantized model & tokenizer
```python

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
"your-pretrained-model",
quantization_config=bnb_config,
device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("your-pretrained-model")

```

Step 2: Prepare LoRA config and inject adapters

```python
lora_config = LoraConfig(
task_type=TaskType.CAUSAL_LM,
r=8,
lora_alpha=32,
lora_dropout=0.05,
target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

```
Step 3: (Usual) Training pipeline

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
per_device_train_batch_size=2,
num_train_epochs=1,
logging_steps=10,
output_dir="./output"
)

trainer = Trainer(model=model, args=training_args, train_dataset=my_dataset)
trainer.train()
```

---

## Results

By combining LoRA and quantization, we observed:
- Lower memory usage (able to fit full models on a single GPU or consumer hardware)
- Faster fine-tuning
- Consistent reduction in training loss (see attached wandb logs or training screenshots)
- No loss of predictive power relative to full-precision, full-model tuning in many cases

---

## Conclusion

If you want to fine-tune large AI models without massive compute, use LoRA and quantization together. These techniques democratize AI and enable personalized, efficient, and affordable deep learning in any setting.

---


**Example: Apply LoRA with Hugging Face**

