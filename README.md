# Fine-Tuning StarCoderbase-1B with LoRA and FIM

This repository contains code and configuration details for fine-tuning the [bigcode/starcoderbase-1b](https://huggingface.co/bigcode/starcoderbase-1b) model using Parameter-Efficient Fine-Tuning (PEFT) with LoRA, Fill-in-the-Middle (FIM) augmentation, and 4-bit quantized training.

## 🔧 Setup

### Model & Tokenizer

- **Model**: `bigcode/starcoderbase-1b`
- **Tokenizer**: Loaded via Hugging Face Transformers with the padding token set to the EOS token:
  ```
  tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")
  tokenizer.pad_token = tokenizer.eos_token
  ```
