from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"
local_dir = "./llmmodels/gpt2"

# Download and save locally
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save locally
model.save_pretrained(local_dir)
tokenizer.save_pretrained(local_dir)
