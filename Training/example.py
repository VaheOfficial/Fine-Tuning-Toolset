import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Specify the directory containing the model parts
model_dir = "C:/Users/VaheOfficial/projects/Data Science/Training/Meta-Llama-3-8B-Instruct/"

# Check if the directory exists
if not os.path.isdir(model_dir):
    raise ValueError(f"The specified directory {model_dir} does not exist.")

# List the contents of the directory to verify the presence of files
# print("Contents of model directory:", os.listdir(model_dir))

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Set the pad token to be the same as the eos token
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

# Load the model and move it to the GPU
model = LlamaForCausalLM.from_pretrained(model_dir, quantization_config=bnb_config)

# Apply PEFT
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

print("Model and tokenizer loaded successfully.")

# Load the Alpaca data from the JSON file
data_file = 'C:/Users/VaheOfficial/projects/Data Science/Detroit Data/detroit_become_human_alpaca_fine_tuning_data.json'
with open(data_file, 'r') as file:
    alpaca_data = json.load(file)

print(f"Loaded {len(alpaca_data)} training examples.")

# Define a function to tokenize the instruction-output pairs
def tokenize_examples(examples, tokenizer, max_length=512):
    input_texts = [example['instruction'] for example in examples]
    target_texts = [example['output'] for example in examples]

    inputs = tokenizer(input_texts, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    targets = tokenizer(target_texts, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')

    return inputs, targets

# Tokenize the data
inputs, targets = tokenize_examples(alpaca_data, tokenizer)

print("Tokenized input examples:", inputs['input_ids'][:2])
print("Tokenized target examples:", targets['input_ids'][:2])

class AlpacaDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.inputs.items()}
        item['labels'] = torch.tensor(self.targets['input_ids'][idx])
        return item

# Create the dataset
dataset = AlpacaDataset(inputs, targets)

# Use DataCollatorForSeq2Seq for formatting
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Example of iterating through the DataLoader
for batch in dataloader:
    print(batch)
    break

# Define training arguments
training_args = TrainingArguments(
    output_dir='C:/Users/VaheOfficial/projects/Data Science/Training/results',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='C:/Users/VaheOfficial/projects/Data Science/Training/logs',
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    fp16=True
)

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset  # Optionally, use a separate validation dataset
)

# Start fine-tuning
trainer.train()

# Save the fine-tuned model
model.save_pretrained('C:/Users/VaheOfficial/projects/Data Science/Training/fine-tuned-llama')
tokenizer.save_pretrained('C:/Users/VaheOfficial/projects/Data Science/Training/fine-tuned-llama')

# Evaluate the model
results = trainer.evaluate()
print(f"Evaluation results: {results}")
