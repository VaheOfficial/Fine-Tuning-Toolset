import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig, LoraConfig
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

# Load the base model and tokenizer
model_dir = 'C:/Users/VaheOfficial/projects/Data Science/Training/fine-tuned-llama'  # Directory where your model is saved
base_model_name = "C:/Users/VaheOfficial/projects/Data Science/Training/Meta-Llama-3-8B-Instruct"  # Replace this with the name of the base model you used

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load the adapter configuration from the directory
adapter_config = PeftConfig.from_pretrained(model_dir)

# Ensure that the adapter configuration is of type LoraConfig
if not isinstance(adapter_config, LoraConfig):
    raise TypeError("Expected adapter_config to be an instance of LoraConfig")

# Load the adapter model with the configuration
lora_model = PeftModel.from_pretrained(model_id=model_dir, model=base_model, config=adapter_config)

# Ensure the model is on the correct device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lora_model.to(device)

# Create the HuggingFace pipeline
hf_pipeline = pipeline(
    task='text-generation',
    model=lora_model.model,  # Use the underlying model for text generation
    tokenizer=tokenizer,
    device=device.index if device.type == 'cuda' else -1,
    pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id to avoid padding issues
    num_beams=5,  # Beam search to improve coherence
    early_stopping=True,
    no_repeat_ngram_size=2  # Avoid repetition
)

# Wrap the pipeline in HuggingFacePipeline for LangChain
langchain_pipeline = HuggingFacePipeline(pipeline=hf_pipeline)

# Define the prompt template for the conversation
template = """
You are an AI assistant. The following is a conversation between you and a human.

{conversation}

AI:
"""

prompt = PromptTemplate(template=template, input_variables=["conversation"])

# Create the LangChain LLMChain
conversation_chain = LLMChain(llm=langchain_pipeline, prompt=prompt)

# Interactive loop to converse with the AI using LangChain
def chat_with_ai(conversation_chain):
    conversation = ""
    print("Chat with your AI! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending the conversation. Goodbye!")
            break

        conversation += f"You: {user_input}\n"
        response = conversation_chain({"conversation": conversation})["text"]
        conversation += f"AI: {response.strip()}\n"
        print(f"AI: {response.strip()}")

if __name__ == "__main__":
    chat_with_ai(conversation_chain)
