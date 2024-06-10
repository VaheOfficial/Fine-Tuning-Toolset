import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline

# Load the base model and tokenizer
base_model_name = "C:/Users/VaheOfficial/projects/Data Science/Training/Meta-Llama-3-8B-Instruct"  # Path to the base model

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Ensure the model is on the correct device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)

# Create the HuggingFace pipeline
hf_pipeline = pipeline(
    task='text-generation',
    model=base_model,
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
Human: {human_input}
AI:"""

prompt = PromptTemplate(template=template, input_variables=["conversation", "human_input"])

# Create the LangChain LLMChain
conversation_chain = LLMChain(prompt=prompt, llm=langchain_pipeline)

# Interactive loop to converse with the AI using LangChain
def chat_with_ai(conversation_chain):
    conversation = ""
    print("Chat with your AI! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending the conversation. Goodbye!")
            break

        # Prepare input for the model
        inputs = {"conversation": conversation, "human_input": user_input}
        response = conversation_chain(inputs)["text"]

        # Update conversation history
        conversation += f"Human: {user_input}\nAI: {response.strip()}\n"
        print(f"AI: {response.strip()}")

if __name__ == "__main__":
    chat_with_ai(conversation_chain)
