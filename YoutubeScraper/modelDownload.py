from pyannote.audio import Pipeline
from transformers import AutoConfig, AutoModel, AutoTokenizer
import speechbrain as sb
import os


# Specify the local directory where you want to save the model
local_model_path = "localModel"

# Ensure the directory exists
os.makedirs(local_model_path, exist_ok=True)

# Download and save the model locally
pipelinee = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")
print(pipelinee)

# Save the segmentation model
segmentation_model = pipelinee.model['segmentation']
segmentation_model.save_pretrained(os.path.join(local_model_path, 'segmentation'))

# Save the embedding model if it exists
if 'embedding' in pipelinee.model:
    embedding_model = pipelinee.model['embedding']
    embedding_model.save_pretrained(os.path.join(local_model_path, 'embedding'))

# Save the configuration
config = AutoConfig.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="YOUR_HF_TOKEN_HERE")
config.save_pretrained(local_model_path)

# Save the tokenizer if used
tokenizer = AutoTokenizer.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="YOUR_HF_TOKEN_HERE")
tokenizer.save_pretrained(local_model_path)