from mteb import MTEB
from sentence_transformers import SentenceTransformer
from tasks import *
import os
import torch
from transformers import AutoModel, AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define the sentence-transformers model name
model_name = "intfloat/e5-base-v2"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# load from disk
class embedding_model:
    def __init__(self, model, tokenizer, device, inference=False):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        if inference:
            self.model.eval()
            self.model.requires_grad_(False)

    def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def encode(self, sentences, batch_size=32):
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            embedding = self.__call__(batch_sentences)
            # Add each sentence embedding to the list as a separate ndarray
            embeddings.extend(embedding.detach().cpu().numpy())
        return embeddings

    def __call__(self, data):
        tokens_and_mask = self.tokenizer(data, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        model_output = self.model(tokens_and_mask["input_ids"], attention_mask=tokens_and_mask["attention_mask"])
        embedding = embedding_model.average_pool(model_output.last_hidden_state, attention_mask=tokens_and_mask["attention_mask"])
        # normalize the embedding
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding    


model = embedding_model(AutoModel.from_pretrained(model_name), AutoTokenizer.from_pretrained("intfloat/e5-base-v2"), device=device, inference=True)
# model = model.to('mps')
evaluation = MTEB(tasks=["MassiveIntentClassification","MassiveScenarioClassification",SummEvalES(), RedditClusteringES(), TwentyNewsgroupsClusteringES(), SciFactES(), 'STS22', TwitterURLCorpusPCES(), MindSmallRerankingES()], task_langs=["es"])
results = evaluation.run(model, verbosity=2, output_folder=f"results/{model_name}/es")