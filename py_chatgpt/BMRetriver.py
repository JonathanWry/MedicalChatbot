import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset



def load_bmretriever_model(model_name="BMRetriever/BMRetriever-410M", device="cpu"):
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        embedding = last_hidden[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden.shape[0]
        embedding = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
    return embedding


def get_detailed_instruct_query(task_description: str, query: str) -> str:
    return f'{task_description}\nQuery: {query}'


def get_detailed_instruct_passage(passage: str) -> str:
    return f'Represent this passage\npassage: {passage}'


def load_pubmed_documents(num_docs=50):
    ds = load_dataset("MedRAG/pubmed", split="train", streaming=True)
    ds_iter = iter(ds)

    documents = []
    for _ in range(num_docs):
        row = next(ds_iter, None)
        if row is None:
            break

        # row typically has: 'title', 'content' (abstract), 'contents' (full text)
        title = row.get("title", "")
        abstract = row.get("content", "")
        full_txt = row.get("contents", "")

        # Merge into a single string
        combined_passage = (
            f"Title: {title}\n"
            f"Abstract: {abstract}\n"
            f"Full Text: {full_txt}"
        )
        documents.append(combined_passage)

    return documents

def encode_texts_with_eos(texts, tokenizer, max_length=1024, device="cpu"):
    """
    Tokenizes a list of texts, appends the EOS token, and re-pads so each sequence
    has the same length. Returns a dict of PyTorch tensors.
    """
    # 1) First tokenize all texts together
    batch_dict = tokenizer(
        texts,
        max_length=max_length - 1,  # subtract 1 so we have space for the EOS
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    # 2) Convert to lists, append EOS token to each sequence
    eos_id = tokenizer.eos_token_id
    new_input_ids = []
    new_attention_masks = []
    for input_ids, attn_mask in zip(batch_dict["input_ids"], batch_dict["attention_mask"]):
        # Convert each row to a Python list
        seq_list = input_ids.tolist()
        mask_list = attn_mask.tolist()

        # Append EOS token and increment attention mask
        seq_list.append(eos_id)
        mask_list.append(1)  # 1 => attend to the EOS token

        new_input_ids.append(seq_list)
        new_attention_masks.append(mask_list)

    # 3) Re-pad with the tokenizer to ensure uniform length
    #    Note: we pass lists of lists, and let pad handle final shape
    re_batch_dict = tokenizer.pad(
        {
            "input_ids": new_input_ids,
            "attention_mask": new_attention_masks
        },
        padding=True,
        return_tensors='pt'
    )

    # 4) Move to device (cpu or cuda)
    re_batch_dict = {k: v.to(device) for k, v in re_batch_dict.items()}

    return re_batch_dict

# Precompute doc embeddings
def encode_documents(documents, tokenizer, model, max_length=1024, device="cpu"):
    formatted_docs = [get_detailed_instruct_passage(doc) for doc in documents]
    docs_batch = encode_texts_with_eos(formatted_docs, tokenizer, max_length=max_length, device=device)

    with torch.no_grad():
        outputs = model(**docs_batch)
        doc_embeddings = last_token_pool(outputs.last_hidden_state, docs_batch["attention_mask"])
    return doc_embeddings
def encode_query(query, tokenizer, model, max_length=1024, device="cpu"):
    task = "Given a scientific query, retrieve documents that are most relevant from pubmed"
    formatted_query = get_detailed_instruct_query(task, query)
    query_batch = encode_texts_with_eos([formatted_query], tokenizer, max_length=max_length, device=device)

    with torch.no_grad():
        outputs = model(**query_batch)
        query_embedding = last_token_pool(outputs.last_hidden_state, query_batch["attention_mask"])
    # query_embedding is shape [1, hidden_dim], so do .squeeze(0) if desired
    return query_embedding[0]

# def retrieve_relevant_documents(query: str, model, tokenizer, documents, top_k=5):
#     # 1. Prepare the query & doc strings
#     task = "Given a scientific query, retrieve documents that are most relevant from pubmed"
#     formatted_query = get_detailed_instruct_query(task, query)
#     formatted_documents = [get_detailed_instruct_passage(doc) for doc in documents]
#     # Combine query + docs
#     input_texts = [formatted_query] + formatted_documents
#     # 2. Tokenize with standard padding/truncation
#     max_length = 1024
#     batch_dict = tokenizer(
#         input_texts,
#         max_length=max_length - 1,  # or keep just max_length=512
#         padding=True,
#         truncation=True,
#         return_tensors='pt'
#     )
#
#     # 3. Manually append EOS token to each sequence
#     #    We convert each row in encoded['input_ids'] to a list, add eos, then re-pad
#     new_input_ids = []
#     batch_dict = tokenizer.pad(batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')
#     for seq in batch_dict['input_ids']:
#         # Convert the tensor to a Python list, then append EOS
#         seq_list = seq.tolist()
#         seq_list.append(tokenizer.eos_token_id)
#         new_input_ids.append(seq_list)
#
#     padded = tokenizer.pad(
#         batch_dict,
#         padding=True,
#         return_attention_mask=True,
#         return_tensors='pt'
#     )
#
#     model.eval()
#     with torch.no_grad():
#         outputs = model(**padded)
#         embeddings = last_token_pool(outputs.last_hidden_state, padded['attention_mask'])
#
#     # 6. Separate query embedding from doc embeddings
#     query_embedding = embeddings[0]
#     doc_embeddings = embeddings[1:]
#
#     # 7. Compute similarity & retrieve top_k
#     scores = (query_embedding @ doc_embeddings.T)
#     top_indices = scores.topk(top_k).indices.cpu().tolist()
#
#     # 8. Return the top-k documents
#     return [documents[idx] for idx in top_indices]

def retrieve_relevant_documents(
    query: str,
    model,
    tokenizer,
    doc_embeddings: torch.Tensor,
    documents: list,
    top_k=5,
    device="cpu"
):
    query_embedding=encode_query(query, tokenizer, model, device=device)
    scores = query_embedding @ doc_embeddings.T
    top_indices = torch.topk(scores, k=top_k).indices.cpu().tolist()

    # --- STEP 4: Return the top_k documents ---
    top_documents = [documents[i] for i in top_indices]
    return top_documents


# Example usage
if __name__ == "__main__":
    model, tokenizer = load_bmretriever_model(model_name="BMRetriever/BMRetriever-410M")
    documents = load_pubmed_documents(num_docs=30)
    documents_embedding=encode_documents(documents, tokenizer, model)
    query = "Give me information regarding o-salicylate."
    top_documents = retrieve_relevant_documents(query, model, tokenizer, documents_embedding,documents, top_k=5, device="cpu")

    print("\nTop Retrieved Papers:")
    for i, doc in enumerate(top_documents):
        print(f"{i + 1}. {doc}\n")

