from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import torch
from torch.nn.functional import softmax
import csv
import pandas as pd
from datasets import load_dataset


TRANSFORMER_MAX_LENGTH = 512


if __name__ == '__main__':
    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
    embedding_model = BertModel.from_pretrained("ProsusAI/finbert")
    classifier_model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")

    embedding_model.eval()
    classifier_model.eval()


    # Load dataset
    dataset = load_dataset("danidanou/Bloomberg_Financial_News")
    df = pd.DataFrame(dataset['train'])
    df['day'] = df['Date'].dt.date
    print(len(df['day'].unique()))
    rows = []
    idx1 = 0
    for day in df['day'].unique():
        idx1 += 1
        print(day)
        filtered = df[df['day'] == day]
        avg_embedding = None
        num_samples = min(5, len(filtered))
        for _, row in filtered.sample(n=num_samples, random_state=42).iterrows():
            article = row['Article']
            max_prob_pos = 0
            max_prob_neg = 0
            embedding_pos = None
            embedding_neg = None
            for i in range(0, len(article), TRANSFORMER_MAX_LENGTH):
                chunk = article[i:i+TRANSFORMER_MAX_LENGTH]
                inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=TRANSFORMER_MAX_LENGTH)
                with torch.no_grad():
                    cls_output = classifier_model(**inputs)
                    # labels = ["positive", "negative", "neutral"]
                    probs = torch.softmax(cls_output.logits, dim=-1)
                    emb_output = embedding_model(**inputs)
                    cls_embedding = emb_output.last_hidden_state[:, 0, :]
                    # positive
                    if probs[0, 0].item() > max_prob_pos:
                        max_prob_pos = probs[0, 0].item()
                        embedding_pos = cls_embedding.detach().clone()
                    # negative
                    if probs[0, 1].item() > max_prob_neg:
                        max_prob_neg = probs[0, 1].item()
                        embedding_neg = cls_embedding.detach().clone()
                embedding = max_prob_pos * embedding_pos + max_prob_neg * embedding_neg
                if avg_embedding is None:
                    avg_embedding = embedding[0, :].detach().clone()
                else:
                    avg_embedding += embedding[0, :].detach().clone()
        avg_embedding = avg_embedding / num_samples
        rows.append({
            'date': day,
            'embedding': avg_embedding.tolist()
        })

    df_embedding = pd.DataFrame(rows)
    df_embedding.to_csv('news.csv', index=False)