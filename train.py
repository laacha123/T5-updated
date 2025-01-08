import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from dataset import T5Dataset
from model import T5Model
from config import config

def train():
    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(config.model_name)
    model = T5Model(config.model_name)
    model = model.cuda() if torch.cuda.is_available() else model

    # loading dataset and DataLoader
    source_texts = [""] # replace with english sentences 
    target_texts = [""] # replace with french sentences
    train_dataset = T5Dataset(source_texts, target_texts, tokenizer, max_length=config.max_seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # Training 
    model.train()
    for epoch in range(config.num_epochs):
        for batch in train_dataloader:
            source_ids = batch["source_ids"].cuda() if torch.cuda.is_available() else batch["source_ids"]
            source_mask = batch["source_mask"].cuda() if torch.cuda.is_available() else batch["source_mask"]
            target_ids = batch["target_ids"].cuda() if torch.cuda.is_available() else batch["target_ids"]

            # Shift target_ids for teacher forcing
            decoder_input_ids = target_ids[:, :-1]
            labels = target_ids[:, 1:].contiguous()

            # Forward pass
            outputs = model(input_ids=source_ids, attention_mask=source_mask, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits

            # Loss computation
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {loss.item()}")

if __name__ == "__main__":
    train()
