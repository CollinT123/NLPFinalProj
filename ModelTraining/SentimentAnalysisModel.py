from datasets import load_dataset
from transformers import AutoTokenizer
from datasets import DatasetDict
from torch.utils.data import DataLoader
import torch
from transformers import AutoModelForSequenceClassification
from tqdm.auto import tqdm
from torch.optim import AdamW


#Load the IMDB dataset
imdb = load_dataset('imdb')

#used for making sure the dataset is set up
print(imdb)

#create a validation split from the original train set
#currently using 10% of the original training data as validation

imdb_train_valid = imdb['train'].train_test_split(
    test_size=0.1, #10% for validation
    shuffle=True, #shuffle before splitting
    seed=42 #for reproducibility
)

#databaseDict to split up training data
imdb_splits = DatasetDict({
    'train': imdb_train_valid["train"], #22500 examples
    'valid': imdb_train_valid["test"], #2500 examples
    'test': imdb["test"] #25000 examples
})

#Quick sanity checks
print("Train size:", len(imdb_splits["train"]))
print("Valid size:", len(imdb_splits["valid"]))
print("Test size:", len(imdb_splits["test"]))

#example training item
print("\nExample train item:")
print(imdb_splits["train"][0])

#max sequence length
max_Length = 512

#Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

#tokenize the dataset
def tokenize_function(batch):
    encoded = tokenizer(
        batch["text"],
        padding = "max_length",
        truncation = True,
        max_length=max_Length,
    )

    encoded["labels"] = batch["label"]
    return encoded

#create a new variable for the tokenized dataset
tokenized_imdb = imdb_splits.map(
    tokenize_function,
    batched = True,
)

#remove extra column labels
tokenized_imdb = tokenized_imdb.remove_columns(["text", "label"])

#format into pyTorch Sensors
tokenized_imdb.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

example = tokenized_imdb["train"][0]
print(example)


train_batch_size = 16
eval_batch_size = 32

#dataloaders for each split
train_loader = DataLoader(
    tokenized_imdb["train"],
    batch_size=train_batch_size,
    shuffle=True  # important for training
)

valid_loader = DataLoader(
    tokenized_imdb["valid"],
    batch_size=eval_batch_size,
    shuffle=False
)

test_loader = DataLoader(
    tokenized_imdb["test"],
    batch_size=eval_batch_size,
    shuffle=False
)

batch = next(iter(train_loader))

print(batch.keys())          # should be dict_keys(['input_ids', 'attention_mask', 'labels'])
print(batch["input_ids"].shape)      # e.g., torch.Size([16, 256])
print(batch["attention_mask"].shape) # same shape
print(batch["labels"].shape)

#check for good GPU and if not then train on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

#loads a pretrained model from distilBert
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2 # num possible classifications
)

#states where to move the model
model.to(device)

#number of epochs to start
num_epochs = 3

#Create optimizer
learning_rate = 2e-5 #how much do new passes effect weight
optimizer = AdamW(model.parameters(), lr=learning_rate) #

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 30)
    # ---- Training ----
    model.train()
    total_train_loss = 0
    total_train_correct = 0
    total_train_examples = 0
    from torch.optim import AdamW


    for batch in tqdm(train_loader, desc="Training"):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass (model computes loss when labels are provided)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        total_train_loss += loss.item() * input_ids.size(0)

        # Track accuracy
        preds = torch.argmax(logits, dim=-1)
        total_train_correct += (preds == labels).sum().item()
        total_train_examples += labels.size(0)
    avg_train_loss = total_train_loss / total_train_examples
    train_accuracy = total_train_correct / total_train_examples

    print(f"Train loss: {avg_train_loss:.4f} | Train acc: {train_accuracy:.4f}")

model.eval()
total_test_correct = 0
total_test_examples = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        total_test_correct += (preds == labels).sum().item()
        total_test_examples += labels.size(0)

test_accuracy = total_test_correct / total_test_examples
print(f"Test accuracy: {test_accuracy:.4f}")

model.save_pretrained("distilbert-imdb-sentiment")
tokenizer.save_pretrained("distilbert-imdb-sentiment")






