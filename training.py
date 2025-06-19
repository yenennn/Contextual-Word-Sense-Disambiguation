import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ContextAwareWSDDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        word = row['word']
        sentence = row['sentence']

        marked_sentence = f"{word} [SEP] {sentence}"

        encoding = self.tokenizer(
            marked_sentence,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].squeeze()
        sep_indices = (input_ids == self.tokenizer.sep_token_id).nonzero().flatten()
        if len(sep_indices) == 0:
            sep_pos = 0
        else:
            sep_pos = sep_indices[0].item()

        word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
        target_positions = []

        for i in range(sep_pos + 1, len(input_ids) - len(word_tokens) + 1):
            if input_ids[i:i + len(word_tokens)].tolist() == word_tokens:
                target_positions.extend(list(range(i, i + len(word_tokens))))

        if not target_positions:
            target_positions = [sep_pos + 1]

        target_mask = torch.zeros_like(input_ids)
        for pos in target_positions:
            if pos < self.max_len:
                target_mask[pos] = 1
        max_positions = 5
        padded_positions = target_positions[:max_positions] + [0] * (max_positions - len(target_positions))
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'target_mask': target_mask,
            'target_positions': torch.tensor(padded_positions[:max_positions], dtype=torch.long),
            'labels': torch.tensor(row['sense_id'], dtype=torch.long)
        }

class ContextAwareWSDModel(torch.nn.Module):
    def __init__(self, num_labels, hidden_size=768):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.attention = torch.nn.Linear(hidden_size, 1)
        self.dropout = torch.nn.Dropout(0.1)

        self.context_fusion = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.classifier = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, target_mask=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        last_hidden_state = outputs.last_hidden_state

        if target_mask is not None:
            expanded_mask = target_mask.unsqueeze(-1)
            masked_embeddings = last_hidden_state * expanded_mask

            target_embedding_sum = torch.sum(masked_embeddings, dim=1)

            mask_sum = torch.sum(target_mask, dim=1, keepdim=True) + 1e-10

            target_embedding = target_embedding_sum / mask_sum
        else:
            target_embedding = last_hidden_state[:, 0]

        attention_scores = self.attention(last_hidden_state).squeeze(-1)

        attention_scores = attention_scores.masked_fill(~attention_mask.bool(), -10000.0)

        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)

        context_embedding = torch.sum(last_hidden_state * attention_weights, dim=1)

        combined_embedding = torch.cat([target_embedding, context_embedding], dim=1)

        fused_embedding = self.context_fusion(combined_embedding)
        fused_embedding = torch.tanh(fused_embedding)
        fused_embedding = self.dropout(fused_embedding)

        logits = self.classifier(fused_embedding)

        return logits


def train_model(model, train_loader, test_loader, epochs=5, lr=2e-5):
    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != 'target_positions'}

            optimizer.zero_grad()

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                target_mask=batch['target_mask']
            )

            loss = criterion(outputs, batch['labels'])
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items() if k != 'target_positions'}

                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    target_mask=batch['target_mask']
                )

                loss = criterion(outputs, batch['labels'])
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += batch['labels'].size(0)
                correct += (predicted == batch['labels']).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        accuracy = 100 * correct / total

        test_losses.append(avg_test_loss)
        test_accuracies.append(accuracy)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")

        if epoch == epochs - 1:
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds))

            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            plt.savefig('confusion_matrix.png')

    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }

def predict_sense(sentence, target_word, model, tokenizer):
    model.eval()

    marked_sentence = f"{target_word} [SEP] {sentence}"
    encoding = tokenizer(
        marked_sentence,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding['input_ids'][0]
    sep_indices = (input_ids == tokenizer.sep_token_id).nonzero().flatten()
    if len(sep_indices) == 0:
        sep_pos = 0
    else:
        sep_pos = sep_indices[0].item()
    word_tokens = tokenizer.encode(target_word, add_special_tokens=False)
    target_positions = []

    for i in range(sep_pos + 1, len(input_ids) - len(word_tokens) + 1):
        if input_ids[i:i + len(word_tokens)].tolist() == word_tokens:
            target_positions.extend(list(range(i, i + len(word_tokens))))

    if not target_positions:
        target_positions = [sep_pos + 1]

    target_mask = torch.zeros_like(input_ids)
    for pos in target_positions:
        if pos < 128:
            target_mask[pos] = 1

    input_ids = input_ids.to(device)
    attention_mask = encoding['attention_mask'][0].to(device)
    target_mask = target_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            target_mask=target_mask.unsqueeze(0)
        )

    predicted_label = torch.argmax(outputs, dim=1).item()
    predicted_composite_id = label_to_sense_id[predicted_label]
    predicted_meaning = sense_id_to_meaning.get(predicted_composite_id, "Unknown meaning")

    word, sense_id = predicted_composite_id.split('_')

    return {
        'label': predicted_label,
        'sense_id': sense_id,
        'composite_id': predicted_composite_id,
        'meaning': predicted_meaning
    }

def visualize_attention(sentence, target_word, model, tokenizer):
    model.eval()

    marked_sentence = f"{target_word} [SEP] {sentence}"
    encoding = tokenizer(
        marked_sentence,
        padding=False,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

    last_hidden_state = outputs.last_hidden_state
    attention_scores = model.attention(last_hidden_state).squeeze(-1)
    attention_scores = attention_scores.masked_fill(~attention_mask.bool(), -10000.0)
    attention_weights = torch.softmax(attention_scores, dim=1)[0].cpu().detach().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    plt.figure(figsize=(12, 6))
    sns.barplot(x=np.arange(len(tokens)), y=attention_weights)
    plt.xticks(np.arange(len(tokens)), tokens, rotation=90)
    plt.title(f'Attention Weights for "{target_word}" in "{sentence}"')
    plt.tight_layout()
    plt.savefig('attention_visualization.png')

    return list(zip(tokens, attention_weights))

if __name__ == "__main__":
    df = pd.read_csv('labeled_sentences.csv')
    df = df.rename(columns={'Kelime': 'word', 'Anlam No': 'sense_id', 'Örnek Cümle': 'sentence'})
    print(f"Original sense IDs: {df['sense_id'].unique()}")

    df['composite_id'] = df['word'] + '_' + df['sense_id'].astype(str)

    sense_id_to_label = {id: i for i, id in enumerate(sorted(df['composite_id'].unique()))}
    label_to_sense_id = {i: id for id, i in sense_id_to_label.items()}
    sense_id_to_meaning = dict(zip(df['composite_id'], df['Anlam']))

    df['original_sense_id'] = df['sense_id']
    df['sense_id'] = df['composite_id'].map(sense_id_to_label)

    print(f"Remapped sense IDs: {df['sense_id'].unique()}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sense_id'])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = ContextAwareWSDDataset(train_df, tokenizer)
    test_dataset = ContextAwareWSDDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = df['sense_id'].nunique()
    model = ContextAwareWSDModel(num_labels=num_labels)
    model.to(device)
    print("Starting training...")
    history = train_model(model, train_loader, test_loader, epochs=5)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Training Loss')
    plt.plot(history['test_losses'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['test_accuracies'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')

    torch.save({
        'model_state_dict': model.state_dict(),
        'sense_id_to_label': sense_id_to_label,
        'label_to_sense_id': label_to_sense_id,
        'sense_id_to_meaning': sense_id_to_meaning
    }, 'context_aware_wsd_model.pt')

    example_sentence = "bu işletmenin çalışanları çok asık yüzlü."
    target_word = "yüz"

    prediction = predict_sense(example_sentence, target_word, model, tokenizer)
    print("\nPrediction for new sentence:")
    print(f"Sentence: '{example_sentence}'")
    print(f"Target word: '{target_word}'")
    print(f"Predicted label: {prediction['label']}")
    print(f"Predicted sense ID: {prediction['sense_id']}")
    print(f"Predicted meaning: '{prediction['meaning']}'")
    attention_analysis = visualize_attention(example_sentence, target_word, model, tokenizer)
    print("\nToken attention analysis:")
    for token, weight in attention_analysis:
        print(f"{token}: {weight:.4f}")