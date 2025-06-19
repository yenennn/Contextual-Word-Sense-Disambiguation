import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from training import ContextAwareWSDModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path):
    checkpoint = torch.load(model_path, weights_only=False)

    num_labels = len(checkpoint['sense_id_to_label'])
    model = ContextAwareWSDModel(num_labels=num_labels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint['sense_id_to_label'], checkpoint['label_to_sense_id'], checkpoint['sense_id_to_meaning']


def predict_sense(sentence, target_word, model, tokenizer, label_to_sense_id, sense_id_to_meaning):
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

def visualize_attention(sentence, target_word, model, tokenizer, device):
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
    model, sense_id_to_label, label_to_sense_id, sense_id_to_meaning = load_model('context_aware_wsd_model.pt')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    while True:
        sentence = input("Enter a sentence (or 'q' to quit): ")
        if sentence.lower() == 'q':
            break

        target_word = input("Enter the target word: ")

        prediction = predict_sense(sentence, target_word, model, tokenizer, label_to_sense_id, sense_id_to_meaning)
        print("\nPrediction for new sentence:")
        print(f"Sentence: '{sentence}'")
        print(f"Target word: '{target_word}'")
        print(f"Predicted label: {prediction['label']}")
        print(f"Predicted sense ID: {prediction['sense_id']}")
        print(f"Predicted meaning: '{prediction['meaning']}'")


        visualize = input("Visualize attention? (y/n): ")
        if visualize.lower() == 'y':
            attention_analysis = visualize_attention(sentence, target_word, model, tokenizer, device)
            print("\nTop 5 tokens by attention:")
            sorted_tokens = sorted(attention_analysis, key=lambda x: x[1], reverse=True)[:5]
            for token, weight in sorted_tokens:
                print(f"{token}: {weight:.4f}")