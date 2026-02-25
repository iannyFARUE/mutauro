from transformers import AutoTokenizer

def load_tokenizer(checkpoint = "distilbert-base-uncased-finetuned-sst-2-english", raw_inputs = None):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    inputs = tokenizer(raw_inputs, padding=True, truncation=True,return_tensors="pt")
    return inputs

def load_model():

    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

    from transformers import AutoModelForSequenceClassification

    inputs = load_tokenizer(checkpoint, raw_inputs)

    
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    outputs = model(**inputs)
   
    outputs = model(**inputs)
    print(outputs.logits.shape)
    print(outputs.logits)

    import torch

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)
    print(model.config.id2label[torch.argmax(predictions[0]).item()])


if __name__ == "__main__":
    load_model()