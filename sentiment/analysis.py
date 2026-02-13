from transformers import pipeline
import torch

def analyze():
    classifier = pipeline("sentiment-analysis")
    return classifier("I've been waiting for a HuggingFace course my whole life.")