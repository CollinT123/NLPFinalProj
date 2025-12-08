import torch
import torch.nn as nn
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Entity extraction using simple regex-based approach
def extract_entities(text):
    """
    Extract named entities from text using simple pattern matching.
    This is a simplified version - in production you might use spaCy or NER models.
    """
    # Common patterns for entities
    entities = []
    
    # Capitalized words/phrases (simple heuristic)
    # Match sequences of capitalized words
    pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
    matches = re.findall(pattern, text)
    
    # Filter out common words that aren't entities
    common_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 
                   'And', 'Or', 'But', 'If', 'When', 'Where', 'Why', 'How'}
    
    for match in matches:
        if match not in common_words and len(match) > 2:
            entities.append(match)
    
    return list(set(entities))  # Return unique entities

def token_overlap(headline, body):
    """
    Calculate token overlap between headline and body.
    Returns the ratio of overlapping tokens.
    """
    # Simple tokenization (split on whitespace and lowercase)
    headline_tokens = set(word.lower() for word in headline.split())
    body_tokens = set(word.lower() for word in body.split())
    
    if len(headline_tokens) == 0:
        return 0.0
    
    overlap = len(headline_tokens & body_tokens)
    return overlap / len(headline_tokens)

def entity_overlap(headline, body):
    """
    Calculate entity overlap between headline and body.
    Returns the ratio of overlapping entities.
    """
    headline_entities = set(extract_entities(headline))
    body_entities = set(extract_entities(body))
    
    if len(headline_entities) == 0:
        return 0.0
    
    overlap = len(headline_entities & body_entities)
    return overlap / len(headline_entities) if len(headline_entities) > 0 else 0.0

def cosine_tfidf(headline, body):
    """
    Calculate cosine similarity using TF-IDF vectors.
    """
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([headline, body])
        similarity = (tfidf_matrix[0] * tfidf_matrix[1].T).toarray()[0][0]
        return float(similarity)
    except:
        return 0.0

class EntityMLP(nn.Module):
    """
    Multi-layer perceptron for entity overlap classification.
    Takes 5 features: token_overlap, entity_overlap, cosine_tfidf, 
    num_entities_headline, num_entities_body
    Based on saved model: Linear(5,32) -> ReLU -> (layer2) -> Linear(32,16) -> ReLU -> Linear(16,1) -> Sigmoid
    """
    def __init__(self, input_size=5, hidden_size1=32, hidden_size2=16, output_size=1):
        super(EntityMLP, self).__init__()
        # Structure matches saved model: layers.0, layers.3, layers.5 are Linear layers
        # layers.1, layers.2, layers.4 are activation layers (no params)
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size1),  # layers.0: (5, 32)
            nn.ReLU(),                            # layers.1: no params
            nn.ReLU(),                            # layers.2: no params (or could be Dropout/BatchNorm)
            nn.Linear(hidden_size1, hidden_size2), # layers.3: (32, 16)
            nn.ReLU(),                            # layers.4: no params
            nn.Linear(hidden_size2, output_size), # layers.5: (16, 1)
            nn.Sigmoid()                          # layers.6: no params
        )
    
    def forward(self, x):
        return self.layers(x)

