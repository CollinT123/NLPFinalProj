"""
Test script for DistilBERT Sentiment Model Accuracy
Tests the model's ability to correctly classify sentiment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def square_keep_negative(n):
    """Square a number but keep negative sign if original was negative."""
    if n < 0:
        return -(n * n)
    else:
        return n * n

def get_sentiment_score(text: str, tokenizer, model, max_length: int = 512):
    """Get sentiment score from DistilBERT model."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

    p_neg = probs[0].item()
    p_neu = probs[1].item()
    p_pos = probs[2].item()
    
    # Square p_neg and p_pos using square_keep_negative before computing score
    p_neg_squared = square_keep_negative(p_neg)
    p_pos_squared = square_keep_negative(p_pos)
    score = p_pos_squared - p_neg_squared

    return {
        "p_neg": p_neg,
        "p_neu": p_neu,
        "p_pos": p_pos,
        "score": score,
        "predicted_class": torch.argmax(probs).item()  # 0=neg, 1=neu, 2=pos
    }

def test_sentiment_accuracy():
    """Test the sentiment model with sample data."""
    
    # Load model
    print("Loading DistilBERT model...")
    model_path = "Models/distilbert-imdb-financial-3class"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    print("Model loaded successfully.\n")
    
    # Test cases with expected sentiment
    # Format: (text, expected_sentiment_label)
    # 0 = negative, 1 = neutral, 2 = positive
    test_cases = [
        # Negative examples
        ("The stock market crashed today, causing massive losses for investors.", 0),
        ("Company reports terrible earnings, stock price plummets.", 0),
        ("Economic downturn leads to widespread unemployment.", 0),
        ("Investors panic as market volatility increases dramatically.", 0),
        
        # Neutral examples
        ("The company reported quarterly earnings today.", 1),
        ("Market indices remained stable throughout the trading session.", 1),
        ("Federal Reserve announces interest rate decision.", 1),
        ("Economic data shows moderate growth in Q3.", 1),
        
        # Positive examples
        ("Stock market surges to record highs on strong earnings.", 2),
        ("Company announces breakthrough innovation, shares soar.", 2),
        ("Economic recovery accelerates with strong job growth.", 2),
        ("Investors celebrate as profits exceed expectations.", 2),
    ]
    
    print("=" * 80)
    print("DISTILBERT SENTIMENT MODEL ACCURACY TEST")
    print("=" * 80)
    print(f"Total test cases: {len(test_cases)}\n")
    
    correct_predictions = 0
    results = []
    
    for i, (text, expected_label) in enumerate(test_cases, 1):
        result = get_sentiment_score(text, tokenizer, model)
        predicted_label = result["predicted_class"]
        is_correct = predicted_label == expected_label
        
        if is_correct:
            correct_predictions += 1
        
        label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
        results.append({
            "text": text[:60] + "..." if len(text) > 60 else text,
            "expected": label_names[expected_label],
            "predicted": label_names[predicted_label],
            "correct": is_correct,
            "confidence": max(result["p_neg"], result["p_neu"], result["p_pos"])
        })
        
        status = "✓" if is_correct else "✗"
        print(f"{status} Test {i}: Expected {label_names[expected_label]}, "
              f"Predicted {label_names[predicted_label]} "
              f"(Confidence: {max(result['p_neg'], result['p_neu'], result['p_pos']):.3f})")
    
    accuracy = (correct_predictions / len(test_cases)) * 100
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Correct predictions: {correct_predictions}/{len(test_cases)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 80)
    
    # Detailed breakdown
    print("\nDETAILED BREAKDOWN:")
    print("-" * 80)
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"{status} {r['text']}")
        print(f"   Expected: {r['expected']}, Predicted: {r['predicted']} "
              f"(Confidence: {r['confidence']:.3f})")
        print()
    
    return accuracy, results

if __name__ == "__main__":
    test_sentiment_accuracy()

