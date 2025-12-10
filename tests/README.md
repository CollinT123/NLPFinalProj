# Test Suite

This directory contains test scripts for evaluating model accuracy and performance.

## Tests Available

### `test_sentiment_accuracy.py`
Tests the DistilBERT sentiment model's accuracy on sample financial text.

**Usage:**
```bash
cd tests
python test_sentiment_accuracy.py
```

**What it tests:**
- Classification accuracy (negative/neutral/positive)
- Confidence scores
- Model predictions vs expected labels

**Test cases include:**
- Negative sentiment examples (market crashes, losses, downturns)
- Neutral sentiment examples (reports, announcements, data)
- Positive sentiment examples (surges, growth, profits)

## Running All Tests

```bash
# From project root
python -m pytest tests/  # If using pytest
# Or run individual test files
python tests/test_sentiment_accuracy.py
```

