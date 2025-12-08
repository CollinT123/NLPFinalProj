# News Analysis API

A comprehensive FastAPI-based service that analyzes news articles using three different machine learning models to provide sentiment analysis, entity overlap detection, and fake news classification.

## 🚀 Features

- **Multi-Model Analysis**: Combines three different ML models for comprehensive news analysis
- **Sentiment Analysis**: Uses DistilBERT to analyze sentiment of headlines and articles separately
- **Entity Overlap Detection**: Identifies misleading content by analyzing entity and token overlap between headlines and articles
- **Fake News Detection**: Leverages a HuggingFace model to detect potentially misleading headlines
- **RESTful API**: Clean, well-documented FastAPI endpoint
- **Auto-Generated Documentation**: Interactive API docs available at `/docs`

## 📋 Models Used

### 1. DistilBERT Sentiment Analysis
- **Model**: Fine-tuned DistilBERT for 3-class financial sentiment classification
- **Location**: `Models/distilbert-imdb-financial-3class/`
- **Purpose**: Analyzes sentiment (negative, neutral, positive) of headlines and articles separately
- **Output**: 
  - Probability scores for each class (p_neg, p_neu, p_pos)
  - Sentiment score (p_pos - p_neg) in range [-1, 1]
  - Squared scores (only for DistilBERT model) to emphasize differences

### 2. Entity Overlap Model
- **Model**: Custom PyTorch MLP (`EntityMLP`)
- **Location**: `Models/entity_overlap_model.pt`
- **Purpose**: Detects misleading content by analyzing overlap between headlines and articles
- **Features**:
  - Token overlap ratio
  - Entity overlap ratio
  - Cosine similarity (TF-IDF)
  - Number of entities in headline
  - Number of entities in article
- **Output**: Binary classification (Misleading / Not Misleading) with confidence score

### 3. Fake News Detection
- **Model**: `willphan1712/fake_news_detection` from HuggingFace
- **Purpose**: Classifies whether a headline is misleading or legitimate
- **Output**: Label (LABEL_0 or LABEL_1) with confidence score and human-readable result

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip or pip3

### Step 1: Clone or Navigate to Project Directory
```bash
cd /path/to/NLPFinalProj/NLPFinalProj
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: The first run will download the fake news detection model from HuggingFace, which may take a few minutes.

## 🚀 Usage

### Starting the Server

```bash
# Activate virtual environment (if not already activated)
source venv/bin/activate

# Run the server
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

### API Endpoints

#### GET `/`
Health check endpoint that returns a welcome message.

**Response:**
```json
{
  "message": "News Analysis API - Use POST /predict to analyze news"
}
```

#### POST `/predict`
Main endpoint for news analysis. Accepts a headline and article, processes them through all three models, and returns comprehensive results.

**Request Body:**
```json
{
  "headline": "Your news headline here",
  "article": "The full article text goes here..."
}
```

**Response:**
```json
{
  "distilbert_sentiment": {
    "headline": {
      "p_neg": 0.94,
      "p_neu": 0.04,
      "p_pos": 0.01,
      "score": -0.93,
      "squared_score": -0.86
    },
    "article": {
      "p_neg": 0.01,
      "p_neu": 0.32,
      "p_pos": 0.67,
      "score": 0.66,
      "squared_score": 0.43
    },
    "difference": -1.59,
    "difference_squared": -1.29
  },
  "entity_overlap": {
    "score": 0.44,
    "prediction": "Misleading",
    "features": {
      "token_overlap": 0.29,
      "entity_overlap": 0.0,
      "cosine_tfidf": 0.10,
      "num_entities_headline": 2,
      "num_entities_body": 2
    }
  },
  "fake_news_detection": {
    "label": "LABEL_1",
    "score": 0.99,
    "result": "This is a good heading"
  }
}
```

### Example Usage

#### Using cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "headline": "Experts Warn of Severe Economic Trouble Ahead",
    "article": "Financial analysts are expressing concern about the upcoming economic downturn. Market indicators suggest significant challenges ahead for global markets."
  }'
```

#### Using Python
```python
import requests

url = "http://localhost:8000/predict"
data = {
    "headline": "Experts Warn of Severe Economic Trouble Ahead",
    "article": "Financial analysts are expressing concern about the upcoming economic downturn."
}

response = requests.post(url, json=data)
result = response.json()
print(result)
```

#### Using JavaScript/Node.js
```javascript
const fetch = require('node-fetch');

const url = 'http://localhost:8000/predict';
const data = {
  headline: "Experts Warn of Severe Economic Trouble Ahead",
  article: "Financial analysts are expressing concern about the upcoming economic downturn."
};

fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(data)
})
  .then(res => res.json())
  .then(data => console.log(data));
```

## 📚 API Documentation

FastAPI automatically generates interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These interfaces allow you to test the API directly from your browser.

## 📁 Project Structure

```
NLPFinalProj/
├── Models/
│   ├── distilbert-imdb-financial-3class/
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── ...
│   └── entity_overlap_model.pt
├── app.py                 # Main FastAPI application
├── model.py               # EntityMLP model and helper functions
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── venv/                 # Virtual environment (created during setup)
```

## 🔧 Configuration

### Device Configuration
The application automatically detects and uses:
- **CUDA** (if available) for GPU acceleration
- **CPU** (fallback) for CPU-only execution
- **MPS** (on Apple Silicon) for Metal Performance Shaders

### Model Loading
All models are loaded at application startup using FastAPI's lifespan events. This ensures:
- Models are ready before accepting requests
- No per-request model loading overhead
- Proper resource cleanup on shutdown

## 📊 Response Fields Explained

### DistilBERT Sentiment
- **p_neg, p_neu, p_pos**: Probability scores for negative, neutral, and positive sentiment
- **score**: Sentiment score calculated as `p_pos - p_neg` (range: -1 to 1)
- **squared_score**: Squared version of the score (negative values remain negative)
- **difference**: Difference between headline and article sentiment scores
- **difference_squared**: Squared difference between headline and article scores

### Entity Overlap
- **score**: Confidence score (0 to 1) from the MLP model
- **prediction**: Binary classification ("Misleading" if score < 0.5, "Not Misleading" otherwise)
- **features**: 
  - `token_overlap`: Ratio of overlapping tokens between headline and article
  - `entity_overlap`: Ratio of overlapping named entities
  - `cosine_tfidf`: Cosine similarity using TF-IDF vectors
  - `num_entities_headline`: Count of entities in headline
  - `num_entities_body`: Count of entities in article

### Fake News Detection
- **label**: Model output label (LABEL_0 or LABEL_1)
- **score**: Confidence score (0 to 1)
- **result**: Human-readable interpretation ("This is a good heading" or "This is a misleading heading")

## ⚠️ Troubleshooting

### Model Loading Errors

**Issue**: `RuntimeError: Error(s) in loading state_dict for EntityMLP`
- **Solution**: Ensure the `EntityMLP` architecture in `model.py` matches the saved model structure

**Issue**: `ImportError: cannot import name 'split_torch_state_dict_into_shards'`
- **Solution**: Update dependencies: `pip install --upgrade huggingface-hub accelerate transformers`

### Port Already in Use

**Issue**: `Address already in use`
- **Solution**: Change the port in `app.py` or kill the process using port 8000:
  ```bash
  lsof -ti:8000 | xargs kill
  ```

### Memory Issues

**Issue**: Out of memory errors when loading models
- **Solution**: 
  - Ensure sufficient RAM/VRAM
  - Models are loaded once at startup, so initial memory usage is higher
  - Consider using CPU mode if GPU memory is limited

### HuggingFace Model Download

**Issue**: Slow or failed model downloads
- **Solution**: 
  - Ensure stable internet connection
  - First download may take several minutes
  - Models are cached after first download

## 🧪 Testing

### Quick Test
Test the API with a simple request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "headline": "Test Headline",
    "article": "This is a test article for the news analysis API."
  }'
```

### Comprehensive Test Suite

The project includes a comprehensive test suite with 18 test cases covering:
- **8 Misleading headlines** - Headlines that don't accurately represent the article content
- **9 Non-misleading headlines** - Headlines that accurately represent the article content
- **4 Edge cases** - Short headlines, long headlines, neutral sentiment, etc.

#### View Test Cases
```bash
python test_cases.py
```

#### Run All Tests
```bash
# Make sure the API server is running first
python app.py  # In one terminal

# Then run tests in another terminal
python run_tests.py
```

#### Run Single Test Case
```bash
python run_tests.py 0  # Run test case #0
```

The test runner will:
- Send each test case to the API
- Display results from all three models
- Show expected vs actual predictions
- Provide a summary of all test results

## 📝 Dependencies

Key dependencies include:
- **FastAPI**: Modern web framework for building APIs
- **Transformers**: HuggingFace library for transformer models
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning utilities (TF-IDF)
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation

See `requirements.txt` for the complete list with versions.

## 🔒 Notes

- The DistilBERT model uses **squared scores** to emphasize sentiment differences (this is specific to this model only)
- Entity extraction uses a simple regex-based approach; for production, consider using spaCy or a dedicated NER model
- The fake news detection model truncates input to 512 tokens
- All models run inference in evaluation mode (`model.eval()`)

## 📄 License

This project uses pre-trained models from various sources:
- DistilBERT model: Custom fine-tuned model
- Entity Overlap Model: Custom PyTorch model
- Fake News Detection: `willphan1712/fake_news_detection` from HuggingFace

Please ensure you comply with the licenses of these models when using this project.

## 🤝 Contributing

To contribute:
1. Ensure all models are properly loaded
2. Test with various headline/article combinations
3. Verify response format matches the expected schema

## 📞 Support

For issues or questions:
- Check the API documentation at `/docs`
- Review the troubleshooting section above
- Ensure all dependencies are correctly installed

---

**Built with FastAPI, PyTorch, and Transformers**

