import { Button } from '@willphan1712000/frontend';
import { useState } from 'react';
import './App.css';
import styles from './styles.module.css';
import axios from 'axios';

const url = 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: url,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

interface AnalysisResult {
  distilbert_sentiment: {
    headline: { p_neg: number; p_neu: number; p_pos: number; score: number };
    article: { p_neg: number; p_neu: number; p_pos: number; score: number };
    difference: number;
  };
  entity_overlap: {
    score: number;
    prediction: string;
    features: {
      token_overlap: number;
      entity_overlap: number;
      cosine_tfidf: number;
      num_entities_headline: number;
      num_entities_body: number;
    };
  };
  fake_news_detection: {
    label: string;
    score: number;
    result: string;
  };

  final_ensemble: {
    final_score: number;
    final_prediction: string;
    individual_scores: {
      distilbert_prob: number;
      entity_prob: number;
      fake_prob: number;
    };
  };
}


function App() {
  const [isAnalyze, setAnalyze] = useState<boolean>(false)
  const [heading, setHeading] = useState<string>('')
  const [body, setBody] = useState<string>('')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [error, setError] = useState<string>('')

  const analyzeStart = async () => {
    setAnalyze(true)
    setError('')
    setResult(null)

    try {
      const res = await apiClient.post('/predict', {
        headline: heading,
        article: body
      })

      setResult(res.data)
    } catch (error: any) {
      console.error('Error: ', error)
      setError(error.response?.data?.detail || error.message || 'An error occurred')
    }

    setAnalyze(false)
  }

  return (
    <div className="App">
      <h1 className={styles.heading}>Misleading Headline Analysis</h1>
      <div className={styles.container}>
        <form className={styles.form}>
          <div className={styles.formArea}>
            <label htmlFor="heading" className={styles.label}>Heading</label>
            <input type="text" id="heading" className={`${styles.input} ${styles.inputBox}`} value={heading} onChange={e => setHeading(e.target.value)} />
          </div>
          <div className={styles.formArea}>
            <label htmlFor='body' className={styles.label}>Body</label>
            <textarea id='body' className={`${styles.input} ${styles.textArea}`} onChange={e => setBody(e.target.value)} value={body} />
          </div>
          <Button 
            buttonType='gradient'
            content={isAnalyze ? 'Analyzing...': 'Analyze'}
            onClick={e => {
              e.preventDefault()
              analyzeStart()
            }}
            isLoading={isAnalyze}
          />
          {error && <p style={{color: 'red', marginTop: '1rem'}}>Error: {error}</p>}
        {result && (
  <div style={{ marginTop: "2rem" }}>
    <h3 className={styles.resultLabel}>Results</h3>

    {/* 1. DistilBERT */}
    <div style={{ marginTop: "1rem", padding: "1rem", background: "#f5f5f5", borderRadius: "8px" }}>
      <h4>1. DistilBERT Sentiment Analysis</h4>
                <p><strong>Headline:</strong> Score: {result.distilbert_sentiment.headline.score.toFixed(3)}</p>
                <p><strong>Article:</strong> Score: {result.distilbert_sentiment.article.score.toFixed(3)}</p>
      <p><strong>Difference:</strong> {result.distilbert_sentiment.difference.toFixed(3)}</p>
    </div>

    {/* 2. Entity Overlap */}
    <div style={{ marginTop: "1rem", padding: "1rem", background: "#f5f5f5", borderRadius: "8px" }}>
      <h4>2. Entity Overlap Model</h4>
      <p><strong>Prediction:</strong> {result.entity_overlap.prediction}</p>
      <p><strong>Confidence Score:</strong> {result.entity_overlap.score.toFixed(3)}</p>
      <p><strong>Token Overlap:</strong> {result.entity_overlap.features.token_overlap.toFixed(3)}</p>
      <p><strong>Entity Overlap:</strong> {result.entity_overlap.features.entity_overlap.toFixed(3)}</p>
    </div>

    {/* 3. Fake News */}
    <div style={{ marginTop: "1rem", padding: "1rem", background: "#f5f5f5", borderRadius: "8px" }}>
      <h4>3. Fake News Detection</h4>
      <p><strong>Result:</strong> {result.fake_news_detection.result}</p>
      <p><strong>Confidence:</strong> {result.fake_news_detection.score.toFixed(3)}</p>
    </div>

    {/* 4. Ensemble */}
    <div style={{ marginTop: "1rem", padding: "1rem", background: "#e8f7ff", borderRadius: "8px" }}>
      <h4>4. Final Ensemble Prediction</h4>

      <p><strong>Final Prediction:</strong> {result.final_ensemble?.final_prediction}</p>
      <p><strong>Final Score:</strong> {result.final_ensemble?.final_score?.toFixed(3)}</p>

      <h5 style={{ marginTop: "0.5rem" }}>Individual Model Contributions</h5>

      <p><strong>DistilBERT Probability:</strong> {result.final_ensemble?.individual_scores?.distilbert_prob?.toFixed(3)}</p>
      <p><strong>Entity Model Probability:</strong> {result.final_ensemble?.individual_scores?.entity_prob?.toFixed(3)}</p>
      <p><strong>Fake News Probability:</strong> {result.final_ensemble?.individual_scores?.fake_prob?.toFixed(3)}</p>
    </div>
  </div>
)}

        </form>
      </div>
    </div>
  );
}

export default App;
