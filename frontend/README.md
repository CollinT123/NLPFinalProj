# News Analysis Frontend

React/TypeScript frontend for the News Analysis API.

## Prerequisites

- Node.js 14 or higher
- npm (comes with Node.js)

## Installation

### Step 1: Navigate to Frontend Directory
```bash
cd frontend
```

### Step 2: Install Dependencies
```bash
npm install
```

## Running the Application

### Development Mode
```bash
npm start
```

The application will start on `http://localhost:3000` and automatically open in your browser.

**Note**: Make sure the backend API server is running on `http://localhost:8000` before using the frontend.

## Features

- Input form for headline and article text
- Real-time analysis using all three ML models
- Displays results from:
  - DistilBERT Sentiment Analysis
  - Entity Overlap Model
  - Fake News Detection Model
  - Final Weighted Ensemble Prediction

## Build for Production

```bash
npm run build
```

This creates an optimized production build in the `build/` folder.
