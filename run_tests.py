"""
Script to run test cases against the News Analysis API
Usage: python run_tests.py
"""

import requests
import json
from test_cases import test_cases, get_test_case_json
import time

API_URL = "http://localhost:8000/predict"

def test_api(test_case, index):
    """Test a single test case against the API"""
    print(f"\n{'='*80}")
    print(f"Testing Case #{index + 1}: {test_case['name']}")
    print(f"Category: {test_case['category'].upper()}")
    print(f"{'='*80}")
    print(f"Headline: {test_case['headline'][:80]}...")
    
    try:
        # Prepare request
        payload = get_test_case_json(test_case)
        
        # Make API request
        start_time = time.time()
        response = requests.post(API_URL, json=payload, timeout=30)
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract key predictions
            distilbert = result['distilbert_sentiment']
            entity = result['entity_overlap']
            fake_news = result['fake_news_detection']
            final_ensemble = result.get('final_ensemble', {})
            
            print(f"\n✅ Request successful ({elapsed_time:.2f}s)")
            print(f"\nResults:")
            print(f"  DistilBERT Sentiment:")
            print(f"    Headline: {distilbert['headline']['score']:.3f}")
            print(f"    Article: {distilbert['article']['score']:.3f}")
            print(f"    Difference: {distilbert['difference']:.3f}")
            
            print(f"\n  Entity Overlap:")
            print(f"    Prediction: {entity['prediction']}")
            print(f"    Score: {entity['score']:.3f}")
            print(f"    Token Overlap: {entity['features']['token_overlap']:.3f}")
            print(f"    Entity Overlap: {entity['features']['entity_overlap']:.3f}")
            
            print(f"\n  Fake News Detection:")
            print(f"    Result: {fake_news['result']}")
            print(f"    Confidence: {fake_news['score']:.3f}")
            
            # Expected vs Actual
            expected_misleading = test_case['category'] == 'misleading'
            entity_says_misleading = entity['prediction'] == 'Misleading'
            fake_news_says_misleading = 'misleading' in fake_news['result'].lower()
            
            # DistilBERT: Large difference suggests misleading (threshold: |difference| > 0.5)
            distilbert_difference = abs(distilbert['difference'])
            distilbert_says_misleading = distilbert_difference > 0.5
            
            # Final Ensemble prediction
            ensemble_says_misleading = final_ensemble.get('final_prediction', '') == 'Misleading' if final_ensemble else False
            
            print(f"\n  Expected: {'MISLEADING' if expected_misleading else 'NOT MISLEADING'}")
            print(f"  DistilBERT Model: {'MISLEADING' if distilbert_says_misleading else 'NOT MISLEADING'} (diff: {distilbert_difference:.3f})")
            print(f"  Entity Model: {'MISLEADING' if entity_says_misleading else 'NOT MISLEADING'}")
            print(f"  Fake News Model: {'MISLEADING' if fake_news_says_misleading else 'NOT MISLEADING'}")
            if final_ensemble:
                print(f"  Final Ensemble: {final_ensemble.get('final_prediction', 'N/A')} (score: {final_ensemble.get('final_score', 0):.3f})")
            
            return {
                'success': True,
                'test_case': test_case['name'],
                'category': test_case['category'],
                'expected_misleading': expected_misleading,
                'distilbert_prediction': distilbert_says_misleading,
                'entity_prediction': entity_says_misleading,
                'fake_news_prediction': fake_news_says_misleading,
                'ensemble_prediction': ensemble_says_misleading,
                'elapsed_time': elapsed_time
            }
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return {
                'success': False,
                'test_case': test_case['name'],
                'error': f"Status {response.status_code}: {response.text}"
            }
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error: Is the API server running at {API_URL}?")
        return {
            'success': False,
            'test_case': test_case['name'],
            'error': 'Connection error'
        }
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            'success': False,
            'test_case': test_case['name'],
            'error': str(e)
        }


def run_all_tests():
    """Run all test cases"""
    print("="*80)
    print("NEWS ANALYSIS API - TEST RUNNER")
    print("="*80)
    print(f"\nTesting API at: {API_URL}")
    print(f"Total test cases: {len(test_cases)}")
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code != 200:
            print("⚠️  Warning: Server responded but may not be ready")
    except:
        print("❌ ERROR: Cannot connect to API server!")
        print("   Make sure the server is running: python app.py")
        return
    
    results = []
    
    # Run all tests
    for i, test_case in enumerate(test_cases):
        result = test_api(test_case, i)
        results.append(result)
        time.sleep(0.5)  # Small delay between requests
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"\nTotal tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        avg_time = sum(r['elapsed_time'] for r in successful) / len(successful)
        print(f"Average response time: {avg_time:.2f}s")
        
        # Calculate accuracy for each model
        distilbert_correct = sum(1 for r in successful if r.get('distilbert_prediction') == r.get('expected_misleading'))
        entity_correct = sum(1 for r in successful if r.get('entity_prediction') == r.get('expected_misleading'))
        fake_news_correct = sum(1 for r in successful if r.get('fake_news_prediction') == r.get('expected_misleading'))
        ensemble_correct = sum(1 for r in successful if r.get('ensemble_prediction') == r.get('expected_misleading'))
        
        distilbert_accuracy = (distilbert_correct / len(successful)) * 100
        entity_accuracy = (entity_correct / len(successful)) * 100
        fake_news_accuracy = (fake_news_correct / len(successful)) * 100
        ensemble_accuracy = (ensemble_correct / len(successful)) * 100
        
        print(f"\n{'='*80}")
        print("MODEL ACCURACY")
        print(f"{'='*80}")
        print(f"DistilBERT Sentiment Model: {distilbert_correct}/{len(successful)} = {distilbert_accuracy:.2f}%")
        print(f"Entity Overlap Model: {entity_correct}/{len(successful)} = {entity_accuracy:.2f}%")
        print(f"Fake News Detection Model: {fake_news_correct}/{len(successful)} = {fake_news_accuracy:.2f}%")
        print(f"Final Weighted Ensemble: {ensemble_correct}/{len(successful)} = {ensemble_accuracy:.2f}%")
        print(f"{'='*80}")
        
        # Category breakdown
        misleading_tests = [r for r in successful if r.get('category') == 'misleading']
        non_misleading_tests = [r for r in successful if r.get('category') == 'non-misleading']
        
        print(f"\nTest case breakdown:")
        print(f"  Misleading test cases: {len(misleading_tests)}")
        print(f"  Non-misleading test cases: {len(non_misleading_tests)}")
    
    if failed:
        print(f"\nFailed tests:")
        for r in failed:
            print(f"  - {r['test_case']}: {r.get('error', 'Unknown error')}")
    
    print("\n" + "="*80)


def run_single_test(index):
    """Run a single test case by index"""
    if index < 0 or index >= len(test_cases):
        print(f"Invalid test case index. Must be between 0 and {len(test_cases) - 1}")
        return
    
    test_case = test_cases[index]
    test_api(test_case, index)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Run specific test case
        try:
            index = int(sys.argv[1])
            run_single_test(index)
        except ValueError:
            print("Invalid argument. Use an integer index or no argument to run all tests.")
    else:
        # Run all tests
        run_all_tests()

