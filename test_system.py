"""Test script to verify the system is working correctly"""

import logging
from genai_docs_helper.graph import graph

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_query():
    """Test a simple query through the system"""
    question = "What machine learning models did we use in this project?"
    
    print(f"\n{'='*60}")
    print(f"Testing query: {question}")
    print(f"{'='*60}\n")
    
    try:
        result = graph.invoke({"question": question, "original_question": question})
        
        print(f"\n{'='*60}")
        print("RESULTS:")
        print(f"{'='*60}")
        print(f"Final Answer: {result.get('generation', 'No answer generated')}")
        print(f"Documents Used: {len(result.get('documents', []))}")
        print(f"Confidence Score: {result.get('confidence_score', 'N/A')}")
        print(f"Retry Count: {result.get('retry_count', 0)}")
        
        if result.get('error_log'):
            print(f"\nErrors encountered:")
            for error in result['error_log']:
                print(f"  - {error}")
                
    except Exception as e:
        print(f"Error running test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_query()
