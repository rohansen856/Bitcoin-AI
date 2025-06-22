from deepeval.test_case import LLMTestCase
from .models import LLaMA31Model
import json
import os

llama31 = LLaMA31Model()

# Load the test cases from the JSON file
def load_test_cases_from_json(file_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, file_path)
    with open(full_path, 'r') as file:
        data = json.load(file)
    return data

# Load the questions and expected answers
questions_data = load_test_cases_from_json('qna.json')

# Create test cases dynamically
test_cases = []

def get_test_cases():    
    # Add test cases from JSON data
    for item in questions_data:
        test_cases.append(
            LLMTestCase(
                input=item["prompt"],
                expected_output=item["expected"],
                actual_output=llama31.generate(item["prompt"]),
            )
        )
    
    return test_cases