"""
Exercise data fetching from Exercism and other sources.
"""

import json
import requests
from pathlib import Path
from typing import List, Dict, Any

from models.data_models import Exercise
from utils.logger import get_logger

logger = get_logger(__name__)


class ExercismFetcher:
    """Fetches exercises from Exercism"""
    
    BASE_URL = "https://exercism.org/api/v2"
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_exercise_list(self, language: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of exercises for a language"""
        cache_file = self.cache_dir / f"{language}_exercises.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)[:limit]
        
        try:
            # Exercism's API authentication and rate limiting
            url = f"{self.BASE_URL}/tracks/{language}/exercises"
            response = requests.get(url)
            response.raise_for_status()
            exercises = response.json()["exercises"][:limit]
            
            with open(cache_file, 'w') as f:
                json.dump(exercises, f)
            
            return exercises
        except Exception as e:
            logger.error(f"Failed to fetch exercises: {e}")
            return []
    
    def get_exercise_details(self, language: str, exercise_slug: str) -> Dict[str, Any]:
        """Get detailed information about a specific exercise"""
        cache_file = self.cache_dir / f"{language}_{exercise_slug}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        try:
            url = f"{self.BASE_URL}/tracks/{language}/exercises/{exercise_slug}"
            response = requests.get(url)
            response.raise_for_status()
            exercise_data = response.json()
            
            with open(cache_file, 'w') as f:
                json.dump(exercise_data, f)
            
            return exercise_data
        except Exception as e:
            logger.error(f"Failed to fetch exercise details: {e}")
            return {}


class SampleExerciseProvider:
    """Provides sample exercises for testing"""
    
    @staticmethod
    def get_sample_exercises() -> List[Exercise]:
        """Get sample exercises for testing"""
        return [
            Exercise(
                slug="two-fer",
                name="Two Fer",
                language="python",
                instructions="Create a function that gives out a message: 'One for X, one for me.' where X is the given name. If no name is given, use 'you'.",
                test_code="""
def test_two_fer():
    from solution import two_fer
    assert two_fer() == "One for you, one for me."
    assert two_fer("Alice") == "One for Alice, one for me."
    assert two_fer("Bob") == "One for Bob, one for me."
""",
                starter_code="def two_fer(name=None):\n    pass",
                difficulty="easy"
            ),
            Exercise(
                slug="leap",
                name="Leap Year",
                language="python",
                instructions="Given a year, report if it is a leap year. A leap year occurs on every year that is evenly divisible by 4, except every year that is evenly divisible by 100, unless the year is also evenly divisible by 400.",
                test_code="""
def test_leap_year():
    from solution import is_leap_year
    assert is_leap_year(2000) == True
    assert is_leap_year(2100) == False
    assert is_leap_year(2020) == True
    assert is_leap_year(2021) == False
""",
                difficulty="easy"
            ),
            Exercise(
                slug="fizzbuzz",
                name="FizzBuzz",
                language="javascript",
                instructions="Write a function that returns 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for multiples of both, and the number itself otherwise.",
                test_code="""
const fizzbuzz = require('./solution').fizzbuzz;

console.assert(fizzbuzz(3) === 'Fizz');
console.assert(fizzbuzz(5) === 'Buzz');
console.assert(fizzbuzz(15) === 'FizzBuzz');
console.assert(fizzbuzz(7) === '7');
console.log('All tests passed!');
""",
                difficulty="easy"
            ),
            Exercise(
                slug="reverse-string",
                name="Reverse String",
                language="python",
                instructions="Reverse a string. For example, if the input is 'hello', the output should be 'olleh'.",
                test_code="""
def test_reverse_string():
    from solution import reverse_string
    assert reverse_string("hello") == "olleh"
    assert reverse_string("world") == "dlrow"
    assert reverse_string("") == ""
    assert reverse_string("a") == "a"
""",
                difficulty="easy"
            ),
            Exercise(
                slug="sum-of-multiples",
                name="Sum of Multiples",
                language="python",
                instructions="Given a number, find the sum of all the unique multiples of particular numbers up to but not including that number.",
                test_code="""
def test_sum_of_multiples():
    from solution import sum_of_multiples
    assert sum_of_multiples(20, [3, 5]) == 78
    assert sum_of_multiples(10, [3, 5]) == 23
    assert sum_of_multiples(4, [3, 5]) == 3
""",
                difficulty="medium"
            ),
            Exercise(
                slug="hamming",
                name="Hamming Distance",
                language="python",
                instructions="Calculate the Hamming distance between two DNA strands. The Hamming distance is the number of point mutations between two strings of equal length.",
                test_code="""
def test_hamming():
    from solution import hamming_distance
    assert hamming_distance("GAGCCTACTAACGGGAT", "CATCGTAATGACGGCCT") == 7
    assert hamming_distance("GGACGGATTCTG", "AGGACGGATTCT") == 9
    assert hamming_distance("", "") == 0
""",
                difficulty="easy"
            )
        ]
    
    @staticmethod
    def get_exercises_by_language(language: str) -> List[Exercise]:
        """Get exercises filtered by language"""
        all_exercises = SampleExerciseProvider.get_sample_exercises()
        return [ex for ex in all_exercises if ex.language == language]
    
    @staticmethod
    def get_exercises_by_difficulty(difficulty: str) -> List[Exercise]:
        """Get exercises filtered by difficulty"""
        all_exercises = SampleExerciseProvider.get_sample_exercises()
        return [ex for ex in all_exercises if ex.difficulty == difficulty]