"""
Prompt generation utilities for LLM code generation.
"""

import re
from typing import List

from models.data_models import Exercise


class PromptGenerator:
    """Generates prompts for different exercises"""
    
    @staticmethod
    def generate_prompt(exercise: Exercise) -> str:
        """Generate a prompt for code generation"""
        prompt = f"""You are an expert {exercise.language} programmer. 
Please solve the following programming exercise.

Exercise: {exercise.name}

Instructions:
{exercise.instructions}

{f"Starter code:{chr(10)}{exercise.starter_code}" if exercise.starter_code else ""}

Requirements:
1. Write a complete, working solution in {exercise.language}
2. The code should pass all test cases
3. Use best practices and clean code principles
4. Include only the solution code, no explanations

Please provide the complete solution code:
"""
        return prompt
    
    @staticmethod
    def generate_step_by_step_prompt(exercise: Exercise) -> str:
        """Generate a step-by-step prompt for more complex problems"""
        prompt = f"""You are an expert {exercise.language} programmer. 
Please solve the following programming exercise step by step.

Exercise: {exercise.name}

Instructions:
{exercise.instructions}

{f"Starter code:{chr(10)}{exercise.starter_code}" if exercise.starter_code else ""}

Please approach this step by step:
1. First, understand the problem and identify key requirements
2. Plan your solution approach
3. Implement the solution in {exercise.language}
4. Ensure the code passes all test cases

Requirements:
- Write clean, readable code
- Follow {exercise.language} best practices
- Include only the final solution code

Solution:
"""
        return prompt
    
    @staticmethod
    def generate_few_shot_prompt(exercise: Exercise, examples: List[str]) -> str:
        """Generate a few-shot prompt with examples"""
        prompt = f"""You are an expert {exercise.language} programmer. 
Here are some examples of similar problems and their solutions:

{chr(10).join(examples)}

Now solve this exercise:

Exercise: {exercise.name}

Instructions:
{exercise.instructions}

{f"Starter code:{chr(10)}{exercise.starter_code}" if exercise.starter_code else ""}

Please provide the complete solution code:
"""
        return prompt
    
    @staticmethod
    def extract_code(response: str, language: str) -> str:
        """Extract code from LLM response"""
        # Try to find code blocks first
        code_block_patterns = [
            rf"```{language}\n(.*?)```",
            rf"```{language.lower()}\n(.*?)```",
            r"```\n(.*?)```",
            rf"```{language}(.*?)```",
            r"```(.*?)```"
        ]
        
        for pattern in code_block_patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # If no code blocks, try to extract based on common patterns
        return PromptGenerator._extract_code_heuristic(response)
    
    @staticmethod
    def _extract_code_heuristic(response: str) -> str:
        """Extract code using heuristics when no code blocks found"""
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        # Common code indicators
        code_indicators = [
            'def ', 'class ', 'function ', 'const ', 'let ', 'var ',
            '#include', 'import ', 'from ', 'public class', 'fn ',
            'struct ', 'enum ', 'trait ', 'impl '
        ]
        
        for line in lines:
            stripped = line.strip()
            
            # Check if line looks like code
            if any(stripped.startswith(indicator) for indicator in code_indicators):
                in_code = True
            elif stripped.endswith(('{', '}', ';', ':')):
                in_code = True
            elif 'return ' in stripped:
                in_code = True
            elif '=' in stripped and not stripped.startswith('#'):
                in_code = True
            
            if in_code and stripped:
                code_lines.append(line)
            elif in_code and not stripped:
                code_lines.append(line)  # Keep empty lines within code
            elif not in_code and stripped.startswith(('>', 'Question:', 'Problem:')):
                # Stop if we hit explanatory text
                break
        
        return '\n'.join(code_lines) if code_lines else response
    
    @staticmethod
    def clean_code(code: str) -> str:
        """Clean extracted code"""
        # Remove common artifacts
        code = re.sub(r'^```.*?\n', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n```$', '', code)
        code = re.sub(r'^\s*Here.*?solution.*?:\s*$', '', code, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove excessive blank lines
        code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)
        
        return code.strip()
    
    @staticmethod
    def validate_code_structure(code: str, language: str) -> bool:
        """Basic validation of code structure"""
        if not code.strip():
            return False
        
        # Language-specific basic checks
        if language == "python":
            return 'def ' in code or 'class ' in code
        elif language == "javascript":
            return 'function' in code or '=>' in code or 'const' in code
        elif language == "java":
            return 'public' in code and 'class' in code
        elif language == "cpp":
            return '#include' in code or 'using namespace' in code
        elif language == "rust":
            return 'fn ' in code or 'struct ' in code
        
        return True  # Default to true for unknown languages