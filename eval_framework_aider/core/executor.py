"""
Code execution engine for running and testing generated code.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any

from core.config import EvaluationConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class CodeExecutor:
    """Executes code and runs tests"""
    
    LANGUAGE_CONFIG = {
        "python": {
            "extension": ".py",
            "run_command": ["python3"],
            "test_framework": "pytest"
        },
        "javascript": {
            "extension": ".js",
            "run_command": ["node"],
            "test_framework": "jest"
        },
        "cpp": {
            "extension": ".cpp",
            "compile_command": ["g++", "-std=c++17", "-o", "{output}", "{source}"],
            "run_command": ["./{output}"],
            "test_framework": "catch2"
        },
        "java": {
            "extension": ".java",
            "compile_command": ["javac", "{source}"],
            "run_command": ["java", "{class_name}"],
            "test_framework": "junit"
        },
        "rust": {
            "extension": ".rs",
            "compile_command": ["rustc", "{source}"],
            "run_command": ["./{output}"],
            "test_framework": "cargo test"
        }
    }
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    async def execute_code(self, code: str, language: str, test_code: str) -> Tuple[bool, str]:
        """Execute code with tests and return success status and output"""
        lang_config = self.LANGUAGE_CONFIG.get(language)
        if not lang_config:
            return False, f"Unsupported language: {language}"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Write solution file
            solution_file = tmppath / f"solution{lang_config['extension']}"
            with open(solution_file, 'w') as f:
                f.write(code)
            
            # Write test file
            test_file = tmppath / f"test{lang_config['extension']}"
            with open(test_file, 'w') as f:
                f.write(test_code)
            
            try:
                # Compile if needed
                if "compile_command" in lang_config:
                    success, output = await self._compile_code(
                        lang_config, solution_file, tmpdir
                    )
                    if not success:
                        return False, output
                
                # Run tests
                return await self._run_tests(
                    lang_config, language, test_file, tmpdir
                )
                
            except subprocess.TimeoutExpired:
                return False, "Execution timed out"
            except Exception as e:
                return False, f"Execution error: {str(e)}"
    
    async def _compile_code(self, lang_config: Dict[str, Any], 
                           solution_file: Path, tmpdir: str) -> Tuple[bool, str]:
        """Compile code if needed"""
        compile_cmd = [
            arg.format(source=str(solution_file), output="solution")
            for arg in lang_config["compile_command"]
        ]
        
        result = subprocess.run(
            compile_cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return False, f"Compilation failed: {result.stderr}"
        
        return True, "Compilation successful"
    
    async def _run_tests(self, lang_config: Dict[str, Any], language: str, 
                        test_file: Path, tmpdir: str) -> Tuple[bool, str]:
        """Run tests for the given language"""
        if language == "python":
            test_cmd = ["python3", "-m", "pytest", str(test_file), "-v"]
        elif language == "javascript":
            test_cmd = ["node", str(test_file)]
        else:
            test_cmd = lang_config["run_command"]
        
        result = subprocess.run(
            test_cmd,
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        return success, output
    
    def is_language_supported(self, language: str) -> bool:
        """Check if language is supported"""
        return language in self.LANGUAGE_CONFIG
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        return list(self.LANGUAGE_CONFIG.keys())