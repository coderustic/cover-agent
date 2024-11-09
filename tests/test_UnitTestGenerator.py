from cover_agent.CoverageProcessor import CoverageProcessor
from cover_agent.ReportGenerator import ReportGenerator
from cover_agent.Runner import Runner
from cover_agent.UnitTestGenerator import UnitTestGenerator
from unittest.mock import patch, mock_open
import datetime
import os
import pytest
import tempfile

from unittest.mock import MagicMock
class TestUnitTestGenerator:
    def test_get_included_files_mixed_paths(self):
        with patch("builtins.open", mock_open(read_data="file content")) as mock_file:
            mock_file.side_effect = [
                IOError("File not found"),
                mock_open(read_data="file content").return_value,
            ]
            included_files = ["invalid_file1.txt", "valid_file2.txt"]
            result = UnitTestGenerator.get_included_files(included_files)
            assert (
                result
                == "file_path: `valid_file2.txt`\ncontent:\n```\nfile content\n```"
            )

    def test_get_included_files_valid_paths(self):
        with patch("builtins.open", mock_open(read_data="file content")):
            included_files = ["file1.txt", "file2.txt"]
            result = UnitTestGenerator.get_included_files(included_files)
            assert (
                result
                == "file_path: `file1.txt`\ncontent:\n```\nfile content\n```\nfile_path: `file2.txt`\ncontent:\n```\nfile content\n```"
            )
    def test_get_code_language_no_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_source_file:
            generator = UnitTestGenerator(
                source_file_path=temp_source_file.name,
                test_file_path="test_test.py",
                code_coverage_report_path="coverage.xml",
                test_command="pytest",
                llm_model="gpt-3"
            )
            language = generator.get_code_language("filename")
            assert language == "unknown"

    def test_build_prompt_with_failed_tests(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_source_file:
            generator = UnitTestGenerator(
                source_file_path=temp_source_file.name,
                test_file_path="test_test.py",
                code_coverage_report_path="coverage.xml",
                test_command="pytest",
                llm_model="gpt-3"
            )
            failed_test_runs = [
                {
                    "code": {"test_code": "def test_example(): assert False"},
                    "error_message": "AssertionError"
                }
            ]
            prompt = generator.build_prompt(failed_test_runs)
            assert "Failed Test:" in prompt['user']


    def test_generate_tests_invalid_yaml(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_source_file:
            generator = UnitTestGenerator(
                source_file_path=temp_source_file.name,
                test_file_path="test_test.py",
                code_coverage_report_path="coverage.xml",
                test_command="pytest",
                llm_model="gpt-3"
            )
            generator.build_prompt = lambda: "Test prompt"
            with patch.object(generator.ai_caller, 'call_model', return_value=("This is not YAML", 10, 10)):
                result = generator.generate_tests()
                
                # The eventual call to try_fix_yaml() will end up spitting out the same string but deeming is "YAML."
                # While this is not a valid YAML, the function will return the original string (for better or for worse).
                assert result =="This is not YAML"

    def test_initial_test_suite_analysis_with_prompt_builder(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_source_file:
            generator = UnitTestGenerator(
                source_file_path=temp_source_file.name,
                test_file_path="test_test.py",
                code_coverage_report_path="coverage.xml",
                test_command="pytest",
                llm_model="gpt-3"
            )
            
            # Mock the prompt builder
            mock_prompt_builder = MagicMock()
            generator.prompt_builder = mock_prompt_builder
            mock_prompt_builder.build_prompt_custom.side_effect = [
                "test_headers_indentation: 4",
                "relevant_line_number_to_insert_tests_after: 100\nrelevant_line_number_to_insert_imports_after: 10\ntesting_framework: pytest"
            ]
            
            # Mock the AI caller responses
            with patch.object(generator.ai_caller, 'call_model') as mock_call:
                mock_call.side_effect = [
                    ("test_headers_indentation: 4", 10, 10),
                    ("relevant_line_number_to_insert_tests_after: 100\nrelevant_line_number_to_insert_imports_after: 10\ntesting_framework: pytest", 10, 10)
                ]
                
                generator.initial_test_suite_analysis()
                
                assert generator.test_headers_indentation == 4
                assert generator.relevant_line_number_to_insert_tests_after == 100
                assert generator.relevant_line_number_to_insert_imports_after == 10
                assert generator.testing_framework == "pytest"

                