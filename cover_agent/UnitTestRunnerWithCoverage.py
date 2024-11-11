from cover_agent.Runner import Runner
from cover_agent.CoverageProcessor import CoverageProcessor

class UnitTestRunnerWithCoverage:
    def __init__(self, test_command, test_command_dir, code_coverage_report_path, source_file_path, coverage_type, use_report_coverage_feature_flag, logger):
        self.test_command = test_command
        self.test_command_dir = test_command_dir
        self.code_coverage_report_path = code_coverage_report_path
        self.source_file_path = source_file_path
        self.coverage_type = coverage_type
        self.use_report_coverage_feature_flag = use_report_coverage_feature_flag
        self.logger = logger
        
        self.coverage_processor = CoverageProcessor(
            file_path=self.code_coverage_report_path,
            src_file_path=self.source_file_path,
            coverage_type=self.coverage_type,
            use_report_coverage_feature_flag=self.use_report_coverage_feature_flag
        )

    def run_tests(self):
        self.logger.info(
            f'Running build/test command to generate coverage report: "{self.test_command}"'
        )
        stdout, stderr, exit_code, time_of_test_command = Runner.run_command(
            command=self.test_command, cwd=self.test_command_dir
        )
        assert (
            exit_code == 0
        ), f'Fatal: Error running test command. Are you sure the command is correct? "{self.test_command}"\nExit code {exit_code}. \nStdout: \n{stdout} \nStderr: \n{stderr}'
        return stdout, stderr, exit_code, time_of_test_command

    def parse_coverage_report(self, time_of_test_command):
        last_coverage_percentages = {}
        if self.use_report_coverage_feature_flag:
            self.logger.info(
                "Using the report coverage feature flag to process the coverage report"
            )
            file_coverage_dict = self.coverage_processor.process_coverage_report(
                time_of_test_command=time_of_test_command
            )
            total_lines_covered = 0
            total_lines_missed = 0
            total_lines = 0
            for key in file_coverage_dict:
                lines_covered, lines_missed, percentage_covered = (
                    file_coverage_dict[key]
                )
                total_lines_covered += len(lines_covered)
                total_lines_missed += len(lines_missed)
                total_lines += len(lines_covered) + len(lines_missed)
                if key == self.source_file_path:
                    self.last_source_file_coverage = percentage_covered
                if key not in last_coverage_percentages:
                    last_coverage_percentages[key] = 0
                last_coverage_percentages[key] = percentage_covered
            try:
                percentage_covered = total_lines_covered / total_lines
            except ZeroDivisionError:
                self.logger.error(f"ZeroDivisionError: Attempting to perform total_lines_covered / total_lines: {total_lines_covered} / {total_lines}.")
                percentage_covered = 0

            self.logger.info(
                f"Total lines covered: {total_lines_covered}, Total lines missed: {total_lines_missed}, Total lines: {total_lines}"
            )
            self.logger.info(
                f"coverage: Percentage {round(percentage_covered * 100, 2)}%"
            )
        else:
            lines_covered, lines_missed, percentage_covered = (
                self.coverage_processor.process_coverage_report(
                    time_of_test_command=time_of_test_command
                )
            )
            self.code_coverage_report = f"Lines covered: {lines_covered}\nLines missed: {lines_missed}\nPercentage covered: {round(percentage_covered * 100, 2)}%"
        return percentage_covered, last_coverage_percentages