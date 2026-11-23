import unittest
import sys
import os

# Add the src directory to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.orchestrator.utils import validate_plan_dataflow

class TestPlanValidation(unittest.TestCase):

    def test_valid_sequential_plan(self):
        """Tests a plan with a correct sequence of dependencies."""
        plan = [
            {
                "task": "list_tables",
                "inputs": {},
                "outputs": ["table_list"],
            },
            {
                "task": "get_schemas",
                "inputs": {"tables": "table_list"},
                "outputs": ["table_schemas"],
            },
        ]
        is_valid, message = validate_plan_dataflow(plan)
        self.assertTrue(is_valid)
        self.assertEqual(message, "Plan dataflow is valid.")

    def test_invalid_plan_missing_input(self):
        """Tests a plan where an input artifact is never produced."""
        plan = [
            {
                "task": "get_schemas",
                "inputs": {"tables": "table_list"}, # "table_list" is never created
                "outputs": ["table_schemas"],
            },
        ]
        is_valid, message = validate_plan_dataflow(plan)
        self.assertFalse(is_valid)
        self.assertIn("requires artifact 'table_list'", message)
        self.assertIn("not produced by any preceding step", message)

    def test_invalid_plan_incorrect_order(self):
        """Tests a plan where steps are in the wrong logical order."""
        plan = [
            {
                "task": "get_schemas",
                "inputs": {"tables": "table_list"},
                "outputs": ["table_schemas"],
            },
            {
                "task": "list_tables",
                "inputs": {},
                "outputs": ["table_list"],
            },
        ]
        is_valid, message = validate_plan_dataflow(plan)
        self.assertFalse(is_valid)
        self.assertIn("Step 1 ('get_schemas')", message)
        self.assertIn("requires artifact 'table_list'", message)

    def test_valid_plan_no_dependencies(self):
        """Tests a valid plan where no steps have input dependencies."""
        plan = [
            {
                "task": "get_row_count",
                "inputs": {},
                "outputs": ["row_count"],
            },
            {
                "task": "get_column_names",
                "inputs": {},
                "outputs": ["column_names"],
            },
        ]
        is_valid, message = validate_plan_dataflow(plan)
        self.assertTrue(is_valid)
        self.assertEqual(message, "Plan dataflow is valid.")

    def test_empty_plan(self):
        """Tests that an empty plan is considered valid."""
        plan = []
        is_valid, message = validate_plan_dataflow(plan)
        self.assertTrue(is_valid)
        self.assertEqual(message, "Plan dataflow is valid.")

    def test_valid_plan_multiple_inputs(self):
        """Tests a valid plan with a step that requires multiple inputs."""
        plan = [
            {"task": "get_tables", "inputs": {}, "outputs": ["table_list"]},
            {"task": "get_columns", "inputs": {}, "outputs": ["column_list"]},
            {
                "task": "join_info",
                "inputs": {"tables": "table_list", "columns": "column_list"},
                "outputs": ["joined_info"],
            },
        ]
        is_valid, message = validate_plan_dataflow(plan)
        self.assertTrue(is_valid)
        self.assertEqual(message, "Plan dataflow is valid.")


if __name__ == '__main__':
    unittest.main()
