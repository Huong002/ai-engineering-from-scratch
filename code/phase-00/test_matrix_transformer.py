import math
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("matrix_transformer.py")
source_text = SOURCE.read_text()
definitions = source_text.split("point =", maxsplit=1)[0]
namespace = {"math": math}
exec(compile(definitions, str(SOURCE), "exec"), namespace)


class MatrixTransformerTests(unittest.TestCase):
    def test_reflection_y_has_consistent_function_name(self):
        self.assertEqual(namespace["reflection_y"](), [[-1, 0], [0, 1]])

    def test_matrix_multiplication_computes_each_output_cell(self):
        self.assertEqual(
            namespace["mat_mul"]([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
            [[19, 22], [43, 50]],
        )


if __name__ == "__main__":
    unittest.main()
