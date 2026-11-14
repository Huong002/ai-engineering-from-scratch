import ast
import math
import subprocess
import sys
import unittest
from pathlib import Path


SOURCE = Path(__file__).with_name("fx.py")
tree = ast.parse(SOURCE.read_text())
required = {"numerical_derivate", "numerical_gradient", "hessian_2d"}
functions = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in required
]
namespace = {"math": math}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)


class FxTests(unittest.TestCase):
    def test_script_runs_to_completion(self):
        result = subprocess.run(
            [sys.executable, str(SOURCE)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Learned: y = 2.00x + 1.00", result.stdout)

    def test_numerical_derivative_approximates_square_derivative(self):
        self.assertAlmostEqual(namespace["numerical_derivate"](lambda x: x**2, 2), 4, places=5)

    def test_numerical_gradient_approximates_each_partial_derivative(self):
        def function(point):
            x, y = point
            return x**2 + 3 * x * y + y**2

        gradient = namespace["numerical_gradient"](function, [1.0, 2.0])
        self.assertAlmostEqual(gradient[0], 8, places=5)
        self.assertAlmostEqual(gradient[1], 7, places=5)

    def test_hessian_computes_mixed_partial(self):
        hessian = namespace["hessian_2d"](lambda x, y: x * y, 1.0, 2.0)
        self.assertAlmostEqual(hessian[0][1], 1, places=5)
        self.assertAlmostEqual(hessian[1][0], 1, places=5)


if __name__ == "__main__":
    unittest.main()
