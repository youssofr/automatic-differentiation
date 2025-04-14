import numpy as np

import sys
sys.path.append("../src")
from auto_diff import diff

# define test functions
F = [lambda x : x**2,
     lambda x : np.sin(x),
     lambda x : np.exp(x),
     lambda x : x**2 - x * np.exp2(x)]
# define analytic derivatives
D = [lambda x : 2*x,
     lambda x : np.cos(x),
     lambda x : np.exp(x),
     lambda x : 2*x - np.exp2(x) - x * np.exp2(x) * np.log(2)]

n_tests = len(F)
assert len(D) == n_tests,\
       "Number of test functions does not match number of test derivatives."
# test range
X = np.linspace(-10, 10, 1000)

# element by element tests 
for x in X:
    for i in range(n_tests):
        assert (diff(F[i], x) - D[i](x)) < 1e-12,\
               f"Analytical derivative does not match auto diff derivative.\nanalyic={diff(F[i], x):.10f}, auto_diff={D[i](x):.10f} @ x={x:.10f}"

print("Element by element tests passed successfully!")

# array tests
for i in range(n_tests):
     assert ((diff(F[i], X) - D[i](X)) < 1e-12).all(),\
          f"Analytical derivative does not match auto diff derivative.\nanalyic={diff(F[i], x):.10f}, auto_diff={D[i](x):.10f} @ x={x:.10f}"
     
print("Array tests passed successfully!")