# Question 1
def neville_interpolation(x, x_data, y_data):

  n = len(x_data)
  Q = [0.0] * n

  for i in range(n):
    Q[i] = y_data[i]

  for i in range(1, n):
    for j in range(n - i):
      Q[j] = ((x - x_data[j + i]) * Q[j] +
              (x_data[j] - x) * Q[j + 1]) / (x_data[j] - x_data[j + i])

  return Q[0]


# Given data points
x_data = [3.6, 3.8, 3.9]
y_data = [1.675, 1.436, 1.318]

# Point to interpolate
x_interpolate = 3.7

# Interpolate using Neville's method
interpolated_value = neville_interpolation(x_interpolate, x_data, y_data)
print(interpolated_value)

print("\n")


#Question 2
def forward_difference_table(x_data, y_data):
  n = len(x_data)
  table = [[0] * n for _ in range(n)]

  # Initialize the first column with y_data
  for i in range(n):
    table[i][0] = y_data[i]

  # Compute forward differences
  for i in range(1, n):
    for j in range(1, i + 1):
      table[i][j] = (table[i][j - 1] - table[i - 1][j - 1]) / (x_data[i] -
                                                               x_data[i - j])

  return table


def print_forward_difference_table(table):
  n = len(table)
  print("Forward Difference Table:")
  for i in range(n):
    for j in range(i + 1):
      print(f"{table[i][j]:.6f}", end="\t")
    print("")


# Given data points
x_data = [7.2, 7.4, 7.5, 7.6]
y_data = [23.5492, 25.3913, 26.8224, 27.4589]

# Compute forward difference table
table = forward_difference_table(x_data, y_data)

# Print the forward difference table
print_forward_difference_table(table)

print("\n")

#Question 3
p1 = 23.54920 + (9.210500 * (7.3 - 7.2))
p2 = p1 + (17.001667 * (7.3 - 7.2) * (7.3 - 7.4))
p3 = p2 + (-141.829167 * (7.3 - 7.2) * (7.3 - 7.4) * (7.3 - 7.5))
print(p3)

print("\n")


#Question 4
def hermite_polynomial_approximation(x_data, y_data, y_prime_data):
  n = len(x_data)
  Q = [[0.0] * (2 * n + 2) for _ in range(2 * n + 2)]

  # Step 2: Initialize Q values
  for i in range(n):
    Q[2 * i][0] = y_data[i]
    Q[2 * i + 1][0] = y_data[i]
    Q[2 * i + 1][1] = y_prime_data[i]

  # Step 3: Calculate Q values
  for i in range(2, 2 * n + 2):
    for j in range(2, i + 1):
      if i % 2 == 0 and j == 1:
        Q[i][j] = Q[i][0] - Q[i - 1][0]

      elif (i % 2 == 0 and j > 1) or (i % 2 == 1 and j > 0):
        Q[i][j] = Q[i][j - 1] - Q[i - 1][j - 1]

  return Q


# Given data points and derivative values
x_data = [3.6, 3.8, 3.9]
y_data = [1.675, 1.436, 1.318]
y_prime_data = [-1.195, -1.188, -1.182]

# Calculate Hermite polynomial approximation matrix
hermite_matrix = hermite_polynomial_approximation(x_data, y_data, y_prime_data)

# Output Hermite polynomial approximation matrix
print("Hermite polynomial approximation matrix:")
for row in hermite_matrix:
  print(row)

print("\n")

#Question 5
import numpy as np

# Given data points
x_values = np.array([2, 5, 8, 10])
y_values = np.array([3, 5, 7, 9])

# Calculate intervals
h_values = np.diff(x_values)

# Construct matrix A
A = np.zeros((len(x_values), len(x_values)))
A[0, 0] = 1
A[-1, -1] = 1
for i in range(1, len(x_values) - 1):
  A[i, i - 1] = h_values[i - 1]
  A[i, i] = 2 * (h_values[i - 1] + h_values[i])
  A[i, i + 1] = h_values[i]

# Construct vector b
b = np.zeros(len(x_values))
for i in range(1, len(x_values) - 1):
  b[i] = (6 / h_values[i]) * (y_values[i + 1] - y_values[i]) - \
         (6 / h_values[i - 1]) * (y_values[i] - y_values[i - 1])

# Solve for vector x
x = np.linalg.solve(A, b)

print("Matrix A:")
print(A)
print("\nVector b:")
print(b)
print("\nVector x:")
print(x)
