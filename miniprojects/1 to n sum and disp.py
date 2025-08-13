n = 10

# 1. Print numbers from 1 to n in one line (space-separated)
print(" ".join(str(i) for i in range(1, n+1)))

# 2. Print numbers from 1 to n in one line (using unpacking)
print(*range(1, n+1))

# 3. Print numbers from 1 to n in one line (using a for loop with print and end)
for i in range(1, n+1): print(i, end=' ')
print()

# 4. Sum of first n natural numbers using formula
print(n * (n + 1) // 2)

# 5. Sum of first n natural numbers using sum() and range()
print(sum(range(1, n+1)))
