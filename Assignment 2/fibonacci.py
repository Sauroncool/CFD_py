import numpy as np
def fibonacci(n):
    if n == 0:
        return np.array([])
    elif n == 1:
        return np.array([1])
    else:
        fib = np.zeros(n, dtype=int)
        fib[0] = 1
        fib[1] = 1
        for i in range(2, n):
            fib[i] = fib[i-1] + fib[i-2]
        return fib
    
def main():
    print(fibonacci(5))
    print(fibonacci(10))

if __name__ == '__main__':
    main()