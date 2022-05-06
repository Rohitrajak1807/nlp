# ref: https://www.youtube.com/watch?v=G7FIQ9fXl6U&list=PLM8wYQRetTxBkdvBtz-gw8b9lcVkdXQKV&index=7
import numpy as np
from scipy.linalg import eig


def main():
    states = {
        0: 'burger',
        1: 'pizza',
        2: 'hotdog'
    }

    # Transition matrix A[i, j] = P(X_n = j, X_{n - 1} = i)
    A = np.array([
        [0.2, 0.6, 0.2],
        [0.3, 0.0, 0.7],
        [0.5, 0.0, 0.5]
    ])

    # Random Walk on the chain
    n = 15
    start_state = 0
    curr_state = start_state
    print(states[curr_state], '--->', end=" ")

    while n - 1:
        curr_state = np.random.choice(list(states.keys()), p=A[curr_state])
        print(states[curr_state], '--->', end=" ")
        n -= 1
    print("Random Walk done")

    # Monte Carlo on the chain
    steps = 10 ** 4
    start_state = 1
    curr_state = start_state
    pi = np.array([
        0, 0, 0
    ])
    pi[start_state] = 1

    for i in range(steps):
        curr_state = np.random.choice(list(states.keys()), p=A[curr_state])
        pi[curr_state] += 1

    pi = pi / steps
    print(pi)
    print("Monte Carlo Done")

    # Repeated matrix multiplication
    steps = 10 ** 3
    A_n = A
    for i in range(steps):
        A_n = np.matmul(A_n, A)

    print('A^n = \n', A_n)
    pi = A_n[0]
    print(pi)

    # Finding left eigen vector

    values, left_vec = eig(A, right=False, left=True)
    print(values)
    print(left_vec)
    pi = left_vec[:, 0]
    print(pi)
    pi_normalized = [(x / np.sum(pi)).real for x in pi]
    print(pi_normalized)

    print(find_prob([1, 2, 2, 0], A, pi_normalized))


def find_prob(seq, A, pi):
    print(A, pi)
    start_state = seq[0]
    prob = pi[start_state]
    prev_state, curr_state = start_state, start_state
    for i in range(1, len(seq)):
        curr_state = seq[i]
        prob = prob * A[prev_state][curr_state]
        prev_state = curr_state
    return prob


if __name__ == '__main__':
    main()
