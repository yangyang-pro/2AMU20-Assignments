# Homework 1

## Task 1: Sigma-algebras

### Task 1a

```python
from itertools import combinations

def is_sigma_algebra(Omega, E):
    if Omega not in E:
        return False
    for item in E:
        if Omega - item not in E:
            return False
    indices = [i for i in range(len(E))]
    for i in range(2, len(E)):
        combs = list(combinations(E, i))
        for comb in combs:
            union_res = set()
            for s in comb:
                union_res = union_res.union(s)
            if union_res not in E:
                return False
    return True
```

### Task 1b

Answer: $\Sigma$ is a sigma-algebra over $\Omega$.

Proof:

Given $\Sigma_1$ and $\Sigma_2$ are two sigma-algebras and $\Sigma = \Sigma_1 \cap \Sigma_2$.

- We know that $\Omega \in \Sigma_1$ and $\Omega \in \Sigma_2$, so $\Omega \in \Sigma$.

- We know $\Sigma_1, \Sigma_2$ are closed under complement. $\forall A \in \Sigma_1 \cap \Sigma_2, \bar A \in \Sigma_1 \text{ and } \bar A \in \Sigma_2 \Rightarrow \bar A \in \Sigma_1 \cap \Sigma_2 = \Sigma$, so $\Sigma$ is closed under complement.

- We know $\Sigma_1, \Sigma_2$ are closed under union. For any subsets $\{A_1, A_2, A_3, \ldots\}$ from $\Sigma = \Sigma_1 \cap \Sigma_2$, we know $\bigcup_i A_i \in \Sigma_1, \Sigma_2 \Rightarrow \bigcup_i A_i \in \Sigma$, so $\Sigma$ is closed under union.

In summary, $\Sigma$ is a sigma-algebra over $\Omega$.

### Task 1c

Answer: $\Sigma$ is not a sigma-algebra over $\Omega$.

Proof:

Given $\Sigma_1$ and $\Sigma_2$ are two sigma-algebras and $\Sigma = \Sigma_1 \cup \Sigma_2$.

- We know that $\Omega \in \Sigma_1$ and $\Omega \in \Sigma_2$, so $\Omega \in \Sigma$.
- We know $\Sigma_1, \Sigma_2$ are closed under complement. $\forall A \in \Sigma_1 \cup \Sigma_2, \bar A \in \Sigma_1 \text{ or } \bar A \in \Sigma_2 \Rightarrow \bar A \in \Sigma_1 \cup \Sigma_2 = \Sigma$, so $\Sigma$ is closed under complement.
- We know $\Sigma_1, \Sigma_2$ are closed under union. Take $\{A_1, A_2, A_3, \ldots\}$ from $\Sigma_1$ and $\{B_1, B_2, B_3, \ldots\}$ from $\Sigma_2$ where $A_i \neq B_i$ and $\forall_i A_i, B_i \in \Sigma$, $\exists C = \bigcup_i A_i \cup B_i \notin \Sigma_1 \text{ and } C \notin \Sigma_2 \Rightarrow C \notin \Sigma$, so $\Sigma$ is not closed under union. 
  E.g. For $\Omega = \{1, 2, 3, 4, 5\}, \Sigma_1 = \{\empty, \{1,2\}, \{3,4,5\}, \{1,2,3,4,5\}\}, \Sigma_2 = \{\empty, \{1,2,5\}, \{3,4\}, \{1,2,3,4,5\}\}$, assume $A = \{1,2\}$ and $B = \{3,4\}$, $C = A \cup B = \{1,2,3,4\} \notin \Sigma = \Sigma_1 \cup \Sigma_2$.

In summary, $\Sigma$ is not a sigma-algebra over $\Omega$.

 

