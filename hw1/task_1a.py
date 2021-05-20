from itertools import combinations

def is_sigma_algebra(Omega, E):
    # check if Omega is in E
    if Omega not in E:
        return False
    # check if E is closed under complement
    for item in E:
        if Omega - item not in E:
            return False
    # check if E is closed under union
    for i in range(2, len(E) + 1):
        combs = list(combinations(E, i))
        for comb in combs:
            union_res = set()
            for s in comb:
                union_res = union_res.union(s)
            if union_res not in E:
                return False
    return True

if __name__ == '__main__':
    # Example
    Omega = set([1, 2, 3, 4])
    E = [set(), Omega, set([1, 2]), set([3, 4])]
    print(is_sigma_algebra(Omega, E))
    # Example
    Omega = set([1, 2, 3, 4])
    E = [set(), Omega, set([1, 2]), set([3]), set([4])]
    print(is_sigma_algebra(Omega, E))
    # Example
    Omega = set([1, 2, 3])
    E = [set(), Omega, set([1]), set([2]), set([3]),
         set([1, 2]), set([1, 3]), set([2, 3])]
    print(is_sigma_algebra(Omega, E))
