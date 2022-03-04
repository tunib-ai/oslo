from pprint import pprint

memo = {}
init_L = 8
Q_n = [0.3, 0.06, 0.639, 0.001]
#   = [wte, wpe,  h,     ln_f]


def c(k, i):
    if (k, i) in memo:
        return memo[(k, i)]

    cost = []
    if 2 <= k <= init_L:
        for j in range(i + 1):
            cost_j = max(c(k - 1, j), sum(Q_n[:j]))

            if cost_j > 0:
                cost.append(cost_j)

        if len(cost) > 0:
            min_cost = min(cost)
        else:
            min_cost = -1

        memo[(k, i)] = min_cost

        return memo[(k, i)]
    else:
        return -1


for i in range(0, len(Q_n)):
    output = c(init_L, i)

pprint(memo)
