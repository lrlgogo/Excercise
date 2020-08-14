import random


def fun_profit(demand, store):
    flag = demand - store
    prof_0 = 500 * min(demand, store)
    prof_1 = 100 * flag if flag < 0 else 300 * flag
    return prof_0 + prof_1


profit = []
for in_store in range(10, 31):
    profit_mont = []
    for i in range(int(1e5)):
        demand = random.randint(10, 30)
        profit_mont.append(fun_profit(demand, in_store))
    profit.append(sum(profit_mont) / len(profit_mont))

print(list(range(10, 31))[profit.index(max(profit))])
