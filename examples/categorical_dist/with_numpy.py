import numpy as np

generator = np.random.default_rng(seed=2)

n = 10
k = 3
alpha = np.ones(shape=(k,))
p = generator.dirichlet(alpha)

print("p:", p)
print("sum(p):", sum(p))

sample_from_choice = generator.choice(k, size=n, p=p)
print("sample_from_choice", sample_from_choice)

sample_from_multinomial = generator.multinomial(1, pvals=p, size=n)
print("sample_from_multinomial", sample_from_multinomial)
