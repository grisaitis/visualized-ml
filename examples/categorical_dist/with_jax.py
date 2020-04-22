import jax
import jax.numpy as np
import jax.scipy as scipy


key = jax.random.PRNGKey(seed=0)

n = 10
k = 3
alpha = np.ones(shape=(k,))
p = jax.random.dirichlet(key, alpha)

print("p:", p)
print("sum(p):", sum(p))

# sample_from_choice = generator.choice(k, size=n, p=p)
# print("sample_from_choice", sample_from_choice)

# sample_from_multinomial = generator.multinomial(1, pvals=p, size=n)
# print("sample_from_multinomial", sample_from_multinomial)
