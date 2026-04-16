# nntel 
tensor library with 0 dependencies (even gemm and gemv are implemented from scratch with perfomance near to MPS)

### Backpropogation derivation
```raw
error = dC/dz

z(x) = w * x + b
a(x) = sigmoid(z(x))
C(x) = sum((y - aL(x))**2) / 2N

where aL is activation of Last layer (neural net output)

how to compute gradient of last layer? we need dC/dw and dC/db
we can easily get dz/dw and dz/db
we can easily get da/dz
we can easily get dC/daL (meaning for last layer only)

so what is C(x) more verbosely?
- y is vector (target values)
- aL(x) is vector (net output)

N = len(y)
C(x) = sum([(y[i] - aL(x)[i])**2 for i in range(N)]) / 2N
C(x) = ((y[0] - aL(x)[0])**2 + (y[1] - aL(x)[1])**2 + ... + (y[N - 1] + aL(x)[N - 1])**2) / 2N

so dC/da[i] = 2 * (y[i] - aL(x)[i]) * -1 / 2 = aL(x)[i] - y[i] 
    where x is layer input (or last layer input, activations of previous layer)
da/dz = sigmoid_derivative(z(x))

=> dC/dz[i] = dC/da[i] * da[i]/dz[i] = (aL(x)[i] - y[i]) * sigmoid_derivative(z(x)[i])
or in vector form dC/dz = (aL(x) - y) * sigmoid_derivative(z(x)) where * is element-wise mul

===============================================================================================

but this is for output layer only!
what about inner layers?

dC/dz[i] = dC/dz[i + 1] * dz[i+1]/da[i] * da[i]/dz[i]
z[i](x) = w * a[i-1](x) + b

assume dC/dz[i + 1] is known

dz[i+1]/da[i] is easily derived since
z[i+1](x) = w * a[i] + b
dz[i+1]/da[i] = w[i]

so dC/dz[i] = dC/dz[i+1] * w[i] * sigmoid_derivative(z[i](x))
```

but what if we use different cost function, such as cross-entropy?
```raw
for single neuron output neural net for single output:
H(P, Q) = sum(p * log2(1/q) for p,q in zip(P, Q))
for binary case where P = (p, 1-p) (head or tails), the H(P, Q) = -(p * log2(q) + (1 - p) * log2(1 - q))

output neuron did output aL(x), but target is y, then cross entropy as cost function is:
C(x) = -(y * ln(aL(x)) + (1 - y) * ln(1 - aL(x)))
> ln because it will simplify sigmoid derivative

for all data we compute average cross entropy over data:
C(x) = -sum(y * ln(aL(x)) + (1 - y) * ln(1 - aL(x))) / N

this cost function will simplify sigmoid derivative from dC/dz[i] of the output layer and it will become just:

dC/dz[i] = dC/da[i] * da[i]/dz[i] = (aL(x)[i] - y[i])
```















