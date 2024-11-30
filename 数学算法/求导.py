def derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h
