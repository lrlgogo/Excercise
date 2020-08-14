a = 10
for i in range(2):
    a=i
    print(a)

print(a)

a = 10

def f(b):
    a = b
    a = 1
    b = 2
    return 0

print(f(a))
print(a)

b = a
b += 1
print(b)
print(a)
