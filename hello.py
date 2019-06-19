#print('hello,world')
#print('1024 * 768 = ',1024*768)
#name = input()
#print('please input your name :',name)
#print('please input your name :')
#name = input()
#print('hello,',name)
#print(r'''hello,\n
#world''')
# 
'''
"""
age = 10
if age >= 18:
    print('adult')
else:
    print('teenager')

a = 'abc'
b = a 
a = 'xuy'
print(b)

n = 123
print(n)

f = 123.456
print(f)

s1 = "'Hello, world'"
print(s1)

print("'Hello, \\'Adam\\''")
print(r'\\\t\\')
print('r\'Hello, \"Bart\"\'')
print('r\'\'\'Hello,\nLisa!\'\'\'')

print('包含中文字符')
ord('A')
ord('中')
chr(66)
chr(25991)

print('Hello， %s' % 'world')

print('\'Hello， %s' % 'world\'')
print('\'Hi, %s, you have $%d.\'' % ('Michael', 1000000))

print('%2d-%2d' % (300.99,1))

print('%.2f' % 3.1415926)

s1 = 72
s2 = 85
r = (85-72)/72
print('{0}%'.format(r))
print('growth rate: %d %%' % 7)
"""

classmates = ['Michael','Bob','Tracy']
len(classmates)
classmates[0]
classmates[-1]
'''
'''
age1 = input('input your age')
age = int(age1)
if age >= 18:
    print("your age is ", age)
    print('adult')
elif age>=12:
    print('teenager')
else:
    print('kid')
if age:
    print('True')
'''
'''
height = 1.75
weight = 80.5
bmi = weight/(height * height)
print('bmi = %.2f' % bmi)
if bmi >= 32:
    print ('严重肥胖')
elif bmi >= 28:
    print('肥胖')
elif bmi >= 25:
    print('过重')
elif bmi >= 18.5:
    print('正常')  
else:
    print('过轻')  
'''
'''
names = ['Michael', 'Bob','Tracy']
for name in names:
    print(name)      

sum = 0 
for x in range(101):
    sum = sum + x
print(sum)

sum = 0
n = 99
while n >0:
    sum = sum + n
    n = n-2
print(sum)

L = ['Bart', 'Lisa', 'Adam']
#n = len(L)
for x in L:
    print('Hello, %s' % x)

n1 = 255
n2 = hex(n1)
print(str(n2))
n3 = 1000
print(str(hex(n3)))
'''
'''
def my_abs(x):
    if not isinstance(x,(int, float)):
        raise TypeError('bad operand type')
    if x >= 0:
        return x
    else:
        return -x

#print(my_abs('a'))

def nop():
    pass

age = 0
if age >= 18:
    pass

import math 

def move(x, y, step, angle=0):
    nx = x + step * math.cos(angle)
    ny = y - step *math.sin(angle)
    return nx, ny
x, y = move(100, 100, 60, math.pi / 6)
print(x, y)
'''
# x = -b +- sqrt(b*b -4ac)/2a
'''
import math 
def quadratic(a, b, c):
    if not isinstance(a+b+c, (int, float)):
        raise TypeError('bad operand type')
    if a == 0:
        x = -c / b
        return x
    elif b * b - 4 * a * c < 0:
        print('无解')
    else:
        x1 = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
        x2 = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a)
        return x1, x2

print('quadratic(2, 3, 1) =', quadratic(2, 3, 1))
print('quadratic(1, 3, -4) =', quadratic(1, 3, -4))

if quadratic(2, 3, 1) != (-0.5, -1.0):
    print('测试失败')
elif quadratic(1, 3, -4) != (1.0, -4.0):
    print('测试失败')
else:
    print('测试成功')

print('quadratic(1, z, -4) =', quadratic(z, 3, -4))
'''
'''
def power(x):
    return x * x
power(5)
print(power(5))

def power1(x,n):
    s = 1
    while  n > 0:
        n = n - 1
        s = s * x
    return s
print(power1(5,3))

def power2(x, n = 2):
    s = 1
    while n > 0:
        n = n -1
        s = s * x
    return s
print(power2(5))
print(power2(5,4))
'''
'''
def enroll(name, gender):
    print('name:', name)
    print('gender:', gender)
print(enroll('Sarah', 'F'))

def enroll1(name, gender, age = 6, city = 'Beijing'):
    print('name:', name)
    print('gender:', gender)
    print('age:', age)
    print('city:', city)
print(enroll1('Adam', 'F'))
print(enroll1('Jack', 'M', city = 'Tianjin'))

#默认参数大坑：
def add_end(L = []):
    L.append('END')
    return L
print(add_end([1,2,3]))
print(add_end(['x', 'y', 'z']))
print(add_end())
print(add_end())

def add_end1(L = None):
    if L is None:
        L = []
    L.append('END')
    return L

print(add_end1())
print(add_end1())
'''
'''
def calc(numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum
print(calc([1, 2, 3]))# 调用需要先组装list 或 tuple
                       # print(calc(1, 3, 5, 7))调用不对，相当于函数有四个参数
#把函数参数改为可变参数
def calc1(*numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum
print(calc1(1, 2, 3))

nums = [1, 2, 3]
print(calc1(*nums))

def person(name, age, **kw):
    print('name:', name, 'age:', age, 'other:', kw)
print(person('Michael', 30))
print(person('Bob', 35, city = 'Beijing'))
print(person('Lily', 35, gender = 'M', job = 'Engineer'))

extra = {'city': 'Beijing', 'job': 'Engineer'}
print(person('Jack', 24, city = extra['city'], job = extra['job']))
print(person('Jaxck', 24, **extra))

def person1(name, age, **kw):
    if 'city' in kw:
        pass
    if 'job' in kw:
        pass
    print('name:', name, 'age:', age, 'other:', kw)
print(person1('Jack', 24, city = 'Beijing', addr = 'Chaoang', zipcode = 123456))

def product(x, y):
    return x * y
'''
'''
def product(x  , y = 1, *num):
    s = 1
    s= x * y
    for n in num:
        s = s * n
    return s

#改进
def product( *num):
    if len(num) != 0:
        s = 1
        for n in num:
            s = s * n
        return s
    else:
        raise TypeError

print('product(5) =', product(5))
print('product(5, 6) =', product(5, 6))
print('product(5, 6, 7) =', product(5, 6, 7))
print('product(5, 6, 7, 9) =', product(5, 6, 7, 9))
if product(5) != 5:
    print('测试失败!')
elif product(5, 6) != 30:
    print('测试失败!')
elif product(5, 6, 7) != 210:
    print('测试失败!')
elif product(5, 6, 7, 9) != 1890:
    print('测试失败!')
else:
    try:
        product()
        print('测试失败!')
    except TypeError:
        print('测试成功!')

def fact(n):
    if n == 1:
        return 1
    return n * fact(n-1)
#print(fact(1000))# 递归调用栈溢出

def fact(n):
    return fact_iter(n, 1)

def fact_iter(num, product):
    if num == 1:
        return product
    return fact_iter(num - 1, num * product)
'''
'''
def move(n, a, b, c):
    if n == 1:
        print(a, '-->', c)
    else:
        move(n-1, a, c, b) 
        print(a, '-->', c)       
        move(n-1, b, a, c)  

move(4, 'A', 'B', 'C')       
'''
'''
def trim(s):
    if len(s) == 0:
        raise TypeError
    for n in len(s):
        if s[n] == ' ':
            s1 = s[n+1:]
        else:
            break
    for n in len(s1):
        if s1[-n-1] == ' ':
            s2 = s1[:-n-2]
        else:
            s = s2
    return s 

trim('hello  ')
'''
'''
def trim(s):
    if len(s) == 0:
        return s
    while s[0] == ' ':
        s = s[1:]
        if len(s) == 0:  #切片后，还需要再确认字符串是否非空
            return s
    while s[-1] == ' ':
        s = s[:-1]
        if len(s) == 0:  #切片后，还需要再确认字符串是否非空
            return s       
    return s


if trim('hello  ') != 'hello':
    print('测试失败!')
elif trim('  hello') != 'hello':
    print('测试失败!')
elif trim('  hello  ') != 'hello':
    print('测试失败!')
elif trim('  hello  world  ') != 'hello  world':
    print('测试失败!')
elif trim('') != '':
    print('测试失败!')
elif trim('    ') != '':
    print('测试失败!')
else:
    print('测试成功!')

'''

