def min_count(n):
    if not isinstance(n, int):#explainationS
        raise TypeError('bad operand type')
    m = 1024 - n 
    num = m//64 + m % 64 //16 + m % 16 // 4 + m %4 
    return num

print(min_count(200))