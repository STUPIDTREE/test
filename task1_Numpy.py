#coding=utf-8

# from numpy import *
# a = arange(15).reshape(3, 5)
# print(a)

#1
import numpy as np

#2
#print(np.__version__)
#np.show_config()

#3
#print(np.zeros(10))

#4-------------------------------------------Z.size  Z.itemsize 输入时无自动补全提示
# Z = np.zeros((10,10))
# print("%d bytes" % (Z.size * Z.itemsize))

#5-------------------------------------------
#numpy.info(numpy.add)

#6
# p = np.zeros(10)
# p[4] = 1
# print(p)

#7
# p = np.arange(10,50)
# print(p)

#8-----------------------------反转向量
# p = np.arange(50)
# p = p[::-1]
# print(p)
#------------------------------
'''
#9
p = np.arange(9).reshape(3,3)
print(p)

#10
p = np.nonzero([1,2,0,0,4,0])
print(p)

#11
p = np.eye(3)
print(p)

#12------------------------------random使用
p = np.random.random((3,3,3))
print(p)
#--------------------------------

#13----------------------------p.min(), p.max()无自动补全
p = np.random.random((10,10))
pmin, pmax = p.min(), p.max()
print(pmin, pmax)

#14
p = np.random.random(30)
m = p.mean()
print(m)

#15------------------------ones用法
# [[ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
#  [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
#  [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
#  [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
#  [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
#  [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
#  [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
#  [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
#  [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  1.]
#  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]
p = np.ones((10,10))
p[1:-1,1:-1] = 0
print(p)


#16---------------------------pad
# [[ 0.  0.  0.  0.  0.  0.  0.]
#  [ 0.  1.  1.  1.  1.  1.  0.]
#  [ 0.  1.  1.  1.  1.  1.  0.]
#  [ 0.  1.  1.  1.  1.  1.  0.]
#  [ 0.  1.  1.  1.  1.  1.  0.]
#  [ 0.  1.  1.  1.  1.  1.  0.]
#  [ 0.  0.  0.  0.  0.  0.  0.]]
p = np.ones((5,5))
p = np.pad(p, pad_width=1, mode='constant', constant_values=0)
print(p)


#17----------------------------
# nan
# False
# False
# nan
# False
print(0 * np.nan)
print(np.nan == np.nan)
print(np.inf > np.nan)
print(np.nan - np.nan)
print(0.3 == 3 * 0.1)
#-------------------------------


#18----------------------------------
# [[0 0 0 0 0]
#  [1 0 0 0 0]
#  [0 2 0 0 0]
#  [0 0 3 0 0]
#  [0 0 0 4 0]]
p = np.diag(1 + np.arange(4),k = -1)
print(p)

#19-------------------------------
# [[0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]]

p = np.zeros((8,8), dtype = int)
p[1::2, ::2] = 1
p[::2, 1::2] = 1
print(p)

#20---------------------------------------
#(1, 5, 4)第100个元素的index，唯一？逆序第100个呢？
#print(np.unravel_index(100,(6,7,8)))

#21----------------------------------------
# [[0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]
#  [0 1 0 1 0 1 0 1]
#  [1 0 1 0 1 0 1 0]]
p = np.tile(np.array([[0,1],[1,0]]), (4,4))
print(p)

#22-----------------------------------------  5*5随机矩阵归一化
# [[ 0.7927922   0.19774499  0.73334692  0.06276436  0.        ]
#  [ 0.30162679  0.45261418  0.37645704  0.49987887  0.5873821 ]
#  [ 0.65872373  0.63420666  0.1746962   0.80709986  0.7612635 ]
#  [ 0.26963951  0.06875662  0.60789154  0.09861894  0.64148033]
#  [ 1.          0.45828734  0.97540547  0.41554111  0.75700546]]
p = np.random.random((5,5))
pmax, pmin = p.max(), p.min()
p = (p - pmin)/(pmax - pmin)
print(p)


#23---------------------------------------------
#[('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')]
color = np.dtype([("r", np.ubyte, 1),
("g", np.ubyte, 1),
("b", np.ubyte, 1),
("a", np.ubyte, 1)
])
print(color)

#24---------------------------------------------?
p = np.dot(np.ones(5,3), np.ones(3,2))
print(p)

#25
p = np.arange(11)
p[(3 < p) & (p <= 8)] *= -1
print(p)

#26------------------?
# 9
# 10
print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))

#27-------------
# Z**Z   
# 2 << Z >> 2
# Z <- Z
# 1j*Z
# Z/1/1
# Z<Z>Z

#28-------------- ？
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)

#29------------
#[ -7.  -1.   4.  -8.   8.   6. -10.   2.   6.   7.]
Z = np.random.uniform(-10,+10,10)
print (np.copysign(np.ceil(np.abs(Z)), Z))

#30----------
# [1 3 4 7 9]
Z1 = np.random.randint(0,10,10)
Z2 = np.random.randint(0,10,10)
print(np.intersect1d(Z1,Z2))


#31---------------
# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Back to sanity
_ = np.seterr(**defaults)

#An equivalent way, with a context manager:

with np.errstate(divide='ignore'):
    Z = np.ones(1) / 0
    

#32-------------False
#np.sqrt(-1) == np.emath.sqrt(-1)

#33----------------  ？
# Yesterday is 2019-06-22
# Today is 2019-06-23
# Tomorrow is 2019-06-24
yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
today     = np.datetime64('today', 'D')
tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')
print ("Yesterday is " + str(yesterday))
print ("Today is " + str(today))
print ("Tomorrow is "+ str(tomorrow))

#34
Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
print(Z)

#35
A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)

#36------------------
# [ 7.  6.  3.  4.  7.  1.  1.  1.  2.  1.]
# [ 7.  6.  3.  4.  7.  1.  1.  1.  2.  1.]
# [ 7.  6.  3.  4.  7.  1.  1.  1.  2.  1.]
# [7 6 3 4 7 1 1 1 2 1]
# [ 7.  6.  3.  4.  7.  1.  1.  1.  2.  1.]

Z = np.random.uniform(0,10,10)

print (Z - Z%1)
print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))
print (np.trunc(Z))

#37-------------
# [[ 0.  1.  2.  3.  4.]
#  [ 0.  1.  2.  3.  4.]
#  [ 0.  1.  2.  3.  4.]
#  [ 0.  1.  2.  3.  4.]
#  [ 0.  1.  2.  3.  4.]]
Z = np.zeros((5,5))
Z += np.arange(5)
print (Z)


#38
# [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9.]
def generate():
    for x in range(10):
        yield x
Z = np.fromiter(generate(),dtype=float,count=-1)
print (Z)

#39
#[ 0.09090909  0.18181818  0.27272727  0.36363636  0.45454545  0.54545455
 # 0.63636364  0.72727273  0.81818182  0.90909091]
Z = np.linspace(0,1,11,endpoint=False)[1:]
print (Z)

#40
# [ 0.15345639  0.1940283   0.24612131  0.47467806  0.5925717   0.59725163
#   0.68321409  0.69789732  0.76898197  0.8301158 ]
Z = np.random.random(10)
Z.sort()
print (Z)

#41
Z = np.arange(10)
np.add.reduce(Z)

#42
# False
# False
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
# Assuming identical shape of the arrays and a tolerance for the comparison of values
equal = np.allclose(A,B)
print(equal)
# 方法2
# Checking both the shape and the element values, no tolerance (values have to be exactly equal)
equal = np.array_equal(A,B)
print(equal)

#43------------------??????
# Z = np.zeros(10)
# Z.flags.writeable = False
# Z[0] = 1
# ValueError                               # Traceback (most recent call last)
# <ipython-input-54-6fd4c6570dd1> in <module>()
#     1 Z = np.zeros(10)
#     2 Z.flags.writeable = False
#     3 Z[0] = 1

# ValueError: assignment destination is read-only
#44
# [ 0.65226808  0.25610168  0.86237549  0.76006277  0.98972375  0.63456318
#   0.6723365   0.740228    0.29309949  0.47584925]
# [ 0.36772494  0.64228817  1.22453906  0.88014851  0.69715139  0.6963748
#   1.10921316  0.45026203  1.10296075  1.56786294]
Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print (R)
print (T)

#45
# Z = np.random.random(10)
# Z[Z.argmax()] = 0
# print (Z)

#46-------------
# [[(0.0, 0.0) (0.25, 0.0) (0.5, 0.0) (0.75, 0.0) (1.0, 0.0)]
#  [(0.0, 0.25) (0.25, 0.25) (0.5, 0.25) (0.75, 0.25) (1.0, 0.25)]
#  [(0.0, 0.5) (0.25, 0.5) (0.5, 0.5) (0.75, 0.5) (1.0, 0.5)]
#  [(0.0, 0.75) (0.25, 0.75) (0.5, 0.75) (0.75, 0.75) (1.0, 0.75)]
#  [(0.0, 1.0) (0.25, 1.0) (0.5, 1.0) (0.75, 1.0) (1.0, 1.0)]]
Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(Z)

#47-----------????
# X = np.arange(8)
# Y = X + 0.5
# C = 1.0 / np.subtract.outer(X, Y)
# print(np.linalg.det(C))


#48
# -128
# 127
# -2147483648
# 2147483647
# -9223372036854775808
# 9223372036854775807
# -3.40282e+38
# 3.40282e+38
# 1.19209e-07
# -1.79769313486e+308
# 1.79769313486e+308
# 2.22044604925e-16

for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)

for dtype in [np.float32, np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)

#49

np.set_printoptions(threshold=np.nan)
Z = np.zeros((16,16))
print (Z)

#50
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print (Z[index])

#51
# [((0.0, 0.0), (0.0, 0.0, 0.0)) ((0.0, 0.0), (0.0, 0.0, 0.0))
#  ((0.0, 0.0), (0.0, 0.0, 0.0)) ((0.0, 0.0), (0.0, 0.0, 0.0))
#  ((0.0, 0.0), (0.0, 0.0, 0.0)) ((0.0, 0.0), (0.0, 0.0, 0.0))
#  ((0.0, 0.0), (0.0, 0.0, 0.0)) ((0.0, 0.0), (0.0, 0.0, 0.0))
#  ((0.0, 0.0), (0.0, 0.0, 0.0)) ((0.0, 0.0), (0.0, 0.0, 0.0))]

Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                  ('y', float, 1)]),
                   ('color',    [ ('r', float, 1),
                                  ('g', float, 1),
                                  ('b', float, 1)])])
print (Z)

#52
Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0], Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print (D)

#53
Z = np.arange(10, dtype=np.int32)
Z = Z.astype(np.float32, copy=False)
print (Z)

#54----------------???
#55
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
    print (index, value)
for index in np.ndindex(Z.shape):
    print (index, Z[index])
'''


#56
# X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
# D = np.sqrt(X*X+Y*Y)
# sigma, mu = 1.0, 0.0
# G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
# print (G)

#57
# n = 10
# p = 3
# Z = np.zeros((n,n))
# np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
# print (Z)

#58
# X = np.random.rand(5, 10)
# # Recent versions of numpy
# Y = X - X.mean(axis=1, keepdims=True)
# print(Y)
# # 方法2
# # Older versions of numpy
# Y = X - X.mean(axis=1).reshape(-1, 1)
# print (Y)

#59
# Z = np.random.randint(0,10,(3,3))
# print (Z)
# print (Z[Z[:,1].argsort()])

#60
# Z = np.random.randint(0,3,(3,10))
# print ((~Z.any(axis=0)).any())

#61
# Z = np.random.uniform(0,1,10)
# z = 0.5
# m = Z.flat[np.abs(Z - z).argmin()]
# print (m)

#62
# A = np.arange(3).reshape(3,1)
# B = np.arange(3).reshape(1,3)
# it = np.nditer([A,B,None])
# for x,y,z in it: 
#     z[...] = x + y
# print (it.operands[2])

#63
# class NamedArray(np.ndarray):
#     def __new__(cls, array, name="no name"):
#         obj = np.asarray(array).view(cls)
#         obj.name = name
#         return obj
#     def __array_finalize__(self, obj):
#         if obj is None: return
#         self.info = getattr(obj, 'name', "no name")

# Z = NamedArray(np.arange(10), "range_10")
# print (Z.name)

#64
# Z = np.ones(10)
# I = np.random.randint(0,len(Z),20)
# Z += np.bincount(I, minlength=len(Z))
# print(Z)
# # 方法2
# np.add.at(Z, I, 1)
# print(Z)

#65
#X = [1,2,3,4,5,6]
# I = [1,3,9,3,4,1]
# F = np.bincount(I,X)
# print (F)

#66
# w,h = 16,16
# I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
# #Note that we should compute 256*256 first. 
# #Otherwise numpy will only promote F.dtype to 'uint16' and overfolw will occur
# F = I[...,0]*(256*256) + I[...,1]*256 +I[...,2]
# n = len(np.unique(F))
# print (n)

#67
#A = np.random.randint(0,10,(3,4,3,4))
# # solution by passing a tuple of axes (introduced in numpy 1.7.0)
# sum = A.sum(axis=(-2,-1))
# print (sum)
# # 方法2
# sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
# print (sum)

#68
# D = np.random.uniform(0,1,100)
# S = np.random.randint(0,10,100)
# D_sums = np.bincount(S, weights=D)
# D_counts = np.bincount(S)
# D_means = D_sums / D_counts
# print (D_means)
# # 方法2
# import pandas as pd
# print(pd.Series(D).groupby(S).mean())

#69
# A = np.random.uniform(0,1,(5,5))
# B = np.random.uniform(0,1,(5,5))
# # slow version
# np.diag(np.dot(A, B))
## 方法2
# # Fast version
# np.sum(A * B.T, axis=1)
## 方法3
# # Faster version
# np.einsum("ij,ji->i", A, B)

#70
# Z = np.array([1,2,3,4,5])
# nz = 3
# Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
# Z0[::nz+1] = Z
# print (Z0)

#71
# A = np.ones((5,5,3))
# B = 2*np.ones((5,5))
# print (A * B[:,:,None])

#72
# A = np.arange(25).reshape(5,5)
# A[[0,1]] = A[[1,0]]
# print (A)

#73
# faces = np.random.randint(0,100,(10,3))
# F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
# F = F.reshape(len(F)*3,2)
# F = np.sort(F,axis=1)
# G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
# G = np.unique(G)
# print (G)

#74
# C = np.bincount([1,1,2,3,4,4,6])
# A = np.repeat(np.arange(len(C)), C)
# print (A)

#75
# def moving_average(a, n=3) :
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n
# Z = np.arange(20)

# print(moving_average(Z, n=3))


#76
#from numpy.lib import stride_tricks

# def rolling(a, window):
#     shape = (a.size - window + 1, window)
#     strides = (a.itemsize, a.itemsize)
#     return stride_tricks.as_strided(a, shape=shape, strides=strides)
# Z = rolling(np.arange(10), 3)

# print (Z)

#77
# Z = np.random.randint(0,2,100)
# np.logical_not(Z, out=Z)
# Z = np.random.uniform(-1.0,1.0,100)
# np.negative(Z, out=Z)

#78
# def distance(P0, P1, p):
#     T = P1 - P0
#     L = (T**2).sum(axis=1)
#     U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
#     U = U.reshape(len(U),1)
#     D = P0 + U*T - p
#     return np.sqrt((D**2).sum(axis=1))

# P0 = np.random.uniform(-10,10,(10,2))
# P1 = np.random.uniform(-10,10,(10,2))
# p  = np.random.uniform(-10,10,( 1,2))

# print (distance(P0, P1, p))

#79
# based on distance function from previous question
# P0 = np.random.uniform(-10, 10, (10,2))
# P1 = np.random.uniform(-10,10,(10,2))
# p = np.random.uniform(-10, 10, (10,2))
# print (np.array([distance(P0,P1,p_i) for p_i in p]))

#80
#Z = np.random.randint(0,10,(10,10))
# shape = (5,5)
# fill  = 0
# position = (1,1)

# R = np.ones(shape, dtype=Z.dtype)*fill
# P  = np.array(list(position)).astype(int)
# Rs = np.array(list(R.shape)).astype(int)
# Zs = np.array(list(Z.shape)).astype(int)

# R_start = np.zeros((len(shape),)).astype(int)
# R_stop  = np.array(list(shape)).astype(int)
# Z_start = (P-Rs//2)
# Z_stop  = (P+Rs//2)+Rs%2

# R_start = (R_start - np.minimum(Z_start,0)).tolist()
# Z_start = (np.maximum(Z_start,0)).tolist()
# R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
# Z_stop = (np.minimum(Z_stop,Zs)).tolist()

# r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
# z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
# R[r] = Z[z]
# print (Z)
# print (R)

#81
# Z = np.arange(1,15,dtype=np.uint32)
# R = stride_tricks.as_strided(Z,(11,4),(4,4))
# print (R)

#82
# Z = np.random.uniform(0,1,(10,10))
# U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
# rank = np.sum(S > 1e-10)
# print (rank)

#83
# Z = np.random.randint(0,10,50)
# print (np.bincount(Z).argmax())

#84
# Z = np.random.randint(0,5,(10,10))
# n = 3
# i = 1 + (Z.shape[0]-3)
# j = 1 + (Z.shape[1]-3)
# C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
# print (C)

#85
# class Symetric(np.ndarray):
#     def __setitem__(self, index, value):
#         i,j = index
#         super(Symetric, self).__setitem__((i,j), value)
#         super(Symetric, self).__setitem__((j,i), value)

# def symetric(Z):
#     return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)

# S = symetric(np.random.randint(0,10,(5,5)))
# S[2,3] = 42
# print (S)

#86
# p, n = 10, 20
# M = np.ones((p,n,n))
# V = np.ones((p,n,1))
# S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
# print (S)
# It works, because:
# M is (p,n,n)
# V is (p,n,1)
# Thus, summing over the paired axes 0 and 0 (of M and V independently),
# and 2 and 1, to remain with a (n,1) vector.

#87
# Z = np.ones((16,16))
# k = 4
# S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
#                                        np.arange(0, Z.shape[1], k), axis=1)
# print (S)

#88
# def iterate(Z):
#     # Count neighbours
#     N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
#          Z[1:-1,0:-2]                + Z[1:-1,2:] +
#          Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

#     # Apply rules
#     birth = (N==3) & (Z[1:-1,1:-1]==0)
#     survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
#     Z[...] = 0
#     Z[1:-1,1:-1][birth | survive] = 1
#     return Z

# Z = np.random.randint(0,2,(50,50))
# for i in range(100): Z = iterate(Z)
# print (Z)

#89
# Z = np.arange(10000)
# np.random.shuffle(Z)
# n = 5

# # Slow
# print (Z[np.argsort(Z)[-n:]])
# # 方法2
# # Fast
# print (Z[np.argpartition(-Z,n)[:n]])

#90
# def cartesian(arrays):
#     arrays = [np.asarray(a) for a in arrays]
#     shape = (len(x) for x in arrays)

#     ix = np.indices(shape, dtype=int)
#     ix = ix.reshape(len(arrays), -1).T

#     for n, arr in enumerate(arrays):
#         ix[:, n] = arrays[n][ix[:, n]]

#     return ix

# print (cartesian(([1, 2, 3], [4, 5], [6, 7])))

#91
# Z = np.array([("Hello", 2.5, 3),
#               ("World", 3.6, 2)])
# R = np.core.records.fromarrays(Z.T, 
#                                names='col1, col2, col3',
#                                formats = 'S8, f8, i8')
# print (R)

#92
# x = np.random.rand()
# np.power(x,3)
## 方法2
# x*x*x
## 方法3
# np.einsum('i,i,i->i',x,x,x)

#93
# A = np.random.randint(0,5,(8,3))
# B = np.random.randint(0,5,(2,2))

# C = (A[..., np.newaxis, np.newaxis] == B)
# rows = np.where(C.any((3,1)).all(1))[0]
# print (rows)

#94
# Z = np.random.randint(0,5,(10,3))
# print (Z)

# # solution for arrays of all dtypes (including string arrays and record arrays)
# E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
# U = Z[~E]
# print (U)
# # 方法2
# # soluiton for numerical arrays only, will work for any number of columns in Z
# U = Z[Z.max(axis=1) != Z.min(axis=1),:]
# print (U)

#95
# I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
# B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
# print(B[:,::-1])
# # 方法2
# print (np.unpackbits(I[:, np.newaxis], axis=1))

#96
# Z = np.random.randint(0,2,(6,3))
# T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
# _, idx = np.unique(T, return_index=True)
# uZ = Z[idx]
# print (uZ)

#97
# A = np.random.uniform(0,1,10)
# B = np.random.uniform(0,1,10)
# print ('sum')
# print (np.einsum('i->', A))# np.sum(A)
# print ('A * B')
# print (np.einsum('i,i->i', A, B)) # A * B
# print ('inner')
# print (np.einsum('i,i', A, B))    # np.inner(A, B)
# print ('outer')
# print (np.einsum('i,j->ij', A, B))    # np.outer(A, B)

#98
# phi = np.arange(0, 10*np.pi, 0.1)
# a = 1
# x = a*phi*np.cos(phi)
# y = a*phi*np.sin(phi)

# dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
# r = np.zeros_like(x)
# r[1:] = np.cumsum(dr)                # integrate path
# r_int = np.linspace(0, r.max(), 200) # regular spaced path
# x_int = np.interp(r_int, r, x)       # integrate path
# y_int = np.interp(r_int, r, y)

#99
# X = np.asarray([[1.0, 0.0, 3.0, 8.0],
#                 [2.0, 0.0, 1.0, 1.0],
#                 [1.5, 2.5, 1.0, 0.0]])
# n = 4
# M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
# M &= (X.sum(axis=-1) == n)
# print (X[M])

#100
# X = np.random.randn(100) # random 1D array
# N = 1000 # number of bootstrap samples
# idx = np.random.randint(0, X.size, (N, X.size))
# means = X[idx].mean(axis=1)
# confint = np.percentile(means, [2.5, 97.5])
# print (confint)

