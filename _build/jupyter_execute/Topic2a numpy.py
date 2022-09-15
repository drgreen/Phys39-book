#!/usr/bin/env python
# coding: utf-8

# # Topic 2:  numpy

# We are now moving on the useful libraries for scientific computing.  The first one of interest is numpy.  We load it like this:

# In[1]:


import numpy as np


# numpy is filled with basic math functions that you will use all the time

# In[2]:


print(np.sin(2),np.cos(np.pi))


# Let's make our life easier and define

# In[3]:


pi=np.pi


# The most basic object in numpy is the numpy array.  In some ways, this is just like a list, but it is designed so you can do lots of math operations quickly:

# In[4]:


a=np.array([1,4,1,5,15,20,22,1,45,3])
print(a[0], a[-1], a[3])


# In[5]:


b=np.ones(len(a))*35.1
print(b)


# In[6]:


print(a+b)
print(a*b)
print(a**3)
print(b**3)


# Notice that this is NOT doing matrix or vector mupliplication or addition.  It is performing the operation on each element of the list.  For this to work, it is essential that a and b are the same length.  But, it does understand what you mean if you act with number.

# In[7]:


print(a*4)
print(a+4)


# But if you have two different sized arrays is doesn't know what to do

# In[8]:


c=np.array([2,4,2])
a+c


# Numpy arrays actually have shapes so that they are more like matrices, tensors, etc.

# In[9]:


d=np.array([[1,2],[1,2]])
e=np.ones((3,3,3))
d.shape


# In[10]:


e=np.arange(9)


# In[11]:


e.reshape((3,3))


# In[12]:


e=np.arange(27)
f=e.reshape((3,3,3))
print(f)


# In[13]:


f[0,2]


# In[14]:


f**2


# These definitions are really useful because you can write a function like it acts on a single number, but, if you're careful, it can act on the full array:

# In[15]:


def position(t,x0,v,a):
    position=x0+v*t+a*t**2/2.
    return position


# In[16]:


t=np.linspace(0,100,1000)


# In[17]:


d=position(t,50,-100.,3.)


# In[18]:


import matplotlib.pyplot as plt


# In[19]:


plt.plot(t,d)
plt.show()


# We can also try something more complicated where we solve more than one problem at a time:

# In[20]:


tarray=np.array((t,t,t))
print(tarray.shape)
varray=np.zeros(tarray.shape)
aarray=np.ones(tarray.shape)


# In[21]:


varray[1,:]=np.ones(len(t))
varray[2,:]=np.ones(len(t))*2
aarray[1,:]=np.ones(len(t))*2
aarray[2,:]=np.ones(len(t))*3


# In[22]:


darray=position(tarray,0,varray,aarray)


# In[24]:


plt.plot(t,darray[0,:])
plt.plot(t,darray[1,:])
plt.plot(t,darray[2,:])
plt.show()


# The key advantages of numpy are the orgnization (like the above example) and the speed.  numpy is essentially a bunch of code that is commonly used, but written in C because it is faster than python.  We can see this explicitely with an example

# In[25]:


import time


# In[29]:


def trad_version():
    t1 = time.time() # what is the time when I start
    X = range(int(1e7)) # arrange numbers 0, 10M -1
    Y = range(int(1e7)) # arrange numbers 0, 10M -1
    Z = [] 
    for i in range(len(X)):
        Z.append(X[i] + Y[i]) # make a new list by adding the elements of X and Y
    return time.time() - t1 # what is the difference from the time when I end and started


# In[30]:


def numpy_version():
    t1 = time.time() # what is the time when I start
    X = np.arange(int(1e7)) # arrange numbers 0, 10M -1
    Y = np.arange(int(1e7))# arrange numbers 0, 10M -1
    Z = X + Y # make a new list by adding the elements of X and Y
    return time.time() - t1 # what is the difference from the time when I end and started


# In[31]:


print(trad_version(),numpy_version())


# So we see that numpy about 25x faster than doing the same thing with a list.  The commands are all very similar too.

# ### Some basic numpy tools

# A very convenient tool is to be able to make arrays of a given size and shape quickly.  Here are a few very common examples:

# In[35]:


print(np.arange(10))
print(np.arange(3,14))
print(np.arange(3,14,2))
print(np.arange(0,1,0.1))


# In[36]:


print(np.linspace(2,5,10))
print(np.linspace(2,5,10,endpoint=False))


# In[37]:


print(np.logspace(0,2,10))


# In[39]:


print(np.ones((3,3)))
print(np.ones(10))
print(np.zeros((3,3)))
print(np.zeros(10))


# In[40]:


np.diag((1,2,3))


# In[41]:


np.identity(5)


# All of these quickly generate a numpy array of a given shape. arange, linspace and logspace quickly give you an ordered list of numbers.  When you are sampling value of a function, solving a differential equation, etc. these all come in handy. 

# ones is often useful if you just one to have a constant array, e.g. if you want to plot a horizonal line:

# In[42]:


x=np.arange(0,1,0.025)
plt.plot(x,np.ones(len(x))*20)


# zeros can be useful just to create an array of a given shape that you want to populate with values later.  E.g. I want to find a value to put in, but it isn't easily expressable as an operation on the arrays

# In[46]:


y=np.zeros(len(x))
for i in range(len(x)):
    y[i]=x[i:].sum() # add up all the numbers in x, starting at the ith location


# In[47]:


print(y)


# zeros is also useful because you can quickly see if you did something wrong if there are too many entries that are still zero.  

# ## Meshgrids

# A useful tool when dealing with numpy arrays is the meshgrid.  It allows us to take a multiple vectors and use them to define a higher dimensional space.  The obviuos example is that you have variables x, y and you want to define a function of (x,y).  E.g. Let's define an electric field in terms of x and y on a grid

# In[33]:


Lx=1.
Ly=2.
n=5
x_array=np.linspace(0,Lx,n,endpoint=True)
y_array=np.linspace(0,Ly,n,endpoint=True)


# so x_array and y_array are just 1d numpy arrays that list the x and y coordinates we want.  Now suppose you want $E_x = 5 \sin(2. \pi x/ Lx) \cos(2. \pi y/Ly)$.  You can't just multiply np.sin(x_array) np.cos(y_array) because that will return a 1d array of n values.  You want it to give a array of (n,n) as the shape (i.e. it gives a different value for each x and y you use.  So we make a meshgrid

# In[38]:


x,y=np.meshgrid(x_array,y_array,indexing='ij')


# In[39]:


print(x)
print(x[:,0])
print(x[0,:])


# In[53]:


print(y)
print(y[:,0])
print(y[0,:])


# So now we can define the x-component of the electric field just by multiplying the arrays:

# In[54]:


Ex=5*np.sin(2.*pi*x/ Lx)*np.cos(2.*pi*y/Ly)


# In[55]:


Ex


# In[56]:


Ey=10.*np.sin(2.*pi*y/Ly)


# In[57]:


plt.figure()
plt.quiver(x, y, Ex, Ey, units='width')
plt.show()


# Now let's try in for 3d (x,y,z).  Let's just make a temperature in a box so that $T(x,y,z)= (x+2y) z$.  So we want to add a z-variable and make a higher dimensional array:

# In[59]:


Lz=3
z_array=np.linspace(0,Lz,n,endpoint=True)


# In[60]:


x,y,z=np.meshgrid(x_array,y_array,z_array)


# In[61]:


print(x[0,:,0])
print(y[:,0,0])
print(z[0,0,:])


# In[62]:


T=(x+2*y)*z


# In[63]:


T


# Now we see that there is an easier way we could find the distance for a bunch of different, time, velocities and accelertaions all at once!

# In[67]:


num=20
t_list=np.linspace(0,10,100)
v_list=2*np.linspace(0,10,num)
a_list=3*np.linspace(0,1,num)


# In[68]:


print(a_list)


# In[74]:


t,v,a=np.meshgrid(t_list,v_list,a_list)


# In[75]:


final_d=position(t,0,v,a)


# In[76]:


final_d[1,:,2]


# In[77]:


t[0,:,0]


# In[78]:


for i in range(num):
    for j in range(num):
        plt.plot(t_list,final_d[i,:,j])


# Look, we found 400 solution at 100 points in time and plotted all of them without breaking a sweat!

# In[ ]:




