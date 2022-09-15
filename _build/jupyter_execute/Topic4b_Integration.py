#!/usr/bin/env python
# coding: utf-8

# # Topic 4b: Integration

# In[1]:


import scipy.integrate as integrate
import scipy.special as special
import numpy as np
import matplotlib.pyplot as plt


# The next obvious numerical tool you will encourter in numerical integration.  This is also a good time to introduce special functions, because often the hardest problem is how to integrate these kinds of special functions.

# Let's start with just doing a basic one-dimensional integral.

# In[2]:


integrate.quad(lambda x: np.sin(x),0,2*np.pi) 


# The first number it returns is the intergal.  The second is the estimated error. 

# It doesn't seem any harder to integrate some less well-known functions.  Here is a Bessel J function:

# In[3]:


xlist=np.arange(0,10,0.01)
plt.plot(xlist,special.jv(5,10*xlist))
plt.show()


# Here is the integral of the function

# In[4]:


integrate.quad(lambda x: special.jv(5,10*x),0,10) 


# We can even integrate all the way to $x = \infty$:

# In[5]:


integrate.quad(lambda x: np.exp(-x**3/2),0,np.inf) 


# However, you will encounter a lot more errors when you integrate to infinity:

# In[6]:


integrate.quad(lambda x: special.jv(5,10*x),0,np.inf) 


# Notice that there estimated error is large, so we should be careful about integrating to infinity. 

# ### Making our own integrator

# As with equation solving, we can learn a lot about the code by trying to do it ourselves.  At the end of the day, one-dimensional integration is just adding up a bunch of areas.    

# In[7]:


def my_integrate(f,a,b,dx):
    x=np.arange(a,b,dx)
    y=f(x)
    return (y.sum()-(y[0]+y[-1])/2.)*dx  # we are adding the ar


# In[8]:


my_integrate(lambda x: np.sin(x),0,2*np.pi,0.0000001)


# Okay, that seems like it is working okay.  Let's try it on a bit harder integral

# In[9]:


import time


# In[10]:


t1 = time.time()
print(my_integrate(lambda x: special.jv(5,10*x),0,10,0.000001))
print(time.time() - t1)


# In[11]:


t1 = time.time()
print(integrate.quad(lambda x: special.jv(5,10*x),0,10))
print(time.time() - t1)


# Wow!  Scipy got an answer 1000 times faster.  If we wanted to get an answer that fast we would lose a lot of accuracy

# In[12]:


t1 = time.time()
print(my_integrate(lambda x: special.jv(5,10*x),0,10,0.01))
print(time.time() - t1)


# ### Accuracy setttings

# We can guess that, although scipy is much faster, it must also have some kind of accuracy settings just like ours

# In[13]:


a=integrate.quad(lambda x: np.sin(100*x)*np.exp(-x**2),0,np.inf) 
a


# The exact answer is 0.010002001201201683031...

# In[14]:


exact = 0.01000200120120168303067014934894552744967


# In[15]:


a[0]-exact


# The first thing we can do is to allow a finer spacing on the points used in the itegral by increasing "limit" (default=50)

# In[16]:


b=integrate.quad(lambda x: np.sin(100*x)*np.exp(-x**2),0,np.inf,limit=10000) 
b


# In[17]:


b[0]-exact


# This helps a bit, but not by much.  The next thing we can do is try to force it to tolarate less error.

# In[18]:


c=integrate.quad(lambda x: np.sin(100*x)*np.exp(-x**2),0,np.inf,limit=1000,epsabs=1.49e-14, epsrel=1.49e-14) 
c


# In[19]:


c[0]-exact


# Unforunately, there are still plenty of functions that are very challenging in integrate numerically.  This is particularly a problem if we have to integrate a highly oscilatory function over a lot of cycles:

# In[20]:


fig,ax=plt.subplots(figsize=(12,6),ncols=2)
x_in=np.linspace(0,10,10000)
ax[0].plot(x_in,np.sin(100*x_in)*np.exp(-0.01*x_in))
ax[1].plot(x_in,np.sin(100*x_in)*np.exp(-0.01*x_in))
ax[1].set_xlim(0,1)


# We know how to integrate this function analytically so that $\int_0^\infty dx \sin(100 x) e^{-0.01 x}\approx 100/(100^2+0.01^2) \approx 0.01$.

# Let's try integrating this numerically as before with higher accuracy settings:

# In[21]:


integrate.quad(lambda x: np.sin(100*x)*np.exp(-0.01*x),0,np.inf,limit=100000,epsabs=1.49e-14, epsrel=1.49e-14) 


# The answer itself is is off by 24%, but the estimated error is very large (if we didn't know the exact answer, it would be hard to use the result). 

# One strategy we can try is splitting the integral into two regions, one finite one where we expect most of the area to lie:

# In[22]:


integrate.quad(lambda x: np.sin(100*x)*np.exp(-0.01*x),0,10000,limit=100000,epsabs=1.49e-14, epsrel=1.49e-14) 


# ... and one that is small

# In[23]:


integrate.quad(lambda x: np.sin(100*x)*np.exp(-0.01*x),10000,np.inf,limit=100000,epsabs=1.49e-14, epsrel=1.49e-14) 


# For our purposes this error is totally fine (the integral is very small, which is all we need).  However, we can sometimes get better performance by rescaling the intergation variable:

# In[24]:


integrate.quad(lambda x: np.sin(1000000*x)*np.exp(-100*x)/10000.,1,np.inf,limit=100000,epsabs=1.49e-14, epsrel=1.49e-14) 


# This trick is only helpful when integrating to infinity.  On any finite domain, we are usually doing a Riemann sum that doesn't care about the lables of your variable.  When integrating to infinity, the computer is doing a bit more under the hood and so sometimes you have to trick it into doing the problem in an easier way.

# ## Multi-dimensional Integration

# Multi-dimension integration is where we see the real need to use the lambda notation. Suppose we want to perform an integral: $$\int_0^1 dx \int_0^x dy \sin(x y)$$  If we thought of this as two seperate integral, the boundaries of integration in x become part of the function we integrate in y. 

# In[25]:


integrate.dblquad(lambda y,x: np.sin(y*x),0,1,lambda x:0,lambda x:x)


# Okay - so we can do this kind of integral no problem.  But what exactly is going on?  Let's start by separating off the function we want to integrate:

# In[26]:


def f2d(y,x):
    return np.sin(x*y)


# In[27]:


integrate.dblquad(f2d,0,1,lambda x:0,lambda x:x)


# Okay, so everything else has to be the limits of integration.  Let's also give those names so that we have 
# $$\int_a^b dx \int_{g(x)}^{h(x)} dy \sin(x y)$$

# In[28]:


a=0
b=1
def g(x):
    return 0.
def h(x):
    return x


# In[29]:


integrate.dblquad(f2d,a,b,g,h)


# This way of doing it makes it more clear what we are doing.  In particular, y is the variable of integration that runs to enpoints that are functions of x.  Therefore, we have to feed it a function, even if we only want that function to return 1.  For example, to implement 
# $$\int_0^1 dx \int_{0}^{1} dy \sin(x y)$$
# we still need to include either functions or lambda operators because they still could have been interesting functions of x

# In[30]:


integrate.dblquad(lambda y,x: np.sin(y*x),0,1,lambda x:0,lambda x:1)


# We could have also tried to do this as two different one dimensional integrals.  As one goes to higher dimensional integration, the integrals will often be too slow to calculate using some generic tool like quad and it can be more useful to precompute some of the integrals first.  

# Let's try this on this example by calculating the first integral at a few points:

# In[31]:


xlist=np.arange(-0.01,1.01,0.01)
int_x=np.zeros(len(xlist))
for i in range(len(xlist)):
    int_x[i]=integrate.quad(lambda y: np.sin(y*xlist[i]),0,xlist[i])[0]


# In[32]:


plt.plot(xlist,int_x)
plt.show()


# In[33]:


from scipy import interpolate


# In[34]:


int_func= interpolate.interp1d(xlist,int_x,kind='cubic')


# In[35]:


integrate.quad(int_func,0,1)


# In[36]:


dlbquad_sol=integrate.dblquad(f2d,0,1,lambda x:0,lambda x:x)[0]
quad_int_sol=integrate.quad(int_func,0,1)[0]


# In[37]:


dlbquad_sol-quad_int_sol


# We see that we get a surprisingly accurate answer.  This is a somewhat helpful property of integration: when we calculate the area, small random errors tend to average out. In many circumstances, numerical integration is a highly effective way to find your answers.

# ## Summary

# Numerical integration is a powerful tool that is fairly easy to use and understand.  When integrating over a few variables, on a finite domain, it is often pretty straightforward to get accurate answers.  We saw it is a bit trickier on infinite domains.  We will see in a later lecture that it also becomes difficult when we increase the number of variables we are integrating over (dimensionality).

# In[ ]:




