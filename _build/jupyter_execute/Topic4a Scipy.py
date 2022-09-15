#!/usr/bin/env python
# coding: utf-8

# # Topic 4: Scipy and Numerical Techniques

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# We are now ready to apply some of our basic know-how to basic numerical problems like finding solutions of equations, ODEs, and integrating.   

# Before we get started with examples, we have to learn a useful trick involving lambda functions.  There is some deep origin of this idea in computer science, but our superficial level, they give us a way of defining a function while we are doing something else

# Here for example, is a function that adds x and y

# In[2]:


def add(x,y):
    return x+y
add(1,2)


# We can do the same wit a lambda function

# In[3]:


lambda x,y:x+y


# So for I have just created the fucntion without giving it a name.  I can act on it too by

# In[4]:


(lambda x,y:x+y)(3,5)


# So we can think of lambda x,y: as the same thing as add(x,y) but we never stopped to define it.  At this moment, it might seem stange that we would need such a thing.  However, when it comes to solving equations, the arguements of our numerical technique is a function! We integrate functions and minimize functions, for example, so if we have a code that does that, the input is a function, not a piece of data.  lambda lets us do this easily.

# ### Newton's method

# The best way to understand both how these libraries work and what they do is to try to solve a simple equation numerically. For example, let's that the quadratic equation $x^2 +2x +1 =0$.  We know the solution, $x=-1$ already, but those will be good checks our method

# In[5]:


from scipy import optimize


# In[6]:


def f(x):
    return x**2+2*x+1


# In[7]:


optimize.newton(f,1)


# Okay, so it finds the solution pretty well.  We can see we could have done the same directly with our lambda function

# In[8]:


optimize.newton(lambda x:x**2+2*x+1,1)


# This is why it is nice to have these lambda functions, because it saves us space so that we aren't defining all kinds of boring functions.  We save our functions for things that are actually useful

# Combining back to our problem: what do we think of this answer?  This is a pretty simple equation and while we are close, this is not just some machine precision error, since there are many incorret digits.  What is going on and how can we do better?

# Let's make our own Newton's method code and figure out what's going on.  First, this piece code took a function as an input.  So we must be able to define a function where the name of the function is the input:

# In[9]:


def test_func(func): #func is the name of some function that take 2 inputs
    return func(1,2) #return the function evaluated at (1,2)


# In[10]:


test_func(add)  # Yay! It works


# When a one function uses another function as its arguement, that is when we can use the lambda function instead.  It

# In[11]:


test_func(lambda x,y:x**2+100*y)


# So we know how to feed a function to another function.  But what exactly are we trying to do? Newton's method is supposed to take our function and approximate it as a line based on the point we start $x_0$, 
# $$y = f'(x_0)(x-x_0) + f(x_0)$$  
# It then finds the point where $y=0$, we'll call that $x_1$, whose solution is
# $$x_1 = x_0 - f(x_0)/f'(x_0)$$
# We can guess this is probably close to the solution than where we started, so let's use repeat with $x_1 \to x_0$

# Problem: if I want to make a code that does this, I need to tell it when to stop.  Otherwise it will just repeat forever.  I that two options: stop when my answer is close enough to zero or stop after some fixed number of steps (or what ever comes first)

# In[12]:


def my_newton(f,fprime,x0,err):
    x1=x0-f(x0)/float(fprime(x0)) # new point according to newton
    if abs(f(x1))>err:  # need the absolute value here
        print(f(x1))
        return my_newton(f,fprime,x1,err) # repeat if the error is too large
    else:
        return x1 # return x1 when we are happy


# In[13]:


my_newton(f,lambda x:2*x+2,10,0.0000001)


# Wow!  That's pretty cool.  So what did we learn? This method needs two things that we didn't feed the scipy function: (a) the derivative of f and (b) a measure of when to stop.  Presumably those are options inside the function that we didn't use?

# Scipy isn't going to calculate derivatives for us, so it must be approximating the derivative some other way.  Indeed that is what's happening: if you don't provide it with the derivative, it uses something called the secant method.  To use the true Newton's method, you need to give it the derivative as well:

# In[14]:


newton1=optimize.newton(lambda x:x**2+2*x+1,10,fprime=lambda x:2*x+2)
print(newton1)


# In[15]:


sectan1=optimize.newton(lambda x:x**2+2*x+1,10)
print(sectan1)


# In[16]:


abs(newton1-sectan1)


# First,let's make a guess that the error is set at $10^{-16}$ because we are using 16 bit numbers:

# In[17]:


myresult=my_newton(f,lambda x:2*x+2,10,10**(-16))
print(myresult)


# In[18]:


newton1-myresult


# This seems like a good guess, but when we look at the documention, it seems more like what they did was stop when the difference between x1 and x0 gets small.  It takes an optional input called "tol" whose default is 1.48e-08.  Let's see what happens if we do that instead:

# In[19]:


def my_newton2(f,fprime,x0,err):
    x1=x0-f(x0)/float(fprime(x0)) # new point according to newton
    if abs(x1-x0)>err:  # need the absolute value here
        print(f(x1))
        return my_newton2(f,fprime,x1,err) # repeat if the error is too large
    else:
        return x1


# In[20]:


myresult2=my_newton2(f,lambda x:2*x+2,10,1.48e-08)
print(myresult2)


# In[21]:


newton1


# In[22]:


newton1-myresult2


# We hit exactly the same number as Scipy! We can even get back to our old answer in scipy by replacing 

# In[23]:


newton2=optimize.newton(lambda x:x**2+2*x+1,10,fprime=lambda x:2*x+2,tol=10**(-16))


# In[24]:


newton2-myresult


# Now the one thing that is nice about scipy is that it has more methods built in. We have a pretty good idea of what scipy is doing, although it is definitely coded up in a better way that is probably faster and can handle situations where we don't actually converge (e.g. scipy has a max number of iterations so it doesn't get stuck in an infinite loop).  Yet, based on what we have seen, we can imagine that if we gave it more information about the function, it could probably do better.  For example, we could improve Newton's method to use the second derivative too!  Scipy has this built in for us

# In[25]:


fprime2=optimize.newton(lambda x:x**2+2*x+1,10,fprime=lambda x:2*x+2,fprime2=lambda x:2.,tol=10**(-16))
print(fprime2)


# In[26]:


print(fprime2+1)
print(newton1+1)


# For all our effort, it seems like we have hit a wall.  We can't do much better than what we have.  What's the problem?  Right now we have hit the point where the difference between x1 and x0 is 10^(-16) which is around the machine error for generic floating point numbers.  Presumably if we want to do better, we are going to have to make the working precision higher.

# In[27]:


my_newton2(f,lambda x:2*x+2,10,1.48e-64)


# Let's try increasing the working precision of our code using high precision numbers

# In[28]:


from decimal import Decimal


# In[29]:


Decimal(0.1)


# In[30]:


def my_newton3(f,fprime,x0,err):
    xs=Decimal(x0)
    x1=xs-Decimal(f(xs))/Decimal(fprime(xs)) # new point according to newton
    if abs(Decimal(f(x1)))>Decimal(err):  # need the absolute value here
        print(Decimal(f(x1)))
        return my_newton3(f,fprime,x1,err) # repeat if the error is too large
    else:
        return x1


# In[31]:


my_newton3(f,lambda x:2*x+2,10,1.48e-64)


# In[32]:


Decimal(newton1+1)


# In[33]:


Decimal('-0.9999999999999890669998056123')-Decimal(-1)


# So what are the morals of this exercise: 
# 
# (1) you may not know exactly how things are coded inside numpy or scipy, you should be able to get a pretty good idea how they work 
# 
# (2) numerical techniques always have some error associated with them.  You can learn a lot about a code by understanding what controls the size of the error and trying to make it smaller 
# 
# (3) there are lots more options built into a scipy solver than you would want to code yourself.  But, the simplest implementations are often pretty close to what you do know how to code up yourself.  The complicated options are usually just the simple idea extended to higher order (more derivatives, more grid points, higher order polynomial extrolations, etc)

# ## Interpolation

# In[34]:


from scipy import interpolate


# Sometimes you don't actually have the analytic form of a function you want to use. Instead you might have some raw data.  Alternatively, you have the function but calculating it is so slow you need to pre-compute it at a bunch of points before you can use it in some other applications (like integrating, equation solving, etc). 

# This is when interpolation is super helpful.  Interpolation is just a rule for filling in the gaps in your function based on the data you have.  

# In[35]:


xI=np.linspace(0,np.pi,40)
yI=np.sin(2*xI**2)


# In[36]:


plt.plot(xI,yI)
plt.show()


# So that is our data.  Now we want to make a function that is defined on the entire range between the endpoints of our data

# Let's just try the 1d interpolate function out of the box and see what it does

# In[37]:


f_try = interpolate.interp1d(xI, yI)


# In[38]:


x_fine=np.linspace(0,np.pi,400)


# In[39]:


plt.plot(xI,yI)
plt.plot(x_fine,f_try(x_fine))
plt.show()


# Okay - so it looks like our default function is just drawing straight lines between the points.  This is exaclty what our plot is doing, and so we get the same sharp corners.  But if we fit with a polynomial between several points, we will get a smoother looking result.  E.g. using a cubic polynomial:

# In[40]:


f_cube= interpolate.interp1d(xI, yI,kind='cubic')


# In[41]:


plt.plot(xI,yI)
plt.plot(x_fine,f_cube(x_fine))
plt.show()


# In[42]:


plt.plot(x_fine,f_try(x_fine)-np.sin(2*x_fine**2))
plt.plot(x_fine,f_cube(x_fine)-np.sin(2*x_fine**2),color='darkorange')
plt.show()


# In[43]:


plt.plot(x_fine,f_cube(x_fine)-np.sin(2*x_fine**2),color='darkorange')
plt.show()


# So we see that our initial function was making about a 10% error.  Using a cubic instead, we get less than 1%.  In fact, for the early points we do really well:

# In[44]:


plt.plot(x_fine,np.abs(f_cube(x_fine)-np.sin(2*x_fine**2)))
plt.yscale('log')
plt.show()


# So we are getting 1 part in a 100 000 up to about $x=1$ using only 40 data points for the full range of $[0,\pi]$

# Notice that the interpolation is a function.  So we can also use this function inside of our solving code:

# In[45]:


optimize.newton(f_cube,1)


# In[46]:


np.sin(2*1.2532996656450606**2)


# ### Operations on functions / interpolations

# One of the central reasons to use a spline is to a approximate a quantity we don't know exactly.  For example, Scipy has a built in function that computes a numerical derivative

# In[47]:


from scipy.misc import derivative


# In[48]:


def g(x):
    return x**3


# In[49]:


derivative(g,1,dx=1e-5)


# We could make our own very simple version:

# In[50]:


def my_derivative(g,x0,dx):
    return (g(x0+dx)-g(x0-dx))/(2*dx)


# In[51]:


my_derivative(g,1,dx=1e-5)


# We matched the result exactly.  Of course, that is just the default, derivative has many ways to improve. E.g. we can include more than just the difference of two poitns but include more points:

# In[52]:


derivative(g,1,dx=1e-5,order=7)


# Of course, we notice that this just returns a number for a single point.  However, if we wanted the derivative as a function we could just take a bunch of derivatives and make an interpolating function:

# In[53]:


def h(x):
    return np.sin(2*x**2)


# In[54]:


dh=np.zeros(len(xI))
for i in range(len(xI)):
    dh[i]=derivative(h,xI[i],dx=1e-5,order=7)
dh_fnc=interpolate.interp1d(xI,dh,kind='cubic')


# In[55]:


plt.plot(xI,h(xI))
plt.plot(xI,dh_fnc(xI),color='red')


# Now we can look for maxima

# In[56]:


max_h=optimize.newton(dh_fnc,1.2)
print(max_h)


# In[57]:


plt.plot(xI,h(xI))
plt.plot([max_h,max_h],[-1,1.5])
plt.xlim(0.5,1.1)
plt.ylim(0.5,1.1)
plt.show()


# ## Special Functions

# A common thing one encounters in physics is the need to use special functions.  Many of these are implemented in scipy for us:

# In[58]:


from scipy import special as sp


# In[59]:


x=np.linspace(-10,1,102)
plt.plot(x,sp.gamma(x))


# In[60]:


x=np.linspace(0,100,102)
plt.plot(x,sp.y0(x))


# We don't have a particular use for them right now, but this is something you will need in most scientific applications.

# It is also worth noting that these functions are defined in terms of complex arguments as well:

# In[61]:


sp.gamma(-1+0.1j)


# The behavior of functions for complex arguments is a very important subject. 

# In[62]:


x=y=np.linspace(-1,1,102)
X,Y=np.meshgrid(x,y)
plt.pcolor(X,Y,np.real(sp.hankel1(0.2,X+(0+1j)*Y)),cmap='RdBu',shading='auto')


# Scipy does not define all special functions for complex values:

# In[63]:


sp.zeta(1j)


# ## Summary

# We are going to see that scipy has a lot of built in code that is extremely powerful.  It is essentially built on top of numpy so they tend to work very well together.  What we hopefully learned in these examples is that there are really two things going on inside scipy: (1) an algorithm and (2) an implementation of the algorithm in terms of code.  The algorithm works on some premise that we can and should understand.  We can try our own version of the algorithm to get a deeper understanding of what scipy is doing.  Fundamentally, scipy is not doing anything deeper or more sophisticed than what we can code ourselves.  What scipy has really done is code it in a very fast way (probably using some neat python/C tricks we don't know) and with lots of built in options that would be time consuming for us to do ourselves.

# In[ ]:




