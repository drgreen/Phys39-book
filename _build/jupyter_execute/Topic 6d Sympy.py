#!/usr/bin/env python
# coding: utf-8

# ## Symbolic Computing

# In[1]:


import sympy
sympy.init_printing()
import numpy as np


# The emphasis in this course has been the use of computers for manipulting data or using numerical techniques to solve problems. These are central problems in scientific computing and are good windows into how computers work.
# 
# Of course, computers can be designed to do much more complex tasks and we should use computers when they are available to simplify/speed up/check our work. Symbolic computing is an excellent example of a computer resource that can assist in solving problems that we might usually associate with a human and a piece of paper. Fundamentally, the computer can do a lot of our math for us, so we should use it to check our results at the very least.

# Sympy is a python library that can do a lot of math for us, in this way.  It is similar to mathematica in both function and design, but is freely available. One of the most basic sources of confusion is that these symbolic tools try to avoid making any assumptions:

# In[2]:


x=sympy.Symbol('x')


# In[3]:


sympy.sqrt(x**2)


# Sympy does not know how to simplify this expression. What if $x$ is a complex number? We will get different answers depending on our assumptions:

# In[4]:


y=sympy.Symbol('y',real=True)


# In[5]:


sympy.sqrt(y**2)


# In[6]:


y=sympy.Symbol('y',positive=True)


# In[7]:


sympy.sqrt(y**2)


# In[8]:


n1=sympy.Symbol('n1',integer=True)
n2=sympy.Symbol('n2',odd=True)


# In[9]:


sympy.cos(sympy.pi*n1)


# In[10]:


sympy.cos(sympy.pi*n2)


# ## Manipulating Expressions

# We can define experessions you might want to work with

# In[11]:


exp=x**2+x*(x*2+x*2)+x*(x+1)+3


# In[12]:


exp


# We can simplify these expressions two ways:

# In[13]:


sympy.simplify(exp)


# In[14]:


exp.simplify()


# Similarly we can factor expressions

# In[15]:


exp2=x**2+2*x+1


# In[16]:


exp2


# In[17]:


sympy.factor(exp2)


# In[18]:


exp2.factor()


# or expand

# In[19]:


((x+2)**5).expand()


# In[20]:


sympy.sin(x+y).expand(trig=True)


# ### Evaluating Expressions 

# In a lot of situations, we need to combine our symbolic work with something numerical.  Sometimes, this means we just want to force the computer to return a demical instead of a symbolic expression for the number:

# In[21]:


sympy.N(sympy.pi)


# In[22]:


sympy.N(sympy.pi,50) # tell it you want 50 decimal places


# In[23]:


sympy.cos(1)


# In[24]:


sympy.N(sympy.cos(1))


# In other situations, we have found a function that we want to be able to evaluate numerically:

# In[25]:


exp


# We can just substitute $x$ with a number (or another variable) if we like

# In[26]:


exp.subs(x,10)


# But we can also define an actual function we can use as normal with lambdify

# In[27]:


exp_fnc=sympy.lambdify(x,exp)


# In[28]:


exp_fnc(10)


# In[29]:


x_array=np.linspace(1,10,10)


# In[30]:


exp_fnc(x_array)


# In[31]:


exp3=sympy.sin(x**2)


# In[32]:


exp_fnc3=sympy.lambdify(x,exp3)


# In[33]:


exp_fnc3(x_array)


# ### Solving Equations

# Not suprisingly, sympy is pretty good at solving equations.

# In[34]:


sympy.solve(x**2-1)


# In[35]:


sympy.solve(sympy.sin(x))


# In[36]:


sympy.solve([x+y+1,x+2*y],[x,y])


# ## Calculus

# Calculus is one of the symbolic tools that tends to be very powerful and reliable. In some cases it would be enormously difficult to do it yourself yet the computer seems to find a nice answer quickly.

# #### Derivatives

# In[37]:


exp4=sympy.sin(x*sympy.cos(x**2 +x+2))
exp4


# In[38]:


sympy.diff(exp4)


# In[39]:


exp4.diff(x)


# In[40]:


exp4.diff(x,x).simplify()


# In[41]:


exp5=x**2*y+y**3*x+x+3*y


# In[42]:


exp5.diff(y)


# In[43]:


exp5.diff(x)


# In[44]:


exp5.diff(x,y)


# #### Series

# An extremely useful tool when trying to understand functions is the Taylor series around a given point

# In[45]:


exp4.series(x,0,3) # this means: series wrt x, at x=0 to order x^3.


# This becomes much more useful when you start dealing with functions you don't konw a lot about

# In[46]:


sympy.besselj(2,x).series(x,0,3)


# Importantly, series is not really a Taylor series.  In includes inverse powers as well:

# In[47]:


sympy.gamma(x).series(x,0,3)


# To see why this is useful consider

# In[48]:


(sympy.gamma(x)*sympy.sin(x)).subs(x,0)


# In[49]:


(sympy.gamma(x)*sympy.sin(x)).series(x,0,2)


# The function $\Gamma[x]\sin(x)$ is perfectly well behaved at $x=0$ but you need to look at the series as $x\to 0$ to see that.  There is also a limit option but in my experience series is always better than limit.

# #### Integration

# Integration is where it is really obvious how much better a computer is than a person.  Even just a minor change to an integral can make it faily challenging

# In[50]:


sympy.integrate(sympy.sin(x),x)


# In[51]:


sympy.integrate(sympy.sin(x**2),x)


# In[52]:


sympy.integrate(x**2*sympy.exp(-x),(x,0,1))


# The notation for $\infty$ is a bit unusual

# In[53]:


sympy.integrate(sympy.exp(-x**2),(x,0,sympy.oo))


# We can also perform a integration that produces a function of some other variable:

# In[54]:


a=sympy.Symbol('a')


# In[55]:


sympy.integrate(sympy.exp(-a*x**2),(x,0,sympy.oo))


# Notice again that it didn't know what to do with our variable $a$ (it could be real, imaginary, complex, negative, postive. Fortunately, we can tell it what to assume to avoid this problem (sometimes it can't figure it out on its own):

# In[56]:


c=sympy.Symbol('c',positive=True)


# In[57]:


sympy.integrate(sympy.exp(-c*x**2),(x,0,sympy.oo))


# ## Summary

# Symbolic computation is extremely powerful and useful. The functionality is really quite endless.  However, one often does not appreciate all that it can do until you really need it.  At the very least, it will keep you from making stupid mistakes on your homework.
