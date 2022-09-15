#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from scipy import integrate


# In[3]:


t_R=np.linspace(0,20,2000)


# In[4]:


def derivative(X,t,w0=4,w1=2,g=0): ## X is the array [x,v]
        ## notice that we need to use t as an argument even though we don't use it
    return [X[1],-g*X[1]-w0**2*X[0]-w1**2*(X[0]-X[2]),X[3],-g*X[3]-w0**2*X[2]-w1**2*(X[2]-X[0])]
X0=np.array([5,0,0,0])


# In[5]:


sol1=integrate.odeint(derivative,X0,t_R)


# In[6]:


plt.plot(t_R,sol1[:,0])  # returns solution for X at each point in time  sol[:,0] is x(t)
plt.plot(t_R,sol1[:,2])  # returns solution for X at each point in time  sol[:,1] is v(t)
plt.show()


# In[7]:


plt.plot(t_R,sol1[:,0]+sol1[:,2])  # returns solution for X at each point in time  sol[:,0] is x(t)
plt.plot(t_R,sol1[:,2]-sol1[:,0])  # returns solution for X at each point in time  sol[:,1] is v(t)
plt.show()


# In[8]:


def derivativeA(X,t,w0=5,g=1): ## X is the array [x,v]
        ## notice that we need to use t as an argument even though we don't use it
    return [X[1],-g*X[1]-w0**2*X[0]]
XA=np.array([0,5])


# In[9]:


solA=integrate.odeint(derivativeA,XA,t_R)


# In[10]:


plt.plot(t_R,solA[:,0])


# In[11]:


def derivativeB(X,t,w0=5,g=1): ## X is the array [x,v]
        ## notice that we need to use t as an argument even though we don't use it
    return [X[1],-g*X[1]-w0**2*X[0]+np.exp(-(t-4)**2/0.01)]
XB=np.array([0,0])


# In[12]:


solB=integrate.odeint(derivativeB,XB,t_R)


# In[13]:


plt.plot(t_R,solB[:,0])


# In[ ]:




