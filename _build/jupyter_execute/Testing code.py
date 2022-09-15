#!/usr/bin/env python
# coding: utf-8

# testing an ODE int function

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def ODE_int(D,X0,t0,tf,dt=0.00001):
    tout=np.arange(t0,tf,dt)
    X=np.zeros((len(tout),len(X0)))
    X[0,:]=X0
    for i in range(len(tout)-1):
        X[i+1,:]=D(X[i],tout[i])*dt+X[i,:]
    return X,tout


# In[3]:


def derivative(X,t,gamma=1,omega_0=4): ## X is the array [x,v]
        ## notice that we need to use t as an argument even though we don't use it
    return np.array([X[1],-gamma*X[1]-omega_0**2*X[0]]) ## return the array dot X defined by our equation


# In[4]:


t_R=np.linspace(0,10,100) # range of time we want to solve
X0=[3.,0.1]     #initial conditions


# In[5]:


from scipy import integrate


# In[6]:


sol1=integrate.odeint(derivative,X0,t_R) 


# In[7]:


plt.plot(t_R,sol1[:,0])  # returns solution for X at each point in time  sol[:,0] is x(t)
plt.plot(t_R,sol1[:,1])  # returns solution for X at each point in time  sol[:,1] is v(t)
plt.show()


# In[8]:


mysol,t_s=ODE_int(derivative,X0,0,10,dt=0.001)


# In[9]:


mysol[:,0].shape


# In[10]:


plt.plot(t_s,mysol[:,0])  # returns solution for X at each point in time  sol[:,0] is x(t) # returns solution for X at each point in time  sol[:,1] is v(t)
plt.show()


# In[ ]:




