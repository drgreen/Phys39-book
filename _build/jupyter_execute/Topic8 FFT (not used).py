#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.fft import fft, fftfreq, ifft, fftshift, fftn
import matplotlib.pyplot as plt


# ## Fourier Tranforms

# A very common problem in science is that we want to take a signal that is given in time and decompose it into different frequencies.  This is very common in music where you want to take the sound that you measure as a pressure vs time and convert it into notes.  The same is true with light, where we want to decompose the light wave into the colors that make up a given signal.

# We can break any time series up into frequencies using a procedure known as the fourier transform.  You will learn the theory behind the fourier transform in great detail in Physics 105a.  However, the principles for why it is useful when applied to data are similar to fitting lines.  

# In[ ]:




