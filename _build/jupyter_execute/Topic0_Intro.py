#!/usr/bin/env python
# coding: utf-8

# # Topic 0: Introduction

# ## What is this course about?

# Computer programing / coding is a part of life for almost everyone working a STEM or STEM-adjacent field.  Of course, this is true in science for calculating things, analyzing and visualizing data (things you will likely need for future courses).  But this is increasingly true in non-academic jobs of all kinds.  The world is overflowing with data to the point that just collecting information and organizing it is a task for a computer.  Familiarity with the basic tools and language around programming will get you a long way.

# In many ways, the role of computers in STEM is similiar to the importance of reading and writing more generally.  You don't have to be a professional author for it be useful to be able to write a clear and succinct email (for example).  A great writer may also be able to craft an beautiful essay that if beyond your capability, but you can still understand what it means and write a something that means the same thing, even if it doesn't have the same elegance.  We find ourselves in the exact same situation when it comes to writing code: an expert will be able to manipulate the (programing) language in ways that you can't envision, but that doesn't mean that you can't understand what they are doing or write a piece of code that does what you need.  

# I am not here to tell you about strict rules about programming a computer anymore than I am to lecture you about the correct grammatical style for writing a news story for the New York Times. Our goal is much more modest to get you started: how do we understand what the computer is doing and how do we talk to it?  We will only scratch the surface of what you can do with python (not to mention all the other programming languages), but the hope if that with this introduction, you will be able to learn more about what else is out there, and apply it, on your own.  

# Computers are incredibly powerful tools, but only if we can get it to do what we want.  Just like sending a clear set of instructions in writing to someone else (e.g. like a recipe), we have to make sure we write in a way that doesn't confuse our target. The reality of computer programming is that anything you will do will be built on enormous amounts of code written by someone else.  As a result, the first step to doing anything usually requires that we understand how someone else built the language / function / package so that we can use it ourselves.  Unfortunately, people can make fairly illogical decisions and so there is sometimes no rhyme or reason to why the syntax of a particular language is written in a particularly way.

# Fortunately, unlike people, computers are perfectly logical.  While the person who wrote a particular piece of code might have done something for no particularly good reason, what the computer is doing underneath is often more rigid.  If the computer is trying to solve an equation, regardless of the syntax of the code, the computer is probably implementing one of a handful of basic algorithms.  Even though we might not know exactly what the computer is doing in detail, we can learn a lot about what our code is doing just by understanding a bit about the basics of how a computer operates.

# ## What are my goals for you?

# I am not a computational physicist or someone that writes professional quality code for a living.  I am not going to force any style of coding upon you as if you were about to start a job as a professional software developer.  My code, like yours, will be filled will all kinds of mistakes that a professional would not tolerate.  Our goal is just to write code that works for now.  We are going to learn to code by doing.  Once you have a feel for it, you can go back and improve your technique.  This strategy is similar to learning to write stories in parallel with learning grammar/spelling.

# A bit of programing knowledge will open a lot of doors for you.  The key skill we are trying to develop as physicist is how to think and process information.  However, at this stage of your career, many of your best opportunities will come from combining your own thinking skills with some basic computer skills.  My goal is to teach you enough about the tools available to your (with python) that you can figure out the rest when you need it.

# (think of this like chess: you far from being grand masters right now, but if you know how to run any good computer chess program, you can still compete against the very best human players)

# E.g. you have probably spent a lot of time learning how to solve integrals by hand.  However, a computer can do a lot of that work for you:

# In[1]:


import sympy


# In[2]:


sympy.init_printing()
x=sympy.Symbol("x")
sympy.integrate(x**3*sympy.sin(x),x)


# In[3]:


sympy.integrate(x**3*sympy.sin(x),(x,0,1))


# In[4]:


sympy.integrate(x**3*sympy.sin(x),(x,0,1)).evalf()


# We can also evaluate the integrals numerically:

# In[5]:


from scipy.integrate import quad
import numpy as np


# In[6]:


quad(lambda y:y**3*np.sin(y),0,1)


# Of course, knowing how to integrate is still very important; it is pretty hard to debug a computer if you don't understand what it is trying to do.  Instead, the value of the computer is that it can dramatically enhance and/or speed you ability to solve problems by giving the computer the more difficult or time consuming tasks.

# With that in mind there are specific things I want for you from this course: 
# 
# (1) you will all be able to make a piece of python code do some basic things for you: read/sort/analyze data, calculate things you care about (integrals, solve algebraic or differential equations), make figures, etc.  
# 
# (2) you will be sufficiently comfortable with python that when a piece of code is not working, you can debug the problem and get it to do what you want.  Debugging problems is really the key step that you will need to succeed in the world.  No one ever writes code that works the first time, so identifying and fixing your own mistakes is essential to succeeding on your own.  Every mistake is unique, so it isn't about memorizing a list of possible errors and checking them, but it is building the confidence that you can figure out the problem yourself.

# ## Why Python?

# #### Here are some reasons we are using python: 
# (1) it is easy to read and write (2) it's free / open source (3) it is used by LOTS of people in both industry and academia and (4) it is versatile.

# Confession: I learned C++ when I was in your position.  It was enourmously helpful in getting research experience as an undergrad.  Yet, I really didn't like dealing with C++ and I promptly forgot how to use it the second my career didn't require it.  I avoided using computers in my research for about 10 years starting in grad school. Why?  Because C++ is confusing to read, confusing to use and it was a lot of work to write a working piece of code even for a very small problem.  There was really no point to using C++ unless your life depended on it.

# What brought me back was Python.  The way the code works is pretty simple and easy to get started.  It makes it easy to find your mistakes and also worth your time to write something in python even if it is just a little calculation, figure, etc.  The jupyter notebook enviroment makes this even easier because you can just run the little bits of code you need without all the formal parts that go into a proper python script.

# Here is an example.  Let's say I just want to add up some numbers:

# In[7]:


1+2+4+10+3


# Let's say I want to do a really big sum: $\sum_{n=1}^{1000} 1/n^2$

# In[8]:


sum=0
for i in range(1000):
    sum+=1/(i+1)**2
print(sum)


# This was really easy (a few seconds of typing).  In the old days, you would have to do a lot of work to do something like this in C++.

# ##### What is wrong with python?
# It is slow.  If you need to do a very hard and time consuming calculating, python will probably be too slow.  Relatedly, if you are using some big code someone else developed to solve a hard problem, it probably isn't written in python.

# This weakness of python is really a very tiny problem: the most time consuming part of solving most problems is writing and testing the code.  Python makes this part (the human part) way easier.  If you reach the point that your ability to solve interesting problems is limited by python itself and not your coding and debugging skills, then I am also confident that you will have no problem learning the tools you need to speed it up using another language (there are also tools to run pieces of code written in other languages inside python so that only the part of your code that needs to be blazing fast is written in something other than python).  

# In[ ]:




