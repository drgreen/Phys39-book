#!/usr/bin/env python
# coding: utf-8

# # Topic 1b: Functions and Basic Operations

# ## Basics Operations

# Now that we know the basic data structures, now what can we do with them.  
# 
# There are some basic logic structures that are useful: are things equal

# In[1]:


listtest=[i for i in range(10)]


# In[2]:


listtest


# In[3]:


5==4


# In[4]:


5==5


# In[5]:


[1,2,3]==[3,2,1]


# In[6]:


[1,2,3]==[1,2,3]


# Inequalities are also very useful.  Both >,<, >=, and <= do the obvious thing

# In[7]:


print(5>3, 5>5, 5>=5, 5<=5)


# The value of the equals operation shall be pretty clear when you start running over long lists of numbers looking for something in particular.

# Already, you might also have noticed that I cheated you: True and False are data types we didn't discuss yet!  True and False are the outputs of logical operations, but are also things we can save for later use:

# The simplest possible version is if statements, which mean can be understood as "if True: do something":

# In[8]:


Is_my_hair_too_long=True
if Is_my_hair_too_long:
    print('Get a haircut!')
else:
    print('continue on')


# We also have our basic and / or statements:

# In[9]:


print(1==1 and 2==2)
print(1==1 and 2==1)
print(1==1 or 2==1)
print(1==1 or 2==2) # notice that OR is inclusive True or True is True.


# Truth statements are particularly useful in if statements in combination with loops.  For example, we can run through a list of objects but then do something only under certain conditions

# In[10]:


list1=range(1,10)  # this automatically makes a list of 1,9


# To see how this works, let's just make a for loop and print what is happening:

# In[11]:


for item in list1:
    print(item)


# Okay - so our for loop just picks an elements from the list, one at a time and then lets us do something with it.  Now let's see what happens when we put an if statment inside the loop: 

# In[12]:


for item in list1:
    if item==3:
        print('Yay!!')
    else:
        print('nay')


# In[13]:


for item in list1:
    if item>4:
        print('yay')
    elif item >2:
        print('hmmm')
    else:
        print('nay')


# We also have a bunch of basic math operations.  The least obvious is powers:

# In[14]:


print(2**4, 2**(12), 2**12)  #using backets with ** isn't necessary but highly encouraged.


# In[15]:


print(2**(1/2.))


# There are also some nice tricks built for tasks you will use a lot.  For example if I want to add up something, I can do += instead of x=x+y (which looks confusing)

# In[16]:


costs=[1.2,3.4,4.5,4.,6.]
total=0
for cost in costs:
    total+=cost
print(total)


# We can also do the same with substraction, multiplication or divisions

# In[17]:


cash=100
for cost in costs:
    cash-=cost
print(cash)


# *Aside: Notice that we have a nice example of numerical error here!  It should have been 80.1, but because of the way our floating point number is stored it isn't.*

# In[18]:


19.1-19


# *What is happening here?  A pretty good guess is that your computer wants to store information in binary form, which means that it works in powers of 2.  I.e. $1=2^0$, $2=2^1$, $3 = 2^1 + 2^0$, $4=2^2$.  For decimals, then we might expect it is storing it in terms of fractions $1/2 =2^{-1}$, $1/4 = 2^{-2}$. So we can check that is what is happening if we pick nice fractions of 2 writen as a decimal:*

# In[19]:


19.5-19


# In[20]:


19.25-19


# In[21]:


19.125-19


# *Now we can look up the definition of a floating point number in python and find out that it is a 64 bit number, meaning that it contains 64 binary numbers or can represent from 1 to $2^{63}$*

# In[22]:


2**(64)/10**19


# *We see it is pretty close to our 16 decimal places from before.  To udnerstand why it isn't exactly 16, we should also remember that we need to store the overall power as well (because it can store numbers much bigger or smaller than $2^{64}$, so it must be using some of the 64 bits for that as well.  There is clearly a bit more going on under the hood too*

# In[23]:


2**(1000)


# *End of Aside*

# Now, for multiplication and division, we have to be very careful not to accidently multiply or divide by zero:

# In[24]:


factorial=1
for i in range(5):
    factorial*=i
print(factorial)


# In[25]:


factorial=1
for i in range(5):
    factorial*=i+1
print(factorial)


# For division, we have to worry about division by zero:

# In[26]:


inverse_factorial=1
for i in range(5):
    inverse_factorial/=i
print(inverse_factorial)


# In[31]:


inverse_factorial=1
for i in range(5):
    inverse_factorial/=(i+1)
print(inverse_factorial,1/inverse_factorial)


# Aside: Notice that I have been running through lists of numbers using range. In Python 2, you were told not to do this (but to use something called xrange) to avoid creating a list that is saved to memory for no reason.  Range in python 3 is basically what xrange used to be an they removed the function xrange  

# Range is often useful when you running through things in a list

# In[32]:


short_list=['box','car','keys']
for i in range(3):
    print(short_list[i])


# Now if you just wanted the elements in the list, you can also just call them one by one:

# In[33]:


for item in short_list:
    print(item)


# And, sometimes, you want both: you want the item in the list but you also want to know the element it came frmo.  For that, you have enumerate:

# In[34]:


for i,item in enumerate(short_list):
    print(i, i**2, i+1) # look I can do math with the index from which each entry came
    print(item) # I also have actual entry in the list available to use


# ## Functions

# That we have enough basic tools, we can start defining our own functions for our specific problems.  The simpliest way to proceed is by example.

# The basic setup is as follows: we define it using def and put in some information we want to use in the function

# In[35]:


def say_hi(name):
    print('Hi '+name)


# In[36]:


say_hi('Professor Green')


# As written, the above function looks fine, but will not work if we don't put the exact right kind of input.

# In[37]:


say_hi(5)


# It is usually a good habit to either (a) force the data to be the kind we want or (b) send a warning

# In[40]:


def say_hi_2(name):
    print('Hi '+str(name))
def say_hi_3(name):
    if type(name)==str:
        print('Hi '+name)
    else:
        print('Error: not a string')


# In[41]:


say_hi_2(5)
say_hi_3(5)


# Now let's try another example: we saw above that we can make our own verion of a factorial.  So we can just define a function that does this automatically

# In[42]:


def factorial(n):
    out=1
    if n>0: # We are going to force n>0 so that it is well defined
        for i in range(int(n)): ## notice that I am forcing n to be an integer to avoid problems
            out*=(i+1)

    return out
        


# In[43]:


print(factorial(5), factorial(7), factorial(-6.6))


# We notice a few things in this way I have implemented the code: it allows me to input non-integers but gives a integer answer.  I also wrote it in this slightly awkward way of multiplying everything in a loop.

# Now here is where we get to some of the fun of coding!  Just like writing, there can be more than one way to write something that accomplishes the same basic goals, but one might be far better or more elegant than the other.

# We can think about factorial by a different definition.  In fact, it is the same defining feature of the Gamma function $$\Gamma[x] = (x-1) \Gamma[x-1]$$ and $$\Gamma[1]=1$$ so that $\Gamma(n) = (n-1)!$.

# In[44]:


def Gamma(n):
  
  if n<=1:
      #If n=0, or any value it can't determine, set to 1
    return 1
  else:
      #If it is greater than one, call again iteratively by definition
    return (n-1)*Gamma(n-1)


# In[45]:


print(Gamma(5),Gamma(3.2))


# Note that this isn't actually the true gamma function because we defined $\Gamma(x<1) =1$ rather than just $\Gamma(1)=1$.

# Regardless, this is a clever implementation, because it defines the factorial recursively, rather than by brute force.

# As our last example, we are going to make a sorting function.  We want to take a list of numbers are reorder them from largest to smallest.  

# In[46]:


def dumb_sort(alist):
    b=[] # I am making a place to put my numbers in order
    
    c=alist[:] # I am making a copy of the list so that I don't ruin my original list
    
    while len(c)>0: # I am going to loop over c and remove the largest each time.  I stop when there is nothing left
        
        largest=c[0] # assume the first element is the largest
        
        for item in c: #now run over all the elements in the remaining list 
            if item>largest: # and check if any are bigger
                largest=item # when you get somethign bigger, store it the biggest and continue
        
        b.append(largest) # after running through all the elements, add the largest to the output list
        
        c.pop(c.index(largest)) #remove the largest from c and repeat
    
    return b #when you have all the numbers in b, c=[] and the while loop will end.  Output b


# In[47]:


unordered=[3,5,1,5,2,4,5,2,1]


# In[48]:


dumb_sort(unordered)


# I called the "dumb_sort" for a reason: it is really inefficient.  I had to make multiple copies of my list and looped over the elements a lot of times.  If I have a list of lenght N, I would take at least 3N space in memory and would run look at roughy N^2/2 elements in the list (I keep looking at them over and over again to compare them to the new biggest number).

# In[49]:


def mysort(a):
    for n in range(1,len(a)):
    #Read in one element at a time, starting with second (first is sorted)
        value=a[n]
    #set marker for previous
        i=n-1
        while i>=0 and (value > a[i]):
        #if i is not past the first element, but value > previous element swith the two
            a[i+1]=a[i]
            a[i] = value
      #now move the marker one to the left and repeat
            i-=1
#1st element is sorted.  When we get to the nth element, the n-1 previous elements are sorted.  Just need to place it in the right spot

b=[1,5,7,8,1,9,2]
mysort(b)
print(b)


# Notice that if we give it an ordered list, it only has to check N-numbers instead of N^2/2.  I.e. once we sort part of the list, we don't keep checking it. 

# A very useful thing about functions is that you can given them options that come with defaults.  To see this in action, suppose we want to know the position of a particle subject to an constant external force:

# In[50]:


def position(t,x0,v,a):
    pos=x0+v*t+a*t**2/2. # solution for motion under constant acceleration
    return pos # return position


# In[51]:


position(10,0,0,9.8)


# But maybe there are some standard choices we like, e.g. a=9.8 m/s$^2$ or x0=0.  If that is a case we consider a lot, maybe we just want to assume those values, unless otherwise stated:

# In[52]:


def pos_short(t,x0=0,v=0,a=9.8):
    pos=x0+v*t+a*t**2/2. # solution for motion under constant acceleration
    return pos


# In[53]:


pos_short(10)


# The good part about doing it this way is that I can always put back the values if I want

# In[54]:


pos_short(10,x0=15)


# ## Python Scripts

# Now that you have defined a function and have it working properly might want to do two things: (a) move it somewhere so that you don't accidently mess it up (b) have it available for other projects you might be working on.

# In[55]:


import sort


# In[56]:


b2=[1,5,7,8,1,9,2]
sort.mysort(b2)
print(b2)


# Notice that I can save all kinds of information this way.  E.g. I can just save lists of numbers or specific numbers

# In[57]:


print(sort.my_fav_number)


# This is the same structure we use for all kinds of things. We can make our lift easier by importing a file under a shorted name.  E.g.

# In[58]:


import numpy as np
import sort as s


# In[59]:


np.cos(1.)


# In[60]:


b3=[1,5,7,8,1,9,2]
s.mysort(b3)
print(b3)


# We have more than one option for how to important information.  We could just import as single function, in which case you can just use the name:

# In[61]:


from numpy import cos


# In[62]:


cos(1.)


# If you want to import all of the functions this way, you can use *, but PLEASE don't do this for things like numpy that have a LOT of functions

# In[63]:


from sort import *


# In[64]:


my_fav_number


# ### Summary

# We have run through a lot of the basic functionality of python.  At this level, you have all the functionality you need to do anything.  You have all the basic logical and mathematical operations at your disposal and all the objects you need to store the input and output.
# 
# Now, in practice, converting these basic operations into more advanced algorithms is a lot of work.  You have the power to do it, but it would run very slow and take a lot of your time.  Luckily, more skilled users of python have written a lot of that code using more advanced and elegant techniques.  They probably do the same basic thing that you might, but it will run way faster and have a lot more versality (and is already debugged).  From here, we are going to start learning about some of these software packages and how to understand what they are doing and why they are useful.
# 

# In[ ]:




