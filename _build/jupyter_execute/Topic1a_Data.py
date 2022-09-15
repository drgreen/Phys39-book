#!/usr/bin/env python
# coding: utf-8

# # Topic 1a: Basics of Python

# This is a jupyter notebook.  It is a coding enviroment that is useful for developing, testing code, doing short caluclations, making figures etc.  We will start doing everything here so that we can see what we are doing as we go along.  Later in the course, we will talk about how to write you code into python scripts that live outside of jupyter.

# A juypter notebook works the same as any python code but you can run small pieces of it at a time.  There is also the markdown option so that you can write explainations in latex (like this one).  You can also comment your code directly in line,

# In[1]:


print("hi") # this is a comment inside of a piece of code.  See, python ignores everything after the 


# The notebook format is useful because you run just bits of code and you can keep trying it until it works.  Unlike a script, it will also output some of the results, even if you don't ask it to print them.  E.g.

# In[2]:


5+5


# We can run what is in each box by pressing "shift-enter" (at same time).  You can also go to "Cell" a run that cell, a few cells or the entire notebook.

# There is one big danger of notebooks that doesn't happen in a script: the computer is running it in the order you choose, not the order on the page.  This means you might accidently redefine a variable without noticing.  It also means that you code could be working fine and then when you try to rerun it, it doesn't work anymore.  E.g.

# In[3]:


a=5
b=6
c=7


# In[4]:


a+b+c # The number to the left of this box tells you what order I ran it in 


# In[5]:


c=0


# The way we can avoid this problem is if (a) you are careful to pick good names and not repeat them (b) document your code so that you know what you are doing and why at each step.  Alternatively, if you are just using a variable temporarily, you can delete the variable name after you use it

# In[6]:


del a
del b
del c


# In[7]:


a


# ## Topic 1a: Basic Objects

# ### Data types

# The most basic data types we can use throughout python are the integers (int), floating point numbers (float), strings (str), and complex 

# Here is an some integers (int):

# In[8]:


a = 2
b = 3


# Here is an some floating point numbers (float), think of these as number represented as a decimal:

# In[9]:


c = 2.
d = 3.
e = 3.14


# Here are some strings (str), basically anything we want to treate like words:

# In[10]:


f='2'
g='hi'
h='hello'


# What makes these structures different is how basic operations in python treat them.  We can start with basic math operations.  Let's start with integer operations.

# In[11]:


print(a/b)
print(a+b)
print(a-b)
print(a*b)
print(a/(a+b))


# In[12]:


print(c/d)
print(c+d)
print(c-d)
print(c*d)
print(c/(c+d))


# At this point, there isn't much difference between a float and integer.   However, other objects in python care about the type of data that you feed them and won't accept a float in place of an int. E.g. here we have a list of objects (we'll explain lists in a second)

# In[15]:


list=['this','is','a','list']  # this is a list


# Let's see what happens when we ask for an element of this list using a int or a float.  Let's use our old variables

# In[16]:


print(b,d) # b and d are both 3 but one is an int and the other is a float


# Now let's ask to see an entry (the one labeled by the number three) in our list

# In[17]:


list[b]


# In[18]:


list[d]


# We see why it was important that b is an integer: the list new it was an integer so it knows what entry 3 means, not 3.0.  (You might also have noticed something a bit odd about the fact it returned 'list', don't worry, we'll talk about that in a second)
# 
# We can solve this issue by forcing d to be an integer

# In[19]:


list[int(d)]


# In[20]:


print(int(d),int(3.12),int(3.91))


# This maybe isn't quite what you expected, it just strips off all the decimals, but it does returna an integer.

# Side note: In Python 2.7, a/b would have been considered integer division.  Integer division is confusing because it forces the result to be an integer by striping off the decimal points.  I.e. it doesn't round up, it always rounds down.  In python 3, integer division has its own syntax

# In[21]:


999//1000


# Fortunately, in Python 3 you won't do this by accident.  With that said, the idea that you want to have integer division for integers is not a bad idea.  We will see very soon that there are situations where it only makes sense to use an integer and so it is essential that it knows 4/2 is an integer and not a float (for example). 

# We can also force on type to become another type 

# In[22]:


print(a+float(b))
print(a+float(f))  # remember f is a string that happens to be a number
print(a+int(f))


# Of course it can't work miracles

# In[23]:


float(g) # g='hi' can't become a float


# Now, when you mix floats and integers you always get float out:

# In[23]:


print(a/d)
print(a+d)
print(a-d)
print(a*d)
print(a/(c+b))


# I have taken it as self evident what a string (word) and an integer are.  This is probably a good time to pause and ask, what is a floating point number?  From the looks of it, it is some kind of number with decimals.  Of course, in math a decimal can go on forever.  Our computer, as a default, does not give unlimited space to store numbers.  
# 
# Let me prove to you that for a float, the computer is only storing some of the digits:

# In[32]:


x=12.+0.0001


# In[33]:


(x-12.)*100000000000000000000000000000


# But a floating point number can remember small numbers 

# In[34]:


0.000000000000000000000000000000001


# In[35]:


0.000000000000000000000000000000001*100000000000000000000000000000000000000


# Clearly the problem is not the size of the number but how many decimal places.  We can figure this out for ourselves

# In[40]:


x=1.+0.000000000000001
(x-1.)*100000000000000000000


# In[41]:


x=1.+0.0000000000000001
(x-1.)*1000000000000000000000000


# By my count, this fails when I move it to the 16th decimal place.  Since the leading number is 1 that means a float is carrying 16 digits, and when I move the number to the 17th digit, it effectively vanishes.  This correct: a float is bascially a 16 digit number in scientific notation (i.e. we store 16 digits times a power of 10).

# In[35]:


x=.0000001+0.00000000000000000000001
(x-0.0000001)*100000000000000000000000000000000000000


# In[36]:


x=.0000001+0.000000000000000000000001
(x-0.0000001)*100000000000000000000000000000000000000


# Finally, a string is like a word, math operations don't work on it except addition, which just joins the letters

# In[42]:


f+g+h


# In[44]:


str(a)+g


# In[47]:


for i in range(10):
    print('Hi number '+str(i)+'.txt')


# In[48]:


a/f


# Lastly, there are complex numbers, which act like two floats for the real and imaginary parts.  It uses the engineering notation of j instead of i:

# In[53]:


z1=1+2j


# In[54]:


z1/3


# It this bothers you, you can also define them by

# In[55]:


z2=complex(1,2)


# In[56]:


z1/z2


# In[58]:


z1*z1


# ## Lists, Tuples and Dictionaries

# In a large number of situations, we will want more than just a single number, word, etc but collections of pieces of information (data) organized in various ways.  The basic structures in python are lists, tuples and dictionaries.

# ### Lists

# A list of is an ordered group of objects.  They can really be anything.  

# In[40]:


list1=[1,2,3,'hi',4,5,6,7]


# That being said, it is generally a good idea to make your lists out of a single common data type.

# Now we reach the point where we have to talk about how python organizes lists, which is somethign that drives may people crazy.  You can isolate a signel element using square brackes:

# In[42]:


print(list1[0],list1[3])


# Notice that the entry is counted starting at 0.  I.e. the nth entry is listed by list1[n-1].  This takes some getting used to.  You can also run through the elements of a list as follows

# In[43]:


for item in list1:
    print(item) # notice that the stuff inside the loop is indented with a tab


# This is our first encounter with the "for loop" and the indented structure.  Python understands order of operations by the formating.  Notice that there is a tab/indent for the thing we want to do as it runs through elements of the list.  We can nest loops inside of each other too

# In[44]:


list2=[1,'b',3]
for item1 in list2:
    for item2 in list2:  # notice the tap
        print('item 1 and 2: '+str(item1)+' '+str(item2))
    print('item 1: '+str(item1)) 
print('all done')


# You can pick out some of the elements in a list a few different ways.  You can pick a range you want

# In[47]:


print(list1[2:5],list1[5]) # notice that is starts at list1[2] but ends without including list[5]


# We notice also that lis1[2:5] returns the output as a new list.  I.e. only when we isolate a single element does it return it as something other than a list.

# You can also go from the beginning to a given number or a given point to then end

# In[48]:


print(list1[:5])
print(list1[3:])


# We can also pick every nth elements using :: 

# In[98]:


print(list1[::2])
print(list1[::3])
print(list1[2::3])


# You can also count from the end using negative numbers

# In[49]:


print(list1[-1])
print(list1[-2])
print(list1[2:-2])


# This also gives us a funny way to print a list backwards

# In[50]:


print(list1[::-1])
print(list1[::-2])


# We will get to know lists and their cousins very well in this course, so this is just the starting point.  One key aspect of lists is that they can be changed:

# In[51]:


list1[0]='k'
print(list1)


# Now comes the scary part of the fact that lists can change: if you make a list equal to another list, you are just telling the computer to point to the first list.  This is best shown with an example:

# In[52]:


list3=list1 # make list3=list1
print(list3) # see they are equal


# In[65]:


list1[0]=1 #now change list1
print(list3) # see list3 changed too!


# If you want to save a record of list1 before you change it, you need to make a copy.  One way to do this is to tell it you want the list to be equal to the elements of the list.  This also shows us that if we take only a subset of elements in the list, we also won't have this pointing issue.  If you are using the full list, you can also use copy:

# In[66]:


list4=list1[:]
list5=list1.copy()
print(list4,list5)


# In[67]:


list1[0]='hello'
print(list1,list4,list5)


# You can combine lists using addition, so that

# In[68]:


list4+[8,9,10]


# This does not change list4, it just output the combined list.  If you want to keep this list you need to give it a new name.

# You might have a number or string you want to add to a list without making a new list, in which case append is more useful 

# In[70]:


list4.append(11)
print(list4)


# We can also remove entries in the list as follows:

# In[71]:


print(list1)
del list1[3]
print(list1)


# There are lots of other useful tools we have.  For example, you might like to know the length of a list, it largest or smallest values, etc.

# In[72]:


print(len(list4),min([2,5,1,11]),max([2,5,1,11]))


# The elements of a list can be lists themselves.  When we start using some of pythons mode powerful computational tools, this structure will we be very important.

# In[88]:


listarray=[[1,2],[3,4]]


# In[89]:


listarray[1]


# In[91]:


listarray[1][1]


# Might like this to work like a matrix, but we will have to way for numpy array for that.  I.e. lists don't treat columns and rows in the same way

# In[101]:


print(listarray[0][:],listarray[:][0])


# ### Tuple

# Now we come to the tuple.  This is very much like a list but we used () instead of []

# In[102]:


tuple1=(1,2,3,'hi')


# In[103]:


print(tuple1[0])
print(tuple1[::2])


# But the key difference is that you can't change the individual entries

# In[104]:


tuple1[0]=5


# The tuple is useful for the reason I mentioned above with lists - it is scary that you might accidently change the values in your list in the middle of your code.  The tuple has some inherent value just from the fact that you can't accidently erase your data. 

# Finally, we come to dictionaries.  These are interesting objects because want to organize information by name.  There are a few ways to define a dictionary:

# In[105]:


dict1={'bob': 12,'alice': 2}


# In[106]:


dict2={}
dict2['bob']=[1,2,3]
dict2['alice']=[4,5,6]


# In[107]:


dict2


# The list of names are called the keys, and we can recover them by 

# In[108]:


dict2.keys()


# This is a very helpful tool when you get to large data sets.  Some programs will output data in the form of a dictionary with a huge numbers of keys.  Being able to make a list of just the keys quickly and easily is a suprisingly useful tool.

# Every entry in a dictionary can be a unique type of data.  It is just a way of grouping things together:

# In[109]:


dict3={'bob': [1,2,3], 'alice':'my name is alice','number':41}


# In[110]:


for key in dict3.keys():
    print(dict3[key])


# ### Summary

# We introduced the basic objects that python employs to read in and output information.  We also saw the most basic mathematical operations and how they work on these different objects.  The next step is to fill out the set of operators so that we can start doing more exciting stuff.   
