my_fav_number=3.141592653589793

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


