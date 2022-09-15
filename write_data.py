import numpy as np

f=open("data1.txt","w")
f.write('# here is some preamble\n')
f.write('# here is some more preamble\n')
t=np.linspace(1,10,100)
for i in range(100):
    f.write(str(t[i])+',')
    for j in range(3):
        f.write(str(np.random.randn()+0.07*t[i]**j)+',')
    f.write('\n')
f.close()
