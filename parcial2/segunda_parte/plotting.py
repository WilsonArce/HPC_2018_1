import matplotlib.pyplot as plt
file = open("ansTime.txt","r")
ans64 = [0, 0.000000, 0.000000, 0.000000]
ans128 = ans64
ans256 = ans64
ans512 = ans64
ans1024 =ans64
ansArrays = [ans64, ans128, ans256, ans512, ans1024]

for i in ansArrays:
  for j in range(10):
    print j
    array = file.readline()
    print array
    i[1] += float(array[1])
    i[2] += float(array[2])
    i[3] += float(array[3])
  i[0] = array[0]
  i[1] = i[1]/10
  i[2] = i[2]/10
  i[3] = i[3]/10
  print i

fig1 = plt.gcf()
exeTime = [1,2,3,4,5]
matSize = [64,128,256,512,1024]
plt.axis([0.000000, 70, 64, 1024])
plt.title('Secuential algorithm')
plt.ylabel('Time (seconds)')
plt.xlabel('Matrix size (n x n)')
plt.plot(exeTime, matSize, color='blue')
plt.show()
fig1.savefig('images/plot1.png', format="png")