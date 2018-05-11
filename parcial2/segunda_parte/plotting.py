import matplotlib.pyplot as plt
file = open("ansTime.txt","r")
ans64 = [0, 0.0, 0.0, 0.0]
ans128 = [0, 0.0, 0.0, 0.0]
ans256 = [0, 0.0, 0.0, 0.0]
ans512 = [0, 0.0, 0.0, 0.0]
ans1024 = [0, 0.0, 0.0, 0.0]
ansArrays = [ans64, ans128, ans256, ans512, ans1024]

for i in ansArrays:
  for j in range(10):
    array = file.readline().split(',')
    i[1] += float(array[1])
    i[2] += float(array[2])
    i[3] += float(array[3])
    i[3] += float("{0:.2f}".format(float(array[3])))
  i[0] = int(array[0])
  i[1] = float("{0:.6f}".format(i[1]/10))
  i[2] = float("{0:.6f}".format(i[2]/10))
  i[3] = float("{0:.6f}".format(i[3]/10))
  #print i
#print ans1024
matSize = [64,128,256,512,1024]

fig1 = plt.gcf()
secExeTime = [ans64[1], ans128[1], ans256[1], ans512[1], ans1024[1]]
#print secExeTime
plt.axis([64, 1024, 0.000, 5.000])
plt.grid(True)
plt.title('Secuential algorithm')
plt.ylabel('Time (seconds)')
plt.xlabel('Matrix size (n x n)')
plt.plot(matSize, secExeTime)
plt.show()
fig1.savefig('images/secuential.png', format="png")

fig2 = plt.gcf()
globalExeTime = [ans64[2], ans128[2], ans256[2], ans512[2], ans1024[2]]
#print globalExeTime
plt.axis([64, 1024, 0.000, 0.020])
plt.grid(True)
plt.title('Global memory algorithm')
plt.ylabel('Time (seconds)')
plt.xlabel('Matrix size (n x n)')
plt.plot(matSize, secExeTime, matSize, globalExeTime)
plt.legend(('Secuential','Global-mem'),
           loc='upper right')
plt.show()
fig2.savefig('images/global.png', format="png")

fig3 = plt.gcf()
sharedExeTime = [ans64[3], ans128[3], ans256[3], ans512[3], ans1024[3]]
#print sharedExeTime
plt.axis([64, 1024, 0.000, 0.009])
plt.grid(True)
plt.title('Shared memory algorithm')
plt.ylabel('Time (seconds)')
plt.xlabel('Matrix size (n x n)')
plt.plot(matSize, secExeTime, matSize, sharedExeTime)
plt.legend(('Secuential','Shared-mem'),
           loc='upper right')
plt.show()
fig3.savefig('images/shared.png', format="png")

speedUpGlobal = [ans64[1]/ans64[2], ans128[1]/ans128[2], ans256[1]/ans256[2], ans512[1]/ans512[2], ans1024[1]/ans1024[2]]
speedUpShared = [ans64[1]/ans64[3], ans128[1]/ans128[3], ans256[1]/ans256[3], ans512[1]/ans512[3], ans1024[1]/ans1024[3]]
#print speedUpGlobal
#print speedUpShared
fig4 = plt.gcf()
plt.axis([64, 1024, 0, 1300])
plt.grid(True)
plt.title('Speed up comparison')
plt.ylabel('Speed up')
plt.xlabel('Matrix size (n x n)')
plt.plot(matSize, speedUpGlobal, matSize, speedUpShared)
plt.legend(('Global-mem','Shared-mem'),
           loc='upper right')
plt.show()
fig4.savefig('images/speed_up.png', format="png")