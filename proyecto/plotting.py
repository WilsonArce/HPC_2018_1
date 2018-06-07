import matplotlib.pyplot as plt

matSize = [512*512,1200*800,1600*748,7680*5022,8000*4500]

cpuTime = [0.0261931, 0.1070182, 0.1631308, 3.9483769, 3.1172331]
gputime = [0.0228917, 0.0859753, 0.1428814, 3.3815264, 3.2769439]
#print cpuTime
#print gputime
fig4 = plt.gcf()
plt.axis([25000, 36000000, 0, 3.5])
plt.grid(True)
plt.title('Throwput')
plt.ylabel('Time (seconds)')
plt.xlabel('Image size (millions of pixels)')
plt.plot(matSize, cpuTime, matSize, gputime)
plt.legend(('CPU time','GPU time'),
           loc='upper right')
plt.show()
fig4.savefig('images/throwput.png', format="png")
