import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
fig1 = plt.gcf()
plt.show()
fig1.savefig('images/plot1.png', format="png")