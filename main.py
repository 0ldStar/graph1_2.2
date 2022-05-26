import matplotlib.pyplot as plt
import numpy as np

cpu = np.transpose(np.loadtxt('../graph1_1.2/cmake-build-debug/write.txt'))
gpu = np.transpose(np.loadtxt('cmake-build-debug/write.txt'))
xc = np.arange(1, len(cpu[0]) + 1)
xg = np.arange(1, len(gpu[0]) + 1)
# plt.plot(xc, cpu[0], label='cpu')
plt.plot(xg, gpu[0], label='gpu')
plt.ylabel('time, msec')
plt.xlabel('matrix size')
# plt.ylim(top=1000)
# plt.xlim(right=100)
plt.grid()
plt.legend()
# plt.savefig('img.jpg')
plt.show()