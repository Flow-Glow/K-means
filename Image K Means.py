import numpy as np
import matplotlib.pyplot as plt
import K_Means_Algorithm as kma

x = plt.imread("imgs/w2.png") * 255

data = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
centroid = kma.kma(data=data, k=10, D=3, N=len(data), t=.1, L=1)
m = kma.membership(data, centroid)
Img_list=[]
for i,a in enumerate(data):
    Img_list.append(kma.pixel_reader(a, centroid))
    print(i)
Img_list = np.array(Img_list)
New_img = Img_list.reshape(x.shape[0], x.shape[1], x.shape[2])

plt.imsave("imgs/Testw3.png",New_img/255)
plt.show()
