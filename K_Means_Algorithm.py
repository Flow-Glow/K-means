import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from datetime import datetime

# 10/13/21 K Means Algorithm By: Wolf
rng = np.random.default_rng()


def K_Means(data, centroids, old_centroids, k, t, D):
    while Check_Static_Condition(centroids, old_centroids, t, k):
        mem = membership(data, centroids)

        old_centroids = centroids

        centroids = mean_compute(mem, centroids, k)
    inertia = CentroidRater(mem, centroids, k)
    inertia = float("{:.2f}".format(inertia))
    scatter_plotter(centroids, k, mem, D, inertia)
    return centroids, inertia


def mean_compute(mem, centk, k):
    for i in range(k):
        mem[i] = np.array(mem[i])
        centk[i] = np.mean(mem[i], axis=0)
    return centk


def euclidean_distance_calc(a, b):
    distance = 0
    for i in range(len(a)):
        bma = (a[i] - b[i])
        bmas = np.square(bma)
        distance = np.sum(bmas) + distance
    return np.sqrt(distance)


def membership(dataset, centroids):
    m = {i: [] for i in range(len(centroids))}
    l = np.empty(len(centroids))

    for i in dataset:
        for lc, c in enumerate(centroids):
            old_distance = euclidean_distance_calc(i, c)
            l[lc] = old_distance
        m[np.argmin(l)].append(i)

    return m


def pixel_reader(data_p,centroids):
    l = np.empty(len(centroids))
    for lc, c in enumerate(centroids):
        old_distance = euclidean_distance_calc(data_p, c)
        l[lc] = old_distance
    return centroids[np.argmin(l)]


def Check_Static_Condition(centroids, old_centroids, threshold, k):
    euclidean_distance = euclidean_distance_calc(centroids, old_centroids)
    booldist = euclidean_distance > threshold
    BoolList = [booldist for _ in range(k)]

    return all(BoolList)


def hex_generator(n):
    clr = []
    for _ in range(n):
        random_number = np.random.randint(0, 16777215)
        hex_number = format(random_number, 'x')
        if len(hex_number) == 6:
            hex_number = '#' + hex_number
            clr.append(hex_number)
        else:
            random_number = np.random.randint(0, 16777215)
            hex_number = format(random_number, 'x')
    return list(set(clr))


def scatter_plotter(centroids, k, mem, d, inertia):
    fig = plt.figure()

    if d == 3:
        ax = fig.add_subplot(projection="3d", azim=-144, elev=9)

    clr = hex_generator(k)
    while len(clr) != k:
        clr = hex_generator(k)

    for i in mem.keys():
        plt.title(f"K Means inertia:{inertia}")
        if d == 2:
            plt.xlabel("X axis")
            plt.ylabel("Y axis")
            plt.scatter(mem[i][:, 0], mem[i][:, 1], c=clr[i], alpha=0.5, s=10)
            plt.scatter(centroids[i][0], centroids[i][1], c="Black", marker="X", s=(30 * .5) ** 2)  # c=clr[i]
        elif d == 3:
            ax.set_ylabel("Y axis")
            ax.set_xlabel("X axis")
            ax.set_zlabel("Z axis")
            ax.scatter(mem[i][:, 0], mem[i][:, 1], mem[i][:, 2], c=clr[i], alpha=0.5, s=5)
            ax.scatter(centroids[i][0], centroids[i][1], centroids[i][2], c="Black", marker="X",
                       s=(30 * .5) ** 2)  # c=clr[i]
            plt.savefig("kmeans3D.png", dpi=300)

    if d == 2:
        plt.savefig("K_Means_Figs\kmeans2D.png", dpi=600)
    else:
        plt.savefig("K_Means_Figs\kmeans3D.png", dpi=300)


def CentroidRater(mem, centroids, k):
    l = []
    for i in range(k):
        for j in mem[i]:
            euclidean_distance = euclidean_distance_calc(j, centroids[i])
            l.append(euclidean_distance)

    return sum(l)


def BestCentroids(data, k):
    c = []
    cr = []
    for i in range(10):
        centroids = data[rng.choice(len(data), k, replace=False)]
        c.append(centroids)
        mem = membership(data, centroids)
        rate = CentroidRater(mem, centroids, k)
        cr.append(rate)
        index = np.argmin(cr)
    return c[index], cr[index]


def kma(data, k, D, N, t, L):
    start = datetime.now()
    centroids_L = []
    rates_L = []

    centroids, rates = BestCentroids(data, k)
    old_centroids = data[rng.choice(len(data), k, replace=False)]

    while old_centroids is centroids:
        old_centroids = (data[rng.choice(len(data), k, replace=False)])

    for i in range(L):
        c, iner = K_Means(data, centroids, old_centroids, k, t, D)
        centroids_L.append(c)
        rates_L.append(iner)
    centroids_L = np.array(centroids_L)
    rates_L = np.array(rates_L)
    index = np.argmin(rates_L)
    end = datetime.now()
    print(f"Successfully ran K mean in: {end - start}")
    print(f"The best centroids for this data set is: \n{centroids_L[index]}\nThe best inertia is:\n{rates_L[index]}")
    return centroids_L[index]
def show():
    plt.show()
if __name__ == '__main__':
    print("start")

    kD = 2
    DD = 3
    ND = 100
    tD = .1
    LD = 1
    Default_data, temp1, temp2 = make_blobs(n_samples=ND, centers=kD, n_features=DD, random_state=0,
                                            return_centers=True, cluster_std=1)
    kma(data=Default_data, k=kD, D=DD, N=ND, t=tD, L=LD)
    show()
