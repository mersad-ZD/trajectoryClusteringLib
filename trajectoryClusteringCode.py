import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector as mc
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer


generalTempList = []
global_Kmeans_labels = []
n = 10
m = 1
slopeMapList = [[] * m for i in range(n)]
polarMapList = [[] * m for i in range(n)]
lenghtMapList = [[] * m for i in range(n)]
newList = [[] * m for i in range(n)]
lenghtSlopeDirMapList = [[] * m for i in range(n)]
slopDirList = [[] * m for i in range(n)]
Slope_and_b_List = [[] * m for i in range(n)]
RIGHT = 1
LEFT = -1
ZERO = 0
n_samples = 10

seg_num = int(input("please enter number of trajectory segements "))
print("seg_num is ", seg_num)

# connect to database
# mytable = mc.connect( host="localhost",user="root",passwd="admin",database="mydb")
# mycursor = mytable.cursor()
# var = 5
# mycursor.execute("select * from trajectorytable where pathID = %s " ,(var,) )
# myresult = mycursor.fetchall()
# global generalTempList
# tempList  =[]
#generalTempList = myresult
# print("mysql connected")
# print(myresult)

generalTempList = pd.read_csv(r"C:\Users\Sadaf\Desktop\trajec_dataset\trajec71.csv")

# silhouette mrthods to find optimal number of clusters
def get_nCluster_silhouette(X ,start,to):         # Valid values for start and to, are 2 to n_samples - 1
    silhouette_list = []
    k_list = range(start, to)

    for p in k_list:

        clusterer = KMeans(n_clusters=p)

        clusterer.fit(X)
        # The higher (up to 1) the better
        s = round(silhouette_score(X, clusterer.labels_), 4)
        silhouette_list.append(s)


    key = silhouette_list.index(max(silhouette_list))
    k = k_list[key]

    return k
# elbow mrthods to find optimal number of clusters
def get_nCluster_elbow(X, start,to):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(start, to), locate_elbow=False, timings=False)

    visualizer.fit(X)  # Fit the data to the visualizer
    # visualizer.show()  # Finalize and render the figure
    # print("visualizer.elbow_value is ", visualizer.elbow_value_)
    k = visualizer.elbow_value_

    return k


def kmeans1(list, n_cluster):  # for features that needs to be scaled

    x = np.array(list)
    kmeans = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=100, random_state=4)
    x = StandardScaler().fit_transform(x)

    kmeans.fit(x)

    global global_Kmeans_labels
    global_Kmeans_labels = kmeans.labels_



def kmeans2(list, n_cluster):  # for just normal vec that doesnt need scaling data
    global global_Kmeans_labels
    x = np.array(list)

    kmeans = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=100, random_state=4)
    kmeans.fit(x)

    global_Kmeans_labels = kmeans.labels_


def lineEquation(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    b = m * (-x1) + y1
    # print("slope is "+ str(m))
    # print("arz az mabdaa "+ str(b))
    return m, b


def slopeMapping(x1, y1, x2, y2):  # find duality
    return lineEquation(x1, y1, x2, y2)


def polarMapping(x1, y1, x2, y2):  # find duality
    (m, b) = lineEquation(x1, y1, x2, y2)
    a = -(m / b)
    b = (1 / b)
    return a, b


def convertoSlope(list):
    v = np.array(generalTempList)

    for i in range(10):
        for z in range(1, seg_num * 2, 2):
            t0 = z
            t1 = z + 1
            t2 = z + 2
            t3 = z + 3

            (m1, b) = slopeMapping(v[i, t0], v[i, t1], v[i, t2], v[i, t3])
            slopeMapList[i].append(m1)


    return slopeMapList


def convertoSlope_and_b(list):
    v = np.array(generalTempList)

    for i in range(10):
        for z in range(1, seg_num * 2, 2):
            t0 = z
            t1 = z + 1
            t2 = z + 2
            t3 = z + 3

            (m1, b) = slopeMapping(v[i, t0], v[i, t1], v[i, t2], v[i, t3])
            Slope_and_b_List[i].append(m1)
            Slope_and_b_List[i].append(b)


    return Slope_and_b_List


def convertoPolar(list):
    v = np.array(generalTempList)

    for i in range(10):
        for z in range(1, seg_num * 2, 2):
            t0 = z
            t1 = z + 1
            t2 = z + 2
            t3 = z + 3

            (m1, b) = polarMapping(v[i, t0], v[i, t1], v[i, t2], v[i, t3])
            polarMapList[i].append(m1)
            polarMapList[i].append(b)


    return polarMapList


def convertoNormalVec(list):
    return list


def euclidianDist(x1, y1, x2, y2):
    x = [x1, x2]
    y = [y1, y2]
    dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
    return dist


def directionOfPoint(xA, yA, xB, yB, xP, yP):
    xB -= xA
    yB -= yA
    xP -= xA
    yP -= yA

    cross_product = xB * yP - yB * xP

    if (cross_product > 0):
        return RIGHT

    elif (cross_product < 0):
        return LEFT

    else:
        return ZERO


def convertOlenght(list):
    newList = slopeMapList.copy()

    for i in range(len(newList)):
        k = 0
        for j in range(seg_num * 2):
            if j % 2 == 0:
                lenghtMapList[i].insert(j, newList[i][k])  # geting out slope and putting in even col
                k += 1

    v = np.array(generalTempList)
    tempList = []
    for i in range(10):
        for z in range(1, seg_num * 2, 2):
            t0 = z
            t1 = z + 1
            t2 = z + 2
            t3 = z + 3

            dist = euclidianDist(v[i, t0], v[i, t1], v[i, t2], v[i, t3])
            tempList.append(dist)
    k = 0
    for i in range(len(newList)):
        for j in range(seg_num * 2):
            if j % 2 == 1:
                lenghtMapList[i].insert(j, tempList[k])
                k += 1

    print("lenghtMapList lenght: ", len(lenghtMapList[0]))
    return lenghtMapList


def convertOlenghtSlopeDir(list):
    newList = []
    newList = slopeMapList.copy()

    tempList = []
    for i in range(len(newList)):
        k = 0
        for j in range(len(newList[i])):
            # if j % 2 == 0:
            tempList.append(newList[i][k])  # all slope in one vectore in this list
            k += 1

    k = 0
    for i in range(10):
        # print("new list is ", len(newList))
        a = int(seg_num * 3)

        for j in range(a):
            if j % 3 == 0:
                lenghtSlopeDirMapList[i].insert(j, tempList[k])  # geting out slope and putting in 3k col
                k += 1

    v = np.array(generalTempList)
    tempList = []
    for i in range(10):
        for z in range(1, seg_num * 2, 2):
            t0 = z
            t1 = z + 1
            t2 = z + 2
            t3 = z + 3

            dist = euclidianDist(v[i, t0], v[i, t1], v[i, t2], v[i, t3])  # calculate euclidian Distance as a feature
            tempList.append(dist)

    k = 0
    for i in range(len(newList)):
        a = int(seg_num * 3)
        for j in range(a):
            if j % 3 == 1:
                lenghtSlopeDirMapList[i].insert(j, tempList[k])  # put euclidian distance feature in 3k+1 th column
                k += 1

    tempList = []
    for i in range(10):
        for z in range(1, (seg_num - 1) * 2, 2):
            t0 = z
            t1 = z + 1
            t2 = z + 2
            t3 = z + 3
            t4 = z + 4
            t5 = z + 5

            direction = directionOfPoint(v[i, t0], v[i, t1], v[i, t2], v[i, t3], v[i, t4], v[i, t5])  # find direction
            tempList.append(direction)

        tempList.append(1)

    k = 0
    for i in range(len(newList)):
        a = int(seg_num * 3)  # 15 = 5 slope + 5 lenght + 5 direction
        for j in range(a):
            if j % 3 == 2:
                lenghtSlopeDirMapList[i].insert(j, tempList[k])  # add direction feature in 3k+2 col
                k += 1

    # print("lenghtSlopeDirMapList lenght: ", len(lenghtSlopeDirMapList[0]))
    return lenghtSlopeDirMapList


def convertToSlopDir(list):
    newList = []
    slopList = slopeMapList.copy()

    for i in range(len(slopList)):
        k = 0
        for j in range(seg_num * 2):  # every segment has 2 feature
            if j % 2 == 0:
                # print("k is ", k)
                slopDirList[i].insert(j, slopList[i][k])  # geting out slope and putting in even col
                k += 1

    v = np.array(generalTempList)
    directionList = []
    for i in range(10):
        for z in range(1, (seg_num - 1) * 2, 2):
            t0 = z
            t1 = z + 1
            t2 = z + 2
            t3 = z + 3
            t4 = z + 4
            t5 = z + 5

            direction = directionOfPoint(v[i, t0], v[i, t1], v[i, t2], v[i, t3], v[i, t4], v[i, t5])  # find direction
            directionList.append(direction)

        directionList.append(1)


    k = 0
    for i in range(len(slopList)):
        for j in range(seg_num * 2):
            if j % 2 == 1:
                slopDirList[i].insert(j, directionList[k])
                k += 1

    print("slopDirList lenght: ", len(slopDirList[0]))
    return slopDirList


def xplot(h):
    t = np.array(generalTempList)
    colors = ['blue', 'black', 'yellow', 'red', 'green', 'orange', 'pink', 'magenta', 'khaki']

    plt.figure(h)

    x1_idx = [(2 * k) + 1 for k in range(seg_num + 1)]
    y1_idx = [(2 * k) + 2 for k in range(seg_num + 1)]
    x1 = []
    y1 = []
    for i in range(10):
        for k1, k2 in zip(x1_idx, y1_idx):
            x1.append(t[i, k1])
            y1.append(t[i, k2])


        plt.plot(x1, y1, color=colors[global_Kmeans_labels[i]], linestyle='dashed', linewidth=2,
                 marker='o', markerfacecolor='pink', markersize=8)
        x1 = []
        y1 = []

    if h == 1:
        plt.title(" normal vector x y")
    elif h == 2:
        plt.title("slope and y intersection")
    elif h == 3:
        plt.title("slope and lenght")
    elif h == 4:
        plt.title("slope and direction and even lenght")
    else:
        plt.title("slope and direction")

    plt.xlabel("x")
    plt.ylabel("y")

    if h == 5:
        plt.show()

# optimal number of clusters between two methods by using maximum of elbow and silhouette
def opt_nCluster(x, start, to):
    x = np.array(x)
    x = StandardScaler().fit_transform(x)

    k1 = get_nCluster_silhouette(x, start, to)
    k2 = get_nCluster_elbow(x, start, to)

    if k2 is None:
        k2=0
    # print("nCluster_elbow is ", k2)

    max_opt_numbers = max((k1,k2))

    return max_opt_numbers +1


convertoSlope(generalTempList)

x1 = convertoNormalVec(generalTempList)
n_cluster1 = opt_nCluster(x1, 2, n_samples-1)

x2 = convertoSlope_and_b(generalTempList)
n_cluster2 = opt_nCluster(x2, 2, n_samples-1)

x3 = convertOlenght(generalTempList)
n_cluster3 = opt_nCluster(x3, 2, n_samples-1)

x4 = convertOlenghtSlopeDir(generalTempList)
n_cluster4 = opt_nCluster(x4, 2, 10)

x5 = x = convertToSlopDir(generalTempList)
n_cluster5 = opt_nCluster(x5, 2, n_samples-1)

kmeans2(x1, n_cluster=n_cluster1)
xplot(1)
print("-----------------")

kmeans1(x2, n_cluster=n_cluster2)
xplot(2)
print("-----------------")

kmeans1(x3, n_cluster=n_cluster3)
xplot(3)
print("-----------------")

kmeans1(x4, n_cluster=n_cluster4)
xplot(4)
print("-----------------")

kmeans1(x5, n_cluster=n_cluster5)
xplot(5)



# convertoPolar()









