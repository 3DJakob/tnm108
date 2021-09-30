import numpy as np
trainingData = np.array([
    [1, 4, 1],
    [1, 2, 2],
    [1, 4, 2],
    [2, 1, 2],
    [1, 1, 1],
    [2, 4, 2],
    [1, 1, 2],
    [2, 1, 1],
])

def manHattan(p1, p2):
    return np.sum(abs(p1 - p2))
 

def kMeans(data):
    # Select centroids
    c1 = data[1]
    c2 = data[5]

    # Create and fill in distances
    rows, columns = data.shape
    centroidDistance = np.zeros((rows,2))
    
    for i, dist in enumerate(data):
        centroidDistance[i,0] = manHattan(c1, dist)
        
    for i, dist in enumerate(data):
        centroidDistance[i,1] = manHattan(c2, dist)
        
    # table = np.hstack((data, centroidDistance))
    # print(table)
    
    # Create cluster tables
    c1Cluster = np.zeros((0,3))
    c2Cluster = np.empty((0,3))
    
    print(data[3])
    c1Cluster
    # print(c1Cluster)
    # for i, dist in enumerate(centroidDistance):
    #     if dist[0] < dist[1] :
    #         np.append(c1Cluster, data[i], axis=0)
    #     else:   
    #         np.append(c2Cluster, data[i], axis=0)
    
    print(c1Cluster)
       
        
     
   
    
    
    

kMeans(trainingData)