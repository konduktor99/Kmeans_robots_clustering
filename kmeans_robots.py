

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as met 



CLUSTER_AMOUNT = 4

COLOR_MAP = np.random.rand(CLUSTER_AMOUNT, 3)

ITERATIONS = 3


#PREPARING 2 MATRICES OF DATASET FOR DISTANCE AND POSITION
def prepare_data():
    data=pd.read_csv('beacon_readings.csv')
  
    distance_data = np.array([data['Distance A'],data['Distance B'],data['Distance C']])
    
    pos_x=data.apply(lambda row : row['Position X'] + np.random.randint(-6, 6), axis = 1)
    pos_y=data.apply(lambda row : row['Position Y'] + np.random.randint(-6, 6), axis = 1)
    position_data = np.array([pos_x,pos_y])
    
    
    return distance_data, position_data


    
#VISUALIZING CLUSTERS BY DISTANCE COORDINATES IN 3D AND BUBBLE CHARTS
def show_clusters(data, centroids, classification,title):
    
    a,b,c =data
    colors = COLOR_MAP[classification]
    
    fig = plt.figure()
    
    ax = plt.subplot2grid((25, 25), (0, 0), colspan=10, rowspan=10, projection='3d')
    ax2 = plt.subplot2grid((25, 25), (10, 10), rowspan=15, colspan=15)
    
    ax2.scatter(c,a,s=b,alpha=0.3,c=colors,linewidths=1.3,sizes=(20,150),edgecolor='k')
    ax2.set_xlabel('Distance C')
    ax2.set_ylabel('Distance A')

    ax.scatter(a,c,b,alpha=0.3,c=colors,linewidths=0.8,marker='x')
    ax.scatter(centroids[0], centroids[1], centroids[2], marker='s',c=COLOR_MAP,alpha=1)
    ax.set_xlabel('Distance A')
    ax.set_ylabel('Distance C')
    ax.set_zlabel('Distance B')
    plt.title(title)
    plt.savefig('clustering_results_dist.png')
    plt.show()
    return plt
    
    
#VISUALIZING  POSITIONS OF GROUPING OBJECTS
def show_positions(data,classification,title):
    
    colors = COLOR_MAP[classification]
    x,y=data

    plt.scatter(x, y,c=colors)
    plt.xlabel('Positon X')
    plt.ylabel('Position Y')
    plt.title(title)
    plt.savefig('clustering_results_pos.png')
    plt.show()
    
        
#GENERATING INITIAL CENTROIDS BY SELECTING RANDOM POINTS  
def generate_initial_centroids(data):
    data_t=data.T
   
    #randomly choosing 3 points as centroids
    indexes = np.random.choice(range(len(data_t)),CLUSTER_AMOUNT, replace=False)
    centroids=data_t[indexes]    
    centroids=centroids.T
    
   
    return centroids

#GENERATING INITIAL CENTROIDS BY MEAN OF COORDINATES WITHIN THE GROUP  
def generate_new_centroids(data, classification):
    global CLUSTER_AMOUNT,COLOR_MAP
     
    new_centroids=[]
    
    for c in range(CLUSTER_AMOUNT):
   
        index_list = [ i for i in range(len(classification)) if classification[i] == c ]
        
        data_df=pd.DataFrame(data)     
        cluster_data_df = data_df[index_list]
        
        new_centroid = cluster_data_df.mean(axis=1)
       
        #Getting rid of "ghost centroids"
        if not index_list:
            CLUSTER_AMOUNT-=1
            COLOR_MAP = np.delete(COLOR_MAP,c,0)
            print(c)
            print(len(COLOR_MAP))
        else:
            new_centroid=new_centroid.to_numpy() 
            new_centroids.append(new_centroid)
        
    new_centroids = np.array(new_centroids).T
    return new_centroids

#ASSIGNING POINTS TO THE CLOSEST CENTROID
def classify(data, centroids):
    distances = []
    
    for point in data.T:
        point_distances = []
        
        for centroid in centroids.T:
            single_dist = np.linalg.norm(point-centroid)
                
            point_distances.append(single_dist)
          
        distances.append(point_distances)
    
    classification = np.argmin(distances, axis=1)
   
    return classification
    
    
    
def main():
    
    initialTitle = 'INITIAL CLUSTERING'
    finalTitle = 'FINAL CLUSTERING'
    
    counter=0
    dist_data, pos_data=prepare_data()
    centroids = generate_initial_centroids(dist_data)    
    classification = classify(dist_data,centroids)
    show_clusters(dist_data, centroids, classification, initialTitle)
    
    show_positions(pos_data,classification, initialTitle)
    
    print(f'FOR K = {CLUSTER_AMOUNT} in K-MEANS ALGORITHM')
    while True:
        
        davies_bouldin_index = met.davies_bouldin_score(dist_data.T,classification)
        print(f'DAVIES-BOULDIN INDEX AFTER CLASSIFICATION NO. {counter}  = ',davies_bouldin_index)
   
        new_centroids = generate_new_centroids(dist_data, classification)
        new_classification = classify(dist_data,new_centroids)
        
        comparison=classification==new_classification
        if(comparison.all()):
            break
        
        classification = new_classification
        centroids=new_centroids
        counter+=1
     
    plt = show_clusters(dist_data, centroids, classification,finalTitle)
    
    show_positions(pos_data,classification,finalTitle)
    
    
if __name__ == '__main__':
    main()


