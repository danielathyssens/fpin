
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_cluster_labels(X_dat,X_dat_kool):
  # Clustering --> add as many clusters as there are vehicles;
  nr_vehicles=X_dat[0]['num_vehicles']
  cust_coords=[np.array(x[1]) for x in X_dat_kool]
  cust_coords_dems=[np.concatenate((np.array(X_dat_kool[0][1]),np.array(X_dat[0]['demands'][1:])[:,None]),axis=1)]
  depo_coords=[np.array(x[0]) for x in X_dat_kool]
  cust_coord_label=[]
  cust_coord_centr=[]
  kmeans = KMeans(
      init="random",
      n_clusters=nr_vehicles,
      n_init=10,
      max_iter=300,
      random_state=42
      )
  for i in range(len(cust_coords)):
    kmeans.fit(cust_coords[i])
    cust_coord_label.append(kmeans.labels_)
    cust_coord_centr.append(cust_coord_centr)

  return cust_coord_label


def transform_X2d(X_dat,X_dat_kool,coord_labels,nr_of_datapoints=1000):
    Dists_lst=[] # list of distance matrices
    X_groups20_new_2d=[]  # list of groups (training instances)
    nr_of_datapoints=len(X_dat)
    too_much_demand_test=0
    for i in range(nr_of_datapoints):
        if np.sum(X_dat[i]['demands'])<=np.sum(X_dat[i]['vehicle_capacities']):
          # VEHICLE FLEET
          n_=np.arange(1,X_dat[i]['num_vehicles']+1)
          v_ids=np.array(pd.get_dummies(n_))[:,:-1]
          vehicle_tuples=list(zip([(x) for x in n_],[n_[-1]] * n_[-1],
                                  X_dat[i]['vehicle_capacities'],
                                  [sum(np.array(X_dat[i]['demands']))/X_dat[i]['vehicle_capacities'][0]]* n_[-1]))
          vehicle_arrays=[np.asarray(x) for x in vehicle_tuples]
          vehicle_gr=np.vstack(vehicle_arrays)
          vehicle_gr[:,-2]=vehicle_gr[:,-2]/X_dat[i]['vehicle_capacities'][0]
          vehicle_group=np.concatenate((v_ids,vehicle_gr[:,1:]),axis=1)

          # DISTANCE MATRIX
          Dists_lst.append(X_dat[i]['distance_matrix'])

          # DEMANDS
          demand_array=np.array(X_dat[i]['demands'])/X_dat[i]['vehicle_capacities'][0]
          #/X_dat[i]['vehicle_capacities'][0]
          demand_array=demand_array.reshape(1,len(demand_array))

          # DEPOT
          Depot_coord=np.array(X_dat_kool[i][0])
          Depot_coord=Depot_coord.reshape(1,len(Depot_coord))
          # normalized depot closeness, v=depot: (N-1)/sum_u(d(v,u))
          Depot_centrality=np.array(len(X_dat[i]['distance_matrix'])-1)/np.sum((X_dat[i]['distance_matrix'])[0]).reshape(-1,1)
          Depot_gr=np.concatenate((Depot_coord,demand_array[:,0].reshape(-1,1),Depot_centrality),axis=1)

          # CUSTOMER NODES
          Customer_coord=np.array(X_dat_kool[i][1])
          # To depot distance
          #To_depot_dist=X_dat[i]['distance_matrix'][0]
          #To_depot_dist=To_depot_dist.reshape(1,len(To_depot_dist))
          # KMEANS LABELS
          ##plain number
          #label=np.array(coord_labels[i]).reshape(1,len(coord_labels[i])).transpose()
          ## one hot encoded
          label_hot=np.atleast_2d(np.array(pd.get_dummies(coord_labels[i]))[:,:-1])
          Customer_gr=np.concatenate((Customer_coord,demand_array[:,1:].transpose(),label_hot),axis=1)
        
          All_dist_mat=X_dat[i]['distance_matrix']
        
          # Combine to Instances
          X_groups20_new_2d.append((vehicle_group,Depot_gr,Customer_gr,demand_array,
                                All_dist_mat))
        else:
          #print('TOO MUCH DEMAND for 4 vehicles with capa 30')
          too_much_demand_test+=1
    return X_groups20_new_2d
    
def transform_X2d_old(X_dat,X_dat_kool,nr_of_datapoints=1000):
    Dists_lst=[] # list of distance matrices
    X_groups20_new_2d=[]  # list of groups (training instances)
    nr_of_datapoints=len(X_dat)
    too_much_demand_test=0
    for i in range(nr_of_datapoints):
        if np.sum(X_dat[i]['demands'])<=np.sum(X_dat[i]['vehicle_capacities']):
          # VEHICLE FLEET
          n_=np.arange(1,X_dat[i]['num_vehicles']+1)
          #v_ids=np.array(pd.get_dummies(n_))[:,:-1]
          vehicle_tuples=list(zip([(x/n_[-1]) for x in n_],[n_[-1]] * n_[-1],
                                  X_dat[i]['vehicle_capacities'],
                                  [sum(np.array(X_dat[i]['demands']))/X_dat[i]['vehicle_capacities'][0]]* n_[-1]))
          vehicle_arrays=[np.asarray(x) for x in vehicle_tuples]
          #.reshape(len(x),1)
          vehicle_gr=np.vstack(vehicle_arrays)
          vehicle_gr[:,-2]=vehicle_gr[:,-2]/X_dat[i]['vehicle_capacities'][0]

          # DISTANCE MATRIX
          Dists_lst.append(X_dat[i]['distance_matrix'])

          # DEMANDS
          demand_array=np.array(X_dat[i]['demands'])/X_dat[i]['vehicle_capacities'][0]
          #/X_dat[i]['vehicle_capacities'][0]
          demand_array=demand_array.reshape(1,len(demand_array))

          # DEPOT
          Depot_coord=np.array(X_dat_kool[i][0])
          Depot_coord=Depot_coord.reshape(1,len(Depot_coord))
          # normalized depot closeness, v=depot: (N-1)/sum_u(d(v,u))
          Depot_centrality=np.array(len(X_dat[i]['distance_matrix'])-1)/np.sum((X_dat[i]['distance_matrix'])[0]).reshape(-1,1)
          Depot_gr=np.concatenate((Depot_coord,demand_array[:,0].reshape(-1,1),Depot_centrality),axis=1)

          # CUSTOMER NODES
          Customer_coord=np.array(X_dat_kool[i][1])
          # To depot distance
          To_depot_dist=X_dat[i]['distance_matrix'][0]
          To_depot_dist=To_depot_dist.reshape(1,len(To_depot_dist))
          #print(To_depot_dist)
          Customer_gr=np.concatenate((Customer_coord,demand_array[:,1:].transpose(),To_depot_dist[:,1:].transpose()),axis=1)
          #[:,1:]
          All_dist_mat=X_dat[i]['distance_matrix']
        
          # Combine to Instances
          X_groups20_new_2d.append((vehicle_gr,Depot_gr,Customer_gr,demand_array,
                                All_dist_mat))
        else:
          #print('TOO MUCH DEMAND for 4 vehicles with capa 30')
          too_much_demand_test+=1
    return X_groups20_new_2d

def preprocess(Kool_Test_Xdat,Kool_Test_Xdat_orig):
    #get cluster labels
    coord_labels=get_cluster_labels(Kool_Test_Xdat,Kool_Test_Xdat_orig)
    #transform test data
    Kool_Test_X=transform_X2d(Kool_Test_Xdat,Kool_Test_Xdat_orig,coord_labels,nr_of_datapoints=len(Kool_Test_Xdat))
    return Kool_Test_X