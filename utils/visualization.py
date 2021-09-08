import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize, StandardScaler

class Feature():
    def __init__(self, layer_number, id):
        self.L = layer_number
        self.id = id
        self.superiors = []
        self.inferiors = []
        self.upside_connected = False
        self.downside_connected = False

class Vision_tree():
    def __init__(self, name, etching_weights, depth, sigma=0.05):
        self.name = name
        self.W = etching_weights
        self.TOP = depth-1
        self.sigma = sigma
        self.tree = {}

    def create_layers(self):
        self.register_most_superior_features(self.W[self.TOP-1])
        for L in reversed(range(self.TOP)):
            self.register_connection(self.W[L], L+1, L)

    def register_most_superior_features(self, C):
        self.tree[self.TOP] = []
        X = np.maximum(1-C, 0) #relu
        norm = np.linalg.norm(X, ord=1, axis=0)
        X=np.divide(X, norm, out=np.zeros(X.shape, dtype=float), where=norm>0)
        X_embedded = TSNE(n_components=2,random_state=10).fit_transform(X.T)
        scaler = StandardScaler()
        scaler.fit(X_embedded)
        coords = scaler.transform(X_embedded)
        for idx,coord in enumerate(coords):
            f = Feature(self.TOP, idx) #create a feature node
            f.coord = coord; f.upside_connected=True;
            self.tree[self.TOP].append(f)

    def register_connection(self, C, L_upper, L_lower):
        self.tree[L_lower] = []
        num_inf, num_sup = C.shape
        X = np.maximum(1-C, 0) #relu
        norm = np.linalg.norm(X, ord=1, axis=0)
        X=np.divide(X, norm, out=np.zeros(X.shape, dtype=float), where=norm>0)
        for idx in range(num_sup):
            embedding = X[:,idx]
            #dominances = np.where(embedding>dominance_thresh, embedding, 0) # 5%이상의 기여도
            f = self.tree[L_upper][idx]
            f.inferiors = embedding #register the list of inferiors
            if np.sum(f.inferiors)>0:
                f.downside_connected = True
            else:
                f.downside_connected = False
        for idx in range(num_inf):
            contributions = X[idx]
            if np.sum(contributions)>0: # there exists connection to superior features
                f = Feature(L_lower, idx) #create a feature node
                f.upside_connected = True
                f.superiors = contributions #register the list of superiors
                mass_weights = contributions**np.array([sup.upside_connected for sup in self.tree[L_upper]])
                coord = np.matmul(np.column_stack([sup.coord for sup in self.tree[L_upper]]), mass_weights/np.sum(mass_weights))
                if np.any(coord == [sup.coord for sup in self.tree[L_upper]]):
                    #if it has the same coord with a superior feature, add a few variations to prevent to overlap some inferior features
                    coord = coord + np.random.normal(0, self.sigma, [2])
                f.coord = coord
            else:
                f = Feature(L_lower, idx)
                f.upside_connected = False
                f.superiors = contributions
                f.coord = np.array([0.0, 0.0])
            self.tree[L_lower].append(f)
    def df_tree(self, dominance_thresh=0.05):
        nodes = pd.DataFrame(columns=['x','y','z','color'])
        connections = pd.DataFrame(columns=['x1','y1','z1', 'x2', 'y2', 'z2', 'color'])
        for L in self.tree:
            for node in self.tree[L]:
                if node.downside_connected == True:
                    if node.upside_connected == True:
                        x1,y1 = node.coord; z1 = node.L+1;
                        new_node = {'x':x1, 'y':y1, 'z':z1, 'color':'red'}
                        nodes = nodes.append(new_node, ignore_index=True) #append a node position
                        sub_nodes_index = np.where(node.inferiors>dominance_thresh)[0]
                        for idx in sub_nodes_index: #find connections
                            #print(idx)
                            sub_node = self.tree[L-1][idx]
                            x2, y2 = sub_node.coord; z2 = sub_node.L+1;
                            new_connection = {'x1':x1, 'y1':y1, 'z1':z1 , 'x2':x2, 'y2':y2, 'z2':z2 ,'color':'green'}
                            connections = connections.append(new_connection, ignore_index=True) #append a connection
                    else:
                        sub_nodes_index = np.where(node.inferiors>dominance_thresh)[0]
                        sub_coords = []
                        for idx in sub_nodes_index: #find connections
                            sub_node = self.tree[L-1][idx]
                            x2, y2 = sub_node.coord; z2 = sub_node.L+1;
                            sub_coords.append([x2, y2])
                        x1, y1 = np.mean(sub_coords, axis=0); z1 = node.L+1;
                        new_node = {'x':x1, 'y':y1, 'z':z1, 'color':'blue'}
                        nodes = nodes.append(new_node, ignore_index=True) #append a node position
                        for x2, y2 in sub_coords:
                            new_connection = {'x1':x1, 'y1':y1, 'z1':z1 , 'x2':x2, 'y2':y2, 'z2':z2 ,'color':'green'}
                            connections = connections.append(new_connection, ignore_index=True) #append a connection
                else:
                    if node.upside_connected == True:
                        x1,y1 = node.coord; z1 = node.L+1;
                        new_node = {'x':x1, 'y':y1, 'z':z1, 'color':'yellow'}
                        nodes = nodes.append(new_node, ignore_index=True) #append a node position
                    else:
                        x1,y1 = node.coord + np.random.normal(0, self.sigma, [2]); z1 = node.L+1;
                        new_node = {'x':x1, 'y':y1, 'z':z1, 'color':'black'}
                        nodes = nodes.append(new_node, ignore_index=True) #append a node position
        return nodes, connections

def cal_stream(class_etching_weights, class_id):
    L4,L3,L2,L1 = class_etching_weights
    L4_indice = list(map(lambda x: supporting_features(L4,x),[class_id]))
    L4_unique = set()
    for l in L4_indice:
        L4_unique=L4_unique.union(l)
    #print("L4 indice", L4_indice)
    #print("L4 unique num {} : ".format(len(L4_unique)), L4_unique)

    L3_indice = list(map(lambda x: supporting_features(L3,x), L4_unique))
    L3_unique = set()
    for l in L3_indice:
        L3_unique=L3_unique.union(l)
    #print("L3 indice", L3_indice)
    #print("L3 unique num {} : ".format(len(L3_unique)), L3_unique)

    L2_indice = list(map(lambda x: supporting_features(L2,x),L3_unique))
    L2_unique = set()
    for l in L2_indice:
        L2_unique=L2_unique.union(l)
    #print("L2 indice", L2_indice)
    #print("L2 unique num {} : ".format(len(L2_unique)), L2_unique)

    L1_indice = list(map(lambda x: supporting_features(L1,x),L2_unique))
    L1_unique = set()
    for l in L1_indice:
        L1_unique=L1_unique.union(l)
    #print("L1 indice", L1_indice)
    #print("L1 unique num {} : ".format(len(L1_unique)), L1_unique)
    return L4_indice, L3_indice, L2_indice, L1_indice, L4_unique, L3_unique, L2_unique, L1_unique


def supporting_features(e, order):
    f = np.maximum(1-e[:,order],0) #relu
    normalized = f/np.linalg.norm(f)
    dominants = np.where(normalized>0.05, normalized, 0) # features over 5 percents of contributions
    indice = np.where(dominants>0)[0]
    return list(indice)

def get_weight_cosine_similarity(weight):
    w = weight
    if len(w.shape) == 4: #convolution
        H,W,C,N = w.shape
        row_dims = H*W*C
        col_dims = N
    else: #dense
        D,N = w.shape
        row_dims = D
        col_dims = N
    w = tf.reshape(w, (row_dims, col_dims))
    norm = tf.norm(w, axis=0)
    w = w/norm #normalize
    wT = tf.transpose(w)
    correlations = tf.matmul(wT,w)
    return correlations

def get_weight_l2norm(weight):
    w = weight
    if len(w.shape) == 4: #convolution
        H,W,C,N = w.shape
        row_dims = H*W*C
        col_dims = N
    else: #dense
        D,N = w.shape
        row_dims = D
        col_dims = N
    w = tf.reshape(w, (row_dims, col_dims))
    norm = tf.norm(w, axis=0)
    return norm

def visualize_weight_orthogonality(model):
    layers_with_weights = []
    for layer in model.layers:
        mpl.rcParams['figure.dpi'] = 800
        if len(layer.get_weights()) > 0:
            weight = layer.get_weights()[0]
            correlations = get_weight_cosine_similarity(weight)
            print(layer.name)
            sns.heatmap(correlations, vmin=-1, vmax=1, cmap='RdBu_r', center=0, annot=True, fmt='.2f',xticklabels=False, yticklabels=False,annot_kws={"size": 4})
            plt.show()
