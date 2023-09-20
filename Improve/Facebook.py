import community
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from random import choice
import numpy as np
import seaborn as sns
from numpy.random import choice
from tqdm.auto import tqdm
from sklearn.metrics.cluster import normalized_mutual_info_score

sns.set(style='whitegrid')


def utility(node_i, S, G):
    total_utility = 0
    k_i = G.degree[node_i]
    for node_j in S:
        if node_i != node_j:
            curr_edges = A[node_i][node_j]

            k_j = G.degree[node_j]
            m = G.number_of_edges()
            degree_prod = (k_i * k_j) / (2.0 * m)

            indiv_utility = curr_edges - degree_prod
            total_utility += indiv_utility
    return total_utility


def internalCommunityEdges(nodes_list, community_label):
    internaledges = 0
    for node in nodes_list:
        p = len(np.where(S[[n for n in G.neighbors(node)]] == community_label)[0])
        # print(node)
        internaledges += p

    inter = internaledges / (2.0)
    return inter


def community_ext(nodes_list):
    unique_external_nodes = []
    total_sum = 0
    community_label = S[nodes_list[0]]
    total_connections = 0
    totalExternalNodeDegree = 0
    for nod in nodes_list:
        # neighbors of node in that community
        for neig in G[nod]:
            # external node
            if S[neig] != S[nod]:
                if neig not in unique_external_nodes:
                    unique_external_nodes.append(neig)
                total_connections += 1

    communityDegree = len(unique_external_nodes)
    for externalNode in unique_external_nodes:
        totalExternalNodeDegree += G.degree(externalNode)

    totalCommunityEdges = totalEdges - internalCommunityEdges(nodes_list, community_label)
    modularityComm = total_connections - ((totalExternalNodeDegree * communityDegree) / (2.0 * totalCommunityEdges))
    modularityComm = modularityComm / (2.0 * totalCommunityEdges)
    return modularityComm


def join(node_i, S, G, lamda, probabilities):
    max_utility = []
    community_toJoin = []
    currentCommunity = S[node_i]
    currentCommunity_nodes_list = np.where(S == currentCommunity)[0]

    loss = utility(node_i, currentCommunity_nodes_list, G)
    # print(loss)
    for neighbor in G[node_i]:
        community_label = S[neighbor]
        community_toJoin.append(community_label)
        nodes_list = np.where(S == community_label)[0]
        gain = utility(node_i, nodes_list, G)

        community_label = S[neighbor]
        #       print(community_label)
        # community_toJoin.append(community_label)
        same_community_nodes_list = np.where(S == community_label)[0]
        other = community_ext(same_community_nodes_list)
        val = lamda * (gain - loss) + (1 - lamda) * (other)
        max_utility.append(val)
        # print(nodes_list)

    maximum_utility_val = max(max_utility)
    if (maximum_utility_val) > 0:
        probabilities[node_i] *= (1 - lamda)
    maxUtilityIndex = max_utility.index(maximum_utility_val)
    communityIndex = community_toJoin[maxUtilityIndex]
    # print(communityIndex)

    # update the community for node_i
    S[node_i] = communityIndex
    return S


def partitionModularity(mod_list, G):
    totalModularity = 0
    for node_i in range(G.number_of_nodes()):
        totalModularity += utility(node_i, np.where(mod_list == mod_list[node_i])[0], G)
    return (totalModularity / (G.number_of_edges() * 2.0))


def communityDetect(S, G, nIter, LAMBDA=1):
    total_nodes = G.number_of_nodes()
    i = 0
    prev_S = initial_S
    val = []
    #   while(i<nIter):
    for _ in tqdm(range(nIter)):
        #         print("Iteration No. : ", i+1)
        probab = np.array(probabilities)
        probab /= probab.sum()
        random_node = choice(nodesList, 1, p=probab)
        S = join(random_node[0], S, G, LAMBDA, probab)
        val.append(partitionModularity(S, G))
        prev_S = S
    #         i += 1

    return val, prev_S


if __name__ == '__main__':
    # load dataset and process into a graph
    l1 = []
    l2 = []
    file = open('./ia-enron-only.txt')
    for line in file:
        l1.append(int(line.split()[0]))
        l2.append(int(line.split()[1]))
    df = pd.DataFrame()
    df[1] = l1
    df[2] = l2

    G = nx.Graph()
    G = nx.from_pandas_edgelist(df, 1, 2)
    mapping = {}
    i = 0
    for node in G.nodes:
        mapping[node] = i
        i += 1
    G = nx.relabel_nodes(G, mapping)
    print(nx.info(G))

    # Calculate the best partition for MODULARITY as evaluation
    part = community.best_partition(G)
    print('the Best Partition is: ', community.modularity(part, G))


    # Storage the base info of graph，Caculate LAMBDA=0.2
    totalEdges = G.number_of_edges()
    totalNodes = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    S = np.array([("C" + str(i)) for i in range(G.number_of_nodes())])
    initial_S = np.array([("C" + str(i)) for i in range(G.number_of_nodes())])
    nodesList = [node_k for node_k in range(G.number_of_nodes())]
    probabilities = [float(1 / (G.number_of_nodes() * 1.0)) for i in range(G.number_of_nodes())]
    value_lambda_point2 = communityDetect(S, G, nIter=1000, LAMBDA=0.2)
    print('Partition Modularity = ' + str(partitionModularity(S, G)))

    # Storage the base info of graph，Caculate LAMBDA=0.8
    totalEdges = G.number_of_edges()
    totalNodes = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    S = np.array([("C" + str(i)) for i in range(G.number_of_nodes())])
    initial_S = np.array([("C" + str(i)) for i in range(G.number_of_nodes())])
    nodesList = [node_k for node_k in range(G.number_of_nodes())]
    probabilities = [float(1 / (G.number_of_nodes() * 1.0)) for i in range(G.number_of_nodes())]
    value_lambda_point8 = communityDetect(S, G, nIter=1000, LAMBDA=0.8)
    print('Partition Modularity = ' + str(partitionModularity(S, G)))

    # Storage the base info of graph，Caculate LAMBDA=1
    totalEdges = G.number_of_edges()
    totalNodes = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    S = np.array([("C" + str(i)) for i in range(G.number_of_nodes())])
    initial_S = np.array([("C" + str(i)) for i in range(G.number_of_nodes())])
    nodesList = [node_k for node_k in range(G.number_of_nodes())]
    probabilities = [float(1 / (G.number_of_nodes() * 1.0)) for i in range(G.number_of_nodes())]
    value_lambda_1 = communityDetect(S, G, nIter=1000, LAMBDA=1)
    print('Partition Modularity = ' + str(partitionModularity(S, G)))


    # Show the result
    plt.style.use('fivethirtyeight')
    sns.set(style='whitegrid')
    plt.title("Enron Dataset")
    sns.lineplot([i for i in range(len(value_lambda_point2[0]))], value_lambda_point2[0], label="lambda 0.2")
    sns.lineplot([i for i in range(len(value_lambda_point8[0]))], value_lambda_point8[0], label="lambda 0.8")

    plt.xlabel("No. of iterations")
    plt.ylabel("Modularity Value")
    plt.legend()
    plt.savefig("Enron_1.svg")

    plt.style.use('fivethirtyeight')
    sns.set(style='whitegrid')
    plt.title("Enron Dataset")
    sns.lineplot([i for i in range(len(value_lambda_1[0]))], value_lambda_1[0], label="lambda 1")
    sns.lineplot([i for i in range(len(value_lambda_point8[0]))], value_lambda_point8[0], label="lambda 0.8")

    plt.xlabel("No. of iterations")
    plt.ylabel("Modularity Value")
    plt.legend()
    plt.savefig("Enron_2.svg")
