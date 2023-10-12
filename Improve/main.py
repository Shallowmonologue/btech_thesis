import community
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from random import choice
import numpy as np
import seaborn as sns
from numpy.random import choice
from tqdm.auto import tqdm


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


def degreeDecreaseProbability(G):
    probabilities = [float(G.degree(i)+1e-3 / (G.number_of_nodes() * 1.0)) for i in range(G.number_of_nodes())]
    return probabilities


def internalCommunityEdges(nodes_list, community_label):
    internaledges = 0
    for node in nodes_list:
        p = len(np.where(S[[n for n in G.neighbors(node)]] == community_label)[0])
        internaledges += p

    inter = internaledges / (2.0)
    return inter


def community_ext(nodes_list):
    unique_external_nodes = []
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

    totalCommunityEdges = G.number_of_edges() - internalCommunityEdges(nodes_list, community_label)
    modularityComm = total_connections - ((totalExternalNodeDegree * communityDegree) / (2.0 * totalCommunityEdges))
    modularityComm = modularityComm / (2.0 * totalCommunityEdges)
    return modularityComm


def join(node_i, S, G, lamda, probabilities, isImitate = True):
    max_utility = []
    community_toJoin = []
    currentCommunity = S[node_i]
    currentCommunity_nodes_list = np.where(S == currentCommunity)[0]

    loss = utility(node_i, currentCommunity_nodes_list, G)

    for neighbor in G[node_i]:
        community_label = S[neighbor]
        community_toJoin.append(community_label)
        nodes_list = np.where(S == community_label)[0]
        gain = utility(node_i, nodes_list, G)

        community_label = S[neighbor]
        same_community_nodes_list = np.where(S == community_label)[0]
        other = community_ext(same_community_nodes_list)
        val = lamda * (gain - loss) + (1 - lamda) * other
        max_utility.append(val)

    # Imitate neighbor's decision
    if isImitate:
        n = len([i for i in G.neighbors(node_i)])
        for neighbor in G[node_i]:
            community_label = S[neighbor]
            a = community_toJoin.index(community_label)
            max_utility[a] += 1/n

    maximum_utility_val = max(max_utility)
    if (maximum_utility_val) > 0:
        probabilities[node_i] *= (1 - lamda)
    maxUtilityIndex = max_utility.index(maximum_utility_val)
    communityIndex = community_toJoin[maxUtilityIndex]

    # update the community for node_i
    S[node_i] = communityIndex
    return S


def partitionModularity(mod_list, G):
    totalModularity = 0
    for node_i in range(G.number_of_nodes()):
        totalModularity += utility(node_i, np.where(mod_list == mod_list[node_i])[0], G)
    return (totalModularity / (G.number_of_edges() * 2.0))


def communityMerge(S, G, LAMBDA, probab, isImitate):
    ...


def communityDetect(S, G, nIter, LAMBDA=1, isImitate=True):
    initial_S = np.array([("C" + str(i)) for i in range(G.number_of_nodes())])
    nodesList = [node_k for node_k in range(G.number_of_nodes())]
    if isImitate:
        probabilities = degreeDecreaseProbability(G)
    else:
        probabilities = [float(1 / (G.number_of_nodes() * 1.0)) for i in range(G.number_of_nodes())]

    prev_S = initial_S
    val = []
    with tqdm(total=nIter) as t:
        for n in range(nIter):
            probab = np.array(probabilities)
            probab /= probab.sum()
            random_node = choice(nodesList, 1, p=probab)
            S = join(random_node[0], S, G, LAMBDA, probab, isImitate)
            prev_S = S

            result = partitionModularity(S, G)
            val.append(result)
            t.set_postfix(modularity=result)
            t.update(1)

            if n > 100:
                if val[-100:] == val[-200:-100]:
                    t.close()
                    break
    return val, prev_S


def loadDataset(path):
    l1 = []
    l2 = []
    file = open(path)
    for line in file:
        l1.append(int(line.split()[0]))
        l2.append(int(line.split()[1]))
    df = pd.DataFrame()
    df[1] = l1
    df[2] = l2

    G = nx.from_pandas_edgelist(df, 1, 2)
    mapping = {}
    i = 0
    for node in G.nodes:
        mapping[node] = i
        i += 1
    G = nx.relabel_nodes(G, mapping)
    print(nx.info(G))
    return G


def drawCompareGraph(value1, value2, name_set):
    dataset, parameters = name_set
    # Show the result
    plt.style.use('fivethirtyeight')
    sns.set(style='whitegrid')
    plt.title(dataset + "Dataset")
    sns.lineplot(x=[i for i in range(len(value1[0]))], y=value1[0], label="lambda_" + str(parameters))
    sns.lineplot(x=[i for i in range(len(value2[0]))], y=value2[0],
                 label="lambda_" + str(parameters) + "_improve")

    plt.xlabel("No. of iterations")
    plt.ylabel("Modularity Value")
    plt.legend()
    plt.savefig(dataset + "_lambda_" + str(parameters) + ".jpg")
    plt.clf()


def saveResult(value1, value2, name_set):
    dataset, parameters = name_set
    df = pd.DataFrame({'base': np.array(value1),
                       'improve': np.array(value2),
                       'average_base': sum(value1) / repeat,
                       'average_improve': sum(value2) / repeat})
    df.to_csv(dataset + '_lambda_' + str(parameters) + '.csv', index=False)


if __name__ == '__main__':
    # hyperparameters
    # dataset_name = ['dolphins', 'enron', 'karate', 'football', 'retweet']
    # dataset_name = ['fb-pages-food','fb-pages-politician','erdos992']
    dataset_name = ['enron']
    hyper_lambda = [0.2, 0.5, 0.8, 1]
    repeat = 10

    for dataset in dataset_name:
        # load dataset and process into a graph
        G = loadDataset('./dataset/' + dataset + '.txt')
        tqdm.write('the Dataset is: '+dataset)
        niter = G.number_of_nodes()**2

        for lambda_iter in hyper_lambda:
            # repeat to caculate average
            tqdm.write('\nthe hyperparameter lamda is: '+str(lambda_iter))
            all_value_base = []
            all_value_improve = []

            for repeat_iter in range(repeat):
                # Storage the base info of graph，calculate LAMBDA
                tqdm.write('---repeat for '+ str(repeat_iter)+ ' times---')
                A = nx.to_numpy_array(G)

                # Calculate origin algorithm
                S = np.array([("C" + str(i)) for i in range(G.number_of_nodes())])
                value_base = communityDetect(S, G, nIter=niter, LAMBDA=lambda_iter, isImitate=False)
                modularity_base = partitionModularity(S, G)
                tqdm.write('BASE: Modularity = ' + str(modularity_base))

                # Calculate new algorithm
                S = np.array([("C" + str(i)) for i in range(G.number_of_nodes())])
                value_improve = communityDetect(S, G, nIter=niter, LAMBDA=lambda_iter, isImitate=True)
                modularity_improve = partitionModularity(S, G)
                tqdm.write('IMPROVE: Modularity = ' + str(modularity_improve))

                # draw compared Pic for the first times
                if repeat_iter == 0:
                    drawCompareGraph(value_base, value_improve, ['./pic/'+dataset, lambda_iter])

                # store repeat
                all_value_base.append(modularity_base)
                all_value_improve.append(modularity_improve)

            # save to .csv
            saveResult(all_value_base, all_value_improve, ['./result/'+dataset, lambda_iter])