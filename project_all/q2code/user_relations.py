import pandas as pd

import igraph

from pprint import pprint

users = pd.read_csv('../data/asker_answerer.csv', encoding='utf-8', dtype=str)
users.dropna(inplace=True)
pprint(users)

users.to_csv('../result/users_edge_list.csv', index=False, sep=' ', header=False)

G = igraph.Graph.Read_Ncol('../result/users_edge_list.csv', directed=False, weights=False)
partition = G.community_multilevel()
subgraphs = partition.subgraphs()

print(len(subgraphs))
print(G.modularity(partition))
print()
for i in range(len(subgraphs)):
    G = subgraphs[i]
    print(G.modularity(G.community_multilevel()))
    community_member_users = subgraphs[i].vs["name"]
    print(community_member_users)
    print()
