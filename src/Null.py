import random
import numpy as np
import summary_statistics as SS

def null_model(B,N,directed = True):
    interaction_types = []
    for k in range(N):
        den_null = np.arange(B*B)
        np.random.shuffle(den_null)
        den_null = den_null.reshape((B, B))
        if not directed:
            for i in range(B):
                for j in range(i):
                    den_null[j,i] = den_null[i,j]
        deg_list = abs(np.sort(-np.arange(B)))
        interaction_type = SS.interaction_type_prob(den_null,deg_list,directed = directed,classification ="Three")
        interaction_types.append(interaction_type)
#         for key in interaction_type.keys():
#             interaction_types.append(key)

    return interaction_types;

def get_occuring_type(interaction_types):
    occuring_types = []
    for i in range(len(interaction_types)):
        for key in interaction_types[i].keys():
            occuring_types.append(key)
    return occuring_types;

def get_dominant_type(interaction_types):
    dominant_types = []
    for i in range(len(interaction_types)):
        max_value = max(interaction_types[i].values())
        max_key = [key for key in interaction_types[i].keys() if interaction_types[i][key] == max_value]
        dominant_types.append(random.choice(max_key))
    return dominant_types;