import pandas as pd
def is_more_specific(h1, h2):
    more_specific_parts = []
    for x, y in zip(h1, h2):
        part = (x == y) or (x == '?')
        more_specific_parts.append(part)
    return all(more_specific_parts)
def generalize_hypothesis(h, instance):
    new_h = []
    for x, y in zip(h, instance):
        if x == '0':
            new_h.append(y)
        elif x != y:
            new_h.append('?')
        else:
            new_h.append(x)
    return new_h
def specialize_hypothesis(h, domains, instance):
    specializations = []
    for i in range(len(h)):
        if h[i] == '?':
            for value in domains[i]:
                if value != instance.iloc[i]: 
                    new_h = h.copy()
                    new_h[i] = value
                    specializations.append(new_h)
        elif h[i] != instance.iloc[i]:
            new_h = h.copy()
            new_h[i] = '0'
            specializations.append(new_h)
    return specializations
def candidate_elimination_algorithm(filename, encoding='utf-8'):
    data = pd.read_csv(filename, encoding=encoding)
    attributes = data.columns[:-1]
    class_label = data.columns[-1]
    domains = [list(data[attribute].unique()) for attribute in attributes]
    G = [['?' for _ in range(len(attributes))]]
    S = [['0' for _ in range(len(attributes))]]
    for index, row in data.iterrows():
        instance = row[:-1]
        if row[class_label] == 'Yes':
            G = [g for g in G if is_more_specific(g, instance)]
            for s in S:
                if not is_more_specific(instance, s):
                    S.remove(s)
                    S.append(generalize_hypothesis(s, instance))
            S = [s for s in S if any(is_more_specific(g, s) for g in G)]
        else:
            S = [s for s in S if not is_more_specific(s, instance)]
            new_G = []
            for g in G:
                if is_more_specific(g, instance):
                    new_G.extend(specialize_hypothesis(g, domains, instance))
                else:
                    new_G.append(g)
            G = [g for g in new_G if any(is_more_specific(g, s) for s in S)]
    print("Final G:", G)
    print("Final S:", S)
filename = 'C:/Users/HP/OneDrive/Documents/weather_data.csv'
candidate_elimination_algorithm(filename, encoding='latin1')
