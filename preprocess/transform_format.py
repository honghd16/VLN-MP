import os
import json
from transformers import AutoTokenizer
import networkx as nx
import numpy as np

os.makedirs('new_ndh', exist_ok=True)
files = [x for x in os.listdir('ndh') if 'test' not in x]

def get_tokenizer():
    cfg_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def encode_instr(tokenizer, instr):
    max_instr_len = 100
    instr_encoding = tokenizer.encode(instr, add_special_tokens=True)[:max_instr_len]
    return instr_encoding

def decode_instr(tokenizer, instr_encoding):
    instr = tokenizer.decode(instr_encoding, skip_special_tokens=True)
    return instr

def load_nav_graphs(connectivity_dir, scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]])
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs

connectivity_dir = 'connectivity'
scans = [x.split("_")[0] for x in os.listdir(connectivity_dir) if 'connectivity' in x]
graphs = load_nav_graphs(connectivity_dir, scans)

tok = get_tokenizer()
for file in files:
    path = os.path.join('ndh', file)
    with open(path, 'r') as f:
        data = json.load(f)
        new_data = []
        for ins in data:
            target = ins['target']
            dialog = ins['dialog_history']
            instr_encoding = ins['instr_encoding']

            # new_instr = f"Go to the room with the {target}"
            new_instr = f"Find this"
            new_instr_encoding = encode_instr(tok, new_instr)

            scan = ins['scan']
            G = graphs[scan]
            start_node = ins['start_pano']
            end_nodes = ins['end_panos']
            # shorest_distance = 1000000
            # for end_node in end_nodes:
            #     distance = nx.shortest_path_length(G, start_node, end_node, weight='weight')
            #     if distance < shorest_distance:
            #         shorest_distance = distance
            #         final_end_node = end_node
            # end_node = final_end_node
            # distance = nx.shortest_path_length(G, start_node, end_node, weight='weight')
            # path = nx.shortest_path(G, start_node, end_node, weight='weight')
            # new_ins = {
            #     "distance": distance,
            #     "scan": scan,
            #     "path_id": ins['instr_id'],
            #     "path": path,
            #     "heading": ins["start_heading"],
            #     "instructions": [new_instr],
            #     "instr_encodings": [new_instr_encoding],
            #     "end_panos": end_nodes,
            #     }

            new_ins = {
                "instr_id": ins['instr_id'],
                "scan": scan, 
                "start_pano": start_node,
                "start_heading": ins["start_heading"],
                "end_panos": end_nodes,
                "instr_encoding": new_instr_encoding
                }
    
            new_data.append(new_ins)

    new_path = os.path.join('image_ndh', file)
    os.makedirs("image_ndh", exist_ok=True)
    with open(new_path, 'w') as f:
        json.dump(new_data, f, indent=4)


            
    