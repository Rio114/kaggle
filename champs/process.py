import pandas as pd
import numpy as np

def assign_atoms_index(df_idx, molecule):
    se_0 = df_idx.loc[molecule]['atom_index_0']
    se_1 = df_idx.loc[molecule]['atom_index_1']
    if type(se_0) == np.int64:
        se_0 = pd.Series(se_0)
    if type(se_1) == np.int64:
        se_1 = pd.Series(se_1)
    assign_idx = pd.concat([se_0, se_1]).unique()
    assign_idx.sort()
    return assign_idx

def get_dist_matrix(df_structures_idx, molecule):
    df_temp = df_structures_idx.loc[molecule]
    locs = df_temp[['x','y','z']].values
    num_atoms = len(locs)
    loc_tile = np.tile(locs.T, (num_atoms,1,1))
    dist_mat = np.linalg.norm(loc_tile - loc_tile.T, axis=1)
    return dist_mat

def get_angle(df_structures_idx, mol):
    locs = df_structures_idx.loc[mol][['x','y','z']].values # ( ,3)
    mat = get_dist_matrix(df_structures_idx, mol)
    sorted_locs = locs[np.argsort(mat)]
    origin = sorted_locs[:, 0]
    nearest = sorted_locs[:, 1]
    second = sorted_locs[:, 2:]
    base = nearest - origin
    out_mat = np.zeros((0, len(mat)))
    for i in range(len(mat)-2):
        sec_vec = second[:,i,:] - origin
        out = (base * sec_vec).sum(axis=1) / np.linalg.norm(sec_vec, axis=1) / np.linalg.norm(base, axis=1)
        out_mat = np.vstack([out_mat, out])
    left = np.ones((len(mat), 1))
    return np.hstack([left, out_mat.T])

def get_orientation(df_structures_idx, mol):
    locs = df_structures_idx.loc[mol][['x','y','z']].values # ( ,3)
    mat = get_dist_matrix(df_structures_idx, mol)
    sorted_locs = locs[np.argsort(mat)]    
    
    origin = sorted_locs[:, 0]
    nearest = sorted_locs[:, 1]
    second = sorted_locs[:, 2]
    try:
        third = sorted_locs[:, 3:]
    except:
        return np.ones(len(sorted_locs))
    
    base = nearest - origin
    sec_vec = second - origin
    out_mat = np.zeros((0, len(mat)))
    left = np.ones((len(mat), 1))
    for i in range(len(mat)-3):
        thi_vec = third[:,i,:] - origin
        proj_1 = sec_vec - base * np.tile(np.linalg.norm(sec_vec, axis=1), (3,1)).T / np.tile(np.linalg.norm(base, axis=1), (3,1)).T
        proj_2 = thi_vec - base * np.tile(np.linalg.norm(thi_vec, axis=1), (3,1)).T / np.tile(np.linalg.norm(base, axis=1), (3,1)).T
        out = (proj_1*proj_2).sum(axis=1) / np.linalg.norm(proj_1, axis=1) / np.linalg.norm(proj_2, axis=1)
        out_mat = np.vstack([out_mat,out])
    return np.hstack([left, (np.hstack([left, out_mat.T]))])

def n_cos(df_structures_idx, mol, thres=1.65):
    dist_mat = get_dist_matrix(df_structures_idx, mol)
    df_temp = df_structures_idx.loc[mol]
    num_atoms = df_temp.shape[0]
    n_idx = df_temp[df_temp['atom'] == 'N']['atom_index'].values
    n_cos2 = []
    n_cos3 = []    

    for i in n_idx:
        dist_argsort = np.argsort(dist_mat[i])
        near_1_idx = dist_argsort[1]
        near_2_idx = dist_argsort[2]
        
        dist_1 = dist_mat[i][near_1_idx]
        dist_2 = dist_mat[i][near_2_idx]
        
        if dist_2 > thres:
            n_cos2.append(-1)
            n_cos3.append(1)
            continue
        else:
            origin_loc = df_temp[df_temp['atom_index'] == i][['x', 'y', 'z']].values[0]
            near_1_loc = df_temp[df_temp['atom_index'] == near_1_idx][['x', 'y', 'z']].values[0]
            near_2_loc = df_temp[df_temp['atom_index'] == near_2_idx][['x', 'y', 'z']].values[0]
            vec_01 = near_1_loc - origin_loc
            vec_02 = near_2_loc - origin_loc
            cos_12 = np.dot(vec_01, vec_02) / dist_1 / dist_2
            n_cos2.append(cos_12)
            try:
                near_3_idx = dist_argsort[3]
                near_3_loc = df_temp[df_temp['atom_index'] == near_3_idx][['x', 'y', 'z']].values[0]
                vec_03 = near_3_loc - origin_loc

                if  dist_mat[i][near_3_idx] < thres:
                    vec_012 = vec_01 / dist_1 + vec_02/ dist_2
                    cos_123 = np.dot(vec_012, vec_03) / np.linalg.norm(vec_012) /dist_mat[i][near_3_idx]
                    n_cos3.append(cos_123)
                else:
                    n_cos3.append(1)
            except:
                 n_cos3.append(1)

    se_n_cos2 = pd.Series(n_cos2, name='cos2')
    se_n_cos3 = pd.Series(n_cos3, name='cos3')
    
    se_n_idx = pd.Series(n_idx, name='atom_index')
    df_bond = pd.concat([se_n_idx, se_n_cos2, se_n_cos3], axis=1)

    df_temp2 = pd.merge(df_temp[['atom', 'atom_index']], df_bond, on='atom_index', how='outer').fillna(1)
    df_temp2['molecule_name'] = mol
    return df_temp2

def c_cos(df_structures_idx, mol, thres=1.65):
    dist_mat = get_dist_matrix(df_structures_idx, mol)
    df_temp = df_structures_idx.loc[mol]
    num_atoms = df_temp.shape[0]

    c_idx = df_temp[df_temp['atom'] == 'C']['atom_index'].values

    c_cos2 = []
    c_cos3 = []

    for i in c_idx:
        dist_argsort = np.argsort(dist_mat[i])

        near_1_idx = dist_argsort[1]
        near_2_idx = dist_argsort[2]

        origin_loc = df_temp[df_temp['atom_index'] == i][['x', 'y', 'z']].values[0]
        near_1_loc = df_temp[df_temp['atom_index'] == near_1_idx][['x', 'y', 'z']].values[0]
        near_2_loc = df_temp[df_temp['atom_index'] == near_2_idx][['x', 'y', 'z']].values[0]

        vec_01 = near_1_loc - origin_loc
        vec_02 = near_2_loc - origin_loc
        
        cos_12 = np.dot(vec_01, vec_02) / dist_mat[i][near_1_idx] / dist_mat[i][near_2_idx] 
        
        c_cos2.append(cos_12)

        try:
            near_3_idx = dist_argsort[3]
            near_3_loc = df_temp[df_temp['atom_index'] == near_3_idx][['x', 'y', 'z']].values[0]
            vec_03 = near_3_loc - origin_loc
            
            if  dist_mat[i][near_3_idx] < thres:
                vec_012 = vec_01 / dist_mat[i][near_1_idx] + vec_02/ dist_mat[i][near_2_idx]
                cos_123 = np.dot(vec_012, vec_03) / np.linalg.norm(vec_012) /dist_mat[i][near_3_idx]
                c_cos3.append(cos_123)
            else:
                c_cos3.append(1)
        except:
             c_cos3.append(1)
            
    se_c_cos2 = pd.Series(c_cos2, name='cos2')
    se_c_cos3 = pd.Series(c_cos3, name='cos3')
    
    se_c_idx = pd.Series(c_idx, name='atom_index')
    df_bond = pd.concat([se_c_idx, se_c_cos2, se_c_cos3], axis=1)

    df_temp2 = pd.merge(df_temp[['atom', 'atom_index']], df_bond, on='atom_index', how='outer').fillna(1)
    df_temp2['molecule_name'] = mol
    return df_temp2

def get_cos(df_structures_idx, mol, thres=1.7):
    c_cos_temp = c_cos(df_structures_idx, mol, thres)[['cos2', 'cos3']].values
    n_cos_temp = n_cos(df_structures_idx, mol, thres)[['cos2', 'cos3']].values
    cos_mat = c_cos_temp + n_cos_temp
    out = np.where(cos_mat > 1, 1, cos_mat)
    return out

def get_pickup(df_idx, df_structures_idx, molecule, num_pickup=5, atoms=['H', 'C', 'N', 'O', 'F']):
    num_feature = 5 #direction, angle, orientation, bond_cos1, bond_cos2
    pickup_dist_matrix = np.zeros([0, len(atoms)*num_pickup*num_feature])
    assigned_idxs = assign_atoms_index(df_idx, molecule) # [0, 1, 2, 3, 4, 5, 6] -> [1, 2, 3, 4, 5, 6]
    dist_mat = get_dist_matrix(df_structures_idx, molecule)
    bond_cos = get_cos(df_structures_idx, molecule)
    angles = get_angle(df_structures_idx, molecule) # (num_atom, num_atom-2)
    orientations = get_orientation(df_structures_idx, molecule) # (num_atom, num_atom-2)

    for idx in assigned_idxs: # [1, 2, 3, 4, 5, 6] -> [2]
        df_temp = df_structures_idx.loc[molecule]
        locs = df_temp[['x','y','z']].values

        dist_arr = dist_mat[idx] # (7, 7) -> (7, )

        atoms_mole = df_structures_idx.loc[molecule]['atom'].values # ['O', 'C', 'C', 'N', 'H', 'H', 'H']
        atoms_mole_idx = df_structures_idx.loc[molecule]['atom_index'].values # [0, 1, 2, 3, 4, 5, 6]

        mask_atoms_mole_idx = atoms_mole_idx != idx # [ True,  True, False,  True,  True,  True,  True]
        masked_atoms = atoms_mole[mask_atoms_mole_idx] # ['O', 'C', 'N', 'H', 'H', 'H']
        masked_atoms_idx = atoms_mole_idx[mask_atoms_mole_idx]  # [0, 1, 3, 4, 5, 6]
        masked_dist_arr = dist_arr[mask_atoms_mole_idx]  # [ 5.48387003, 2.15181049, 1.33269675, 10.0578779, 4.34733927, 4.34727838]
        masked_angles = angles[masked_atoms_idx]
        masked_orientations = orientations[masked_atoms_idx]
        masked_bond_cos = bond_cos[masked_atoms_idx]

        sorting_idx = np.argsort(masked_dist_arr) # [2, 1, 5, 4, 0, 3]
        sorted_atoms_idx = masked_atoms_idx[sorting_idx] # [3, 1, 6, 5, 0, 4]
        sorted_atoms = masked_atoms[sorting_idx] # ['N', 'C', 'H', 'H', 'O', 'H']
        sorted_dist_arr = 1/masked_dist_arr[sorting_idx] #[0.75035825,0.46472494,0.23002898,0.23002576,0.18235297,0.09942455]

        sorted_angles = angles[idx]
        sorted_orientations = orientations[idx]

        sorted_bond_cos = masked_bond_cos[sorting_idx]

        target_matrix = np.zeros([len(atoms), num_pickup*num_feature])
        for a, atom in enumerate(atoms):
            pickup_atom = sorted_atoms == atom # [False, False,  True,  True, False,  True]
            pickup_dist = sorted_dist_arr[pickup_atom] # [0.23002898, 0.23002576, 0.09942455]
            pickup_angles = sorted_angles[pickup_atom]
            pickup_orientations = sorted_orientations[pickup_atom]
            pickup_bond_cos = sorted_bond_cos[pickup_atom]
            
            num_atom = len(pickup_dist)
            if num_atom > num_pickup:
                target_matrix[a, :num_pickup] = pickup_dist[:num_pickup]
                target_matrix[a, num_pickup:num_pickup*2] = pickup_angles[:num_pickup]
                target_matrix[a, num_pickup*2:num_pickup*3] = pickup_orientations[:num_pickup]
                target_matrix[a, num_pickup*3:num_pickup*4] = pickup_bond_cos[:num_pickup, 0]
                target_matrix[a, num_pickup*4:num_pickup*5] = pickup_bond_cos[:num_pickup, 1]
            else:
                target_matrix[a, :num_atom] = pickup_dist
                target_matrix[a, num_pickup:num_pickup+num_atom] = pickup_angles
                target_matrix[a, num_pickup+num_atom:num_pickup*2] = 1
                target_matrix[a, num_pickup*2:num_pickup*2+num_atom] = pickup_orientations
                target_matrix[a, num_pickup*2+num_atom:num_pickup*3] = 1
                target_matrix[a, num_pickup*3:num_pickup*3+num_atom] = pickup_bond_cos[:,0]
                target_matrix[a, num_pickup*3+num_atom:num_pickup*4] = 1
                target_matrix[a, num_pickup*4:num_pickup*4+num_atom] = pickup_bond_cos[:,1]
                target_matrix[a, num_pickup*4+num_atom:-2] = 1

        dist_ang_ori_bond =  target_matrix.reshape(-1)
        # dist_ang_ori_bond_bond = np.hstack([dist_ang_ori_bond,  bond_cos[idx,0],  bond_cos[idx,1]])
        pickup_dist_matrix = np.vstack([pickup_dist_matrix, dist_ang_ori_bond])

    return pickup_dist_matrix #(num_assigned_atoms, num_atoms*num_pickup*5 + 2)

def merge_atom(df, df_distance):
    df_merge_0 = pd.merge(df, df_distance, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])
    df_merge_0_1 = pd.merge(df_merge_0, df_distance, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])
    del df_merge_0_1['atom_index_x'], df_merge_0_1['atom_index_y']
    return df_merge_0_1

def get_cos_3J(df_structures_idx, molecule_name, atom_idx_list):
    pos_list = []
    df_st = df_structures_idx.loc[molecule_name]

    for idx in atom_idx_list:
        pos = df_st.query('atom_index == {}'.format(idx))[['x', 'y', 'z']].values
        pos_list.append(pos)

    v01 = pos_list[1] - pos_list[0]
    v12 = pos_list[2] - pos_list[1]
    v23 = pos_list[3] - pos_list[2]

    v01_12 = v01 - ((np.dot(v01, v12.T) / np.linalg.norm(v12) **2 ) * v12)[0]
    v23_12 = v23 - ((np.dot(v23, v12.T) / np.linalg.norm(v12) **2 ) * v12)[0]

    if np.linalg.norm(v23_12)*np.linalg.norm(v01_12) == 0:
        return np.array([1, 1, np.linalg.norm(v12)])
    
    cos = (np.dot(v01_12, v23_12.T) / np.linalg.norm(v01_12) / np.linalg.norm(v23_12))[0]
    
    return np.array([cos[0], cos[0]**2-1, np.linalg.norm(v12)])

def gen_3JHC_list(df_idx, df_structures_idx, molecule_name):
    pairs_list = []
    df_tr = df_idx.loc[molecule_name]
    df_st = df_structures_idx.loc[molecule_name]

    if type(df_tr) == pd.Series:
        return []

    pairs_3J = df_tr.query('type == "{}"'.format('3JHC'))[['atom_index_0','atom_index_1','id']].values
    dist_matrix = get_dist_matrix(df_structures_idx, molecule_name)

    for p3 in pairs_3J:
        atom_idx_0 = p3[0] 
        con_id = p3[2] 

        dist_arr = dist_matrix[atom_idx_0] 
        mask = dist_arr != 0
        dist_arr_excl_0 = dist_arr[mask]
        masked_idx = df_st['atom_index'].values[mask]
        atom_idx_1 = masked_idx[np.argsort(dist_arr_excl_0)[0]]

        atom_idx_3 = p3[1]
        dist_arr = dist_matrix[atom_idx_3]
        candidate_atom_idx_2 = np.arange(len(dist_arr))[(dist_arr > 1.1) * (dist_arr < 1.65)]
        atom_idx_2 = candidate_atom_idx_2[np.argsort(dist_matrix[atom_idx_1][candidate_atom_idx_2])[0]]
        pair = [atom_idx_0, atom_idx_1, atom_idx_2, atom_idx_3, con_id]
        pairs_list.append(pair)

    return pairs_list

def gen_3JHN_list(df_idx, df_structures_idx, molecule_name):
    pairs_list = []
    df_tr = df_idx.loc[molecule_name]
    df_st = df_structures_idx.loc[molecule_name]

    if type(df_tr) == pd.Series:
        return []

    pairs_3J = df_tr.query('type == "{}"'.format('3JHN'))[['atom_index_0','atom_index_1','id']].values
    dist_matrix = get_dist_matrix(df_structures_idx, molecule_name)

    for p3 in pairs_3J:
        atom_idx_0 = p3[0] 
        con_id = p3[2] 

        dist_arr = dist_matrix[atom_idx_0] 
        mask = dist_arr != 0
        dist_arr_excl_0 = dist_arr[mask]
        masked_idx = df_st['atom_index'].values[mask]
        atom_idx_1 = masked_idx[np.argsort(dist_arr_excl_0)[0]]

        atom_idx_3 = p3[1]
        dist_arr = dist_matrix[atom_idx_3]
        candidate_atom_idx_2 = np.arange(len(dist_arr))[(dist_arr > 1.1) * (dist_arr < 1.65)]
        atom_idx_2 = candidate_atom_idx_2[np.argsort(dist_matrix[atom_idx_1][candidate_atom_idx_2])[0]]
        pair = [atom_idx_0, atom_idx_1, atom_idx_2, atom_idx_3, con_id]
        pairs_list.append(pair)

    return pairs_list

def gen_3JHH_list(df_idx, df_structures_idx, molecule_name):
    pairs_list = []
    df_tr = df_idx.loc[molecule_name]
    df_st = df_structures_idx.loc[molecule_name]
    if type(df_tr) == pd.Series:
        return []
    
    pairs_3J = df_tr.query('type == "{}"'.format("3JHH"))[['atom_index_0','atom_index_1','id']].values
    dist_matrix = get_dist_matrix(df_structures_idx, molecule_name)

    for p3 in pairs_3J:
        atom_idx_0 = p3[0]
        con_id = p3[2]

        dist_arr = dist_matrix[atom_idx_0]
        mask = dist_arr != 0
        dist_arr_excl_0 = dist_arr[mask]
        masked_idx = df_st['atom_index'].values[mask]
        atom_idx_1 = masked_idx[np.argsort(dist_arr_excl_0)[0]]

        atom_idx_3 = p3[1]
        dist_arr = dist_matrix[atom_idx_3]
        mask = dist_arr != 0
        dist_arr_excl_0 = dist_arr[mask]
        masked_idx = df_st['atom_index'].values[mask]
        atom_idx_2 = masked_idx[np.argsort(dist_arr_excl_0)[0]]        
        
        pair = [atom_idx_0, atom_idx_1, atom_idx_2, atom_idx_3, con_id]
        pairs_list.append(pair)
        
    return pairs_list

def gen_pairs_list(df_idx, df_structures_idx, molecule_name, type_3J):
    if type_3J == '3JHH':
        return gen_3JHH_list(df_idx, df_structures_idx, molecule_name)
    elif type_3J == '3JHC':
        return gen_3JHC_list(df_idx, df_structures_idx, molecule_name)
    elif type_3J == '3JHN':
        return gen_3JHN_list(df_idx, df_structures_idx, molecule_name)
    else:
        return []

def type_score(y_val, y_pred):
    return np.log(sum(np.abs(y_val- y_pred)) / len(y_val))

def pickup_bond_value_dist(df_mol_idx_0, dist_arr, bond, target_col):
    df_mol_idx_b = df_mol_idx_0.query('type == "{}"'.format(bond))
    atoms_b = df_mol_idx_b['atom_index_1'].values
    dist_b = dist_arr[atoms_b]
    predicts_b = df_mol_idx_b[target_col]        
    sorting_b = np.argsort(dist_b)
    return predicts_b.values[sorting_b], 1/dist_arr[atoms_b][sorting_b]

def gen_second_data(df_idx, df_structures_idx, m, target_bond='1JHC', target_col='scalar_coupling_constant'):
    try:
        type(df_idx.loc[m]) == pd.Series
    except:
        print("series exception:", m)
        return
    dist_mat = get_dist_matrix(df_structures_idx, m)    

    df_mol = df_idx.loc[m]
    con_id = df_mol.query('type == "{}"'.format(target_bond))['id'].values
    df_mol_idx = df_mol.set_index('id')

    bonds = ['1JHC', '1JHN', '2JHH', '2JHC', '2JHN', '3JHH', '3JHC', '3JHN']

    predict_01 = np.zeros([len(con_id), 2])
    features_0 = np.zeros([len(con_id), len(bonds)*3*2])
    features_1 = np.zeros([len(con_id), len(bonds)*3*2])

    for i, idx in enumerate(con_id):
        focus_0 = df_mol_idx.loc[idx]['atom_index_0']
        focus_1 = df_mol_idx.loc[idx]['atom_index_1']

        dist_arr = dist_mat[focus_0]
        predict_01[i, 0] = df_mol_idx.loc[idx][target_col]
        predict_01[i, 1] = 1/dist_arr[focus_1]

        df_mol_idx_0 = df_mol_idx.loc[df_mol_idx.index != idx].query('atom_index_0 == {}'.format(focus_0))
        for j, b in enumerate(bonds):
            predicts, inv_dist = pickup_bond_value_dist(df_mol_idx_0, dist_arr, b, target_col)
            if len(predicts) > 3:            
                features_0[i, j*3:(j+1)*3] = predicts[:3]
                features_0[i, (j+1)*3:(j+2)*3] = inv_dist[:3]
            else:
                features_0[i, j*3:j*3+len(predicts)] = predicts
                features_0[i, (j+1)*3:(j+1)*3+len(inv_dist)] = inv_dist
        
        df_mol_idx_1 = df_mol_idx.loc[df_mol_idx.index != idx].query('atom_index_1 == {}'.format(focus_1))
        for j, b in enumerate(bonds):
            predicts, inv_dist = pickup_bond_value_dist(df_mol_idx_1, dist_arr, b, target_col)
            if len(predicts) > 3:            
                features_1[i, j*3:(j+1)*3] = predicts[:3]
                features_1[i, (j+1)*3:(j+2)*3] = inv_dist[:3]
            else:
                features_1[i, j*3:j*3+len(predicts)] = predicts
                features_1[i, (j+1)*3:(j+1)*3+len(inv_dist)] = inv_dist
                
    features = np.hstack([predict_01, features_0, features_1])
    df_out = pd.DataFrame(features)
    df_out['id'] = con_id
    return df_out

def c_neighbor(df_structures_idx, mol, thres=1.65):
    dist_mat = get_dist_matrix(df_structures_idx, mol)
    atom_arr = df_structures_idx.loc[mol]["atom"].values
    df_temp = df_structures_idx.loc[mol]
    num_atoms = df_temp.shape[0]

    c_idx = df_temp[df_temp['atom'] == 'C']['atom_index'].values

    neighbor_idx = {}
    neighbor_atoms = {}
#     neighbor_dist = {}

    for i in c_idx:
        dist_argsort = np.argsort(dist_mat[i])
        near_1_idx = dist_argsort[1]
        near_2_idx = dist_argsort[2]
        try:
            near_3_idx = dist_argsort[3]
            if  dist_mat[i][near_3_idx] < thres:
                try:
                    near_4_idx = dist_argsort[4]
                    if  dist_mat[i][near_4_idx] < thres:
                        neighbor_idx[i] = np.array([near_1_idx, near_2_idx, near_3_idx, near_4_idx])
                        neighbor_atoms[i] = atom_arr[neighbor_idx[i]]
#                         neighbor_dist[i] = dist_mat[i][[neighbor_idx[i]]]
                    else:
                        neighbor_idx[i] = np.array([near_1_idx, near_2_idx, near_3_idx, -1])
                        atoms = atom_arr[np.array([neighbor_idx[i][:3]])][0]
                        neighbor_atoms[i] =  np.hstack([atoms, "X"])
#                         dists = dist_mat[i][neighbor_idx[i][:3]][0]
#                         neighbor_dist[i] = np.hstack([dists, 0.0])
                except:
                    neighbor_idx[i] = np.array([near_1_idx, near_2_idx, near_3_idx, -1])
                    atoms = atom_arr[neighbor_idx[i][:3]][0]
                    neighbor_atoms[i] =  np.hstack([atoms, "X"])
#                     dists = dist_mat[i][neighbor_idx[i][:3]][0]
#                     neighbor_dist[i] = np.hstack([dists, 0.0])
            else:
                neighbor_idx[i] = np.array([near_1_idx, near_2_idx, -1, -1])
                atoms = atom_arr[neighbor_idx[i][:2]][0]
                neighbor_atoms[i] =  np.hstack([atoms, "X", "X"])
#                 dists = dist_mat[i][neighbor_idx[i][:2]][0]
#                 neighbor_dist[i] = np.hstack([dists, 0.0, 0.0])
        except:
            neighbor_idx[i] = np.array([near_1_idx, near_2_idx, -1, -1])
            atoms = atom_arr[neighbor_idx[i][:2]][0]
            neighbor_atoms[i] =  np.hstack([atoms, "X", "X"])
#             dists = dist_mat[i][neighbor_idx[i][:2]][0]
#             neighbor_dist[i] = np.hstack([dists, 0.0, 0.0])

    return neighbor_idx, neighbor_atoms#, neighbor_dist

def pickup_neighbors(i, neighbors):
    return neighbors[i]
def gen_df_1JHC(df_idx, df_structures_idx, m):
#     neighbors_idx, neighbors_atom, neighbors_dist = c_neighbor(df_structures_idx, m)
    neighbors_idx, neighbors_atom = c_neighbor(df_structures_idx, m)
    
    df_idx_temp = df_idx.loc[m]
    if type(df_idx_temp) == pd.Series:
        return
    
    df_1JHC_temp = df_idx_temp.query('type == "1JHC"')

    atom_arrays = df_1JHC_temp['atom_index_1'].apply(pickup_neighbors, neighbors=neighbors_atom).values
    idx_arrays = df_1JHC_temp['atom_index_1'].apply(pickup_neighbors, neighbors=neighbors_idx).values
    # dist_arrays = df_1JHC_temp['atom_index_1'].apply(pickup_neighbors, neighbors=neighbors_dist)

    num = len(idx_arrays)
    idx_arr = np.zeros([0,4])
    atom_arr = np.zeros([0,4])
    # dist_arr = np.zeros([0,4])
    for i in range(num):
        idx_arr = np.vstack([idx_arr, idx_arrays[i]])
        atom_arr = np.vstack([atom_arr, atom_arrays[i]])
    #     dist_arr = np.vstack([dist_arr, dist_arrays[i]])

    for j in range(4):
        df_1JHC_temp.loc[:]['neig_idx_{}'.format(j)] = idx_arr[:,j]
        df_1JHC_temp.loc[:]['neig_atom_{}'.format(j)] = atom_arr[:,j]
    #     df_1JHC_temp['neig_dist_{}'.format(j)] = dist_arr[:,j]

    return df_1JHC_temp