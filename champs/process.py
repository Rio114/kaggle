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
                target_matrix[a, num_pickup*4:] = pickup_bond_cos[:num_pickup, 1]
            else:
                target_matrix[a, :num_atom] = pickup_dist
                target_matrix[a, num_pickup:num_pickup+num_atom] = pickup_angles
                target_matrix[a, num_pickup+num_atom:num_pickup*2] = 1
                target_matrix[a, num_pickup*2:num_pickup*2+num_atom] = pickup_orientations
                target_matrix[a, num_pickup*2+num_atom:num_pickup*3] = 1
                target_matrix[a, num_pickup*3:num_pickup*3+num_atom] = pickup_bond_cos[:,0]
                target_matrix[a, num_pickup*3+num_atom:num_pickup*4] = 1
                target_matrix[a, num_pickup*4:num_pickup*4+num_atom] = pickup_bond_cos[:,1]
                target_matrix[a, num_pickup*4+num_atom:] = 1


        pickup_dist_matrix = np.vstack([pickup_dist_matrix, target_matrix.reshape(-1)])
    return pickup_dist_matrix #(num_atoms, num_pickup*5)

def merge_atom(df, df_distance):
    df_merge_0 = pd.merge(df, df_distance, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'])
    df_merge_0_1 = pd.merge(df_merge_0, df_distance, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'])
    del df_merge_0_1['atom_index_x'], df_merge_0_1['atom_index_y']
    return df_merge_0_1
