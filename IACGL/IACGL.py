import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, roc_curve # For AUC
from scipy.io import loadmat # For loading .mat files
import math
import time
import os 
import copy 
import scipy.sparse
import yaml


# Placeholder for Adam optimizer
def adam(params, grad, cache=None, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Simple NumPy implementation of Adam optimizer."""
    if cache is None:
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        t = 0
        cache = (m, v, t)
    
    m, v, t = cache
    t += 1
    
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    
    params_update = params - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    
    cache = (m, v, t)
    return params_update, cache

def generate_outliers(mat_file_path, a, b, c):
    """
    Generates multi-view data with specified outlier types from binary data.

    Loads data from a .mat file containing multi-view binary data (0/1),
    introduces three types of outliers (attribute, class, class-attribute)
    with given proportions, and returns the modified data and outlier labels.
    Attribute and class-attribute outlier generation respects the original
    density of 1s in each view.

    Args:
        mat_file_path (str): Path to the .mat file. The file must contain:
            'X': A structure loadable as a list or array of NumPy arrays/SciPy
                 sparse matrices. Each element X[v] should represent an N x Dv
                 binary (0/1) matrix for view 'v', where N is the number of
                 instances and Dv is the dimensionality of view 'v'.
            'Y': An N x 1 or 1 x N array containing the class labels for
                 the N instances.
        a (float): Proportion of attribute outliers (0 <= a <= 1).
        b (float): Proportion of class outliers (0 <= b <= 1).
        c (float): Proportion of class-attribute outliers (0 <= c <= 1).

    Returns:
        tuple: A tuple containing:
            - X_outlier (list): List of NumPy arrays (float64, but values are 0.0/1.0).
                                Contains the data with introduced outliers.
            - Y_outlier (np.ndarray): N x 1 integer array. Contains outlier labels:
                                      0 for normal instances,
                                      1 for any type of outlier instance.

    Raises:
        ValueError: If input parameters are invalid, file not found, .mat file
                    is missing 'X' or 'Y', data dimensions are inconsistent,
                    or total outlier proportion exceeds 1.
        TypeError: If data in X is not numeric or boolean.
        ImportError: If required libraries (scipy) are not installed.

    Example:
        >>> # Assume 'my_binary_multiview_data.mat' exists with 'X' and 'Y'
        >>> mat_path = 'my_binary_multiview_data.mat'
        >>> a_prop = 0.05  # 5% attribute outliers
        >>> b_prop = 0.05  # 5% class outliers
        >>> c_prop = 0.05  # 5% class-attribute outliers
        >>> try:
        >>>     X_modified, Y_labels = generate_outliers(mat_path, a_prop, b_prop, c_prop)
        >>>     print(f"Generated {len(X_modified)} views.")
        >>>     print(f"Outlier labels shape: {Y_labels.shape}")
        >>>     print(f"Number of outliers indicated: {np.sum(Y_labels)}")
        >>> except Exception as e:
        >>>     print(f"An error occurred: {e}")
    """

    # --- Input Validation ---
    if not isinstance(mat_file_path, str):
        raise ValueError("mat_file_path must be a string.")
    if not os.path.isfile(mat_file_path):
        raise ValueError(f"MAT file not found: {mat_file_path}")

    if not isinstance(a, (int, float)) or not (0 <= a <= 1):
        raise ValueError("Proportion a must be a numeric value between 0 and 1.")
    if not isinstance(b, (int, float)) or not (0 <= b <= 1):
        raise ValueError("Proportion b must be a numeric value between 0 and 1.")
    if not isinstance(c, (int, float)) or not (0 <= c <= 1):
        raise ValueError("Proportion c must be a numeric value between 0 and 1.")
    if (a + b + c) > 1:
        raise ValueError("The sum of proportions (a + b + c) cannot exceed 1.")

    # --- Load Data ---
    print(f"Loading data from: {mat_file_path}")
    try:
        data = scipy.io.loadmat(mat_file_path)
        if 'X' not in data or 'Y' not in data:
            raise ValueError("The .mat file must contain variables named 'X' and 'Y'.")

        # --- Handle different X structures from loadmat ---
        X_raw = data['X']
        X_original = [] # This will hold the list of view matrices

        # Try interpreting X_raw as a list/array of arrays
        # Case 1: MATLAB {1xV cell} -> numpy array(1, V, dtype=object)
        if X_raw.ndim == 2 and X_raw.shape[0] == 1 and X_raw.dtype == 'object':
            X_original = [item for item in X_raw[0, :]]
        # Case 2: MATLAB {Vx1 cell} -> numpy array(V, 1, dtype=object)
        elif X_raw.ndim == 2 and X_raw.shape[1] == 1 and X_raw.dtype == 'object':
             X_original = [item[0] for item in X_raw[:, 0]]
        # Case 3: Already a sequence (less common from loadmat but possible)
        elif isinstance(X_raw, (list, tuple)):
             X_original = list(X_raw)
        # Fallback: Try treating as a single view if it's just a numeric/sparse array
        elif isinstance(X_raw, (np.ndarray, scipy.sparse.spmatrix)) and X_raw.dtype != 'object':
             warnings.warn("Input 'X' appears to be a single matrix, not a cell array/list of views. Treating as one view.", UserWarning)
             X_original = [X_raw]
        else:
             raise ValueError("Could not reliably interpret 'X' from .mat file as a list/array of view matrices.")

        # Ensure all views are NumPy arrays or supported sparse matrices
        temp_X = []
        for i, view in enumerate(X_original):
             if isinstance(view, scipy.sparse.spmatrix):
                 # Keep sparse for density calculation efficiency if needed
                 temp_X.append(view)
             elif isinstance(view, np.ndarray):
                 temp_X.append(view)
             else:
                # Try converting other types if possible, e.g., lists of lists
                try:
                    temp_X.append(np.array(view))
                except Exception as e:
                    raise TypeError(f"View {i} in 'X' is of unsupported type {type(view)}. Could not convert to NumPy array.") from e
        X_original = temp_X

        Y_raw = data['Y']
        if not isinstance(Y_raw, np.ndarray):
            # Should be loaded as ndarray, but handle potential edge cases
            Y_raw = np.array(Y_raw)
        # Ensure Y is a column vector (N, 1)
        Y = Y_raw.reshape(-1, 1)

    except ImportError:
         raise ImportError("Loading .mat files requires scipy. Please install it (`pip install scipy`).")
    except Exception as e:
        # Catch potential errors during loading or initial processing
        raise RuntimeError(f"Failed to load or process data from {mat_file_path}. Error: {e}") from e


    # --- Data Validation and Dimensions ---
    if not X_original: # Check if list is empty
        raise ValueError("X cell array/list has no views.")

    V = len(X_original) # Number of views

    try:
        N = X_original[0].shape[0] # Number of instances from first view
    except IndexError:
         raise ValueError("First view in X appears to be empty or invalid.")

    if Y.shape[0] != N or Y.shape[1] != 1:
         raise ValueError(f"Y must have shape ({N}, 1), but got {Y.shape}.")

    # Verify instance count consistency and data type across views
    for v_idx, view_data in enumerate(X_original):
        if view_data.shape[0] != N:
            raise ValueError(f"Number of instances (rows) must be consistent. View {v_idx} has {view_data.shape[0]} rows, expected {N}.")
        # Check if data is numeric or boolean
        is_numeric = np.issubdtype(view_data.dtype, np.number)
        is_boolean = np.issubdtype(view_data.dtype, np.bool_)
        # Allow sparse matrix check (dtypes are usually numeric)
        is_sparse_ok = isinstance(view_data, scipy.sparse.spmatrix) and np.issubdtype(view_data.dtype, np.number)

        if not (is_numeric or is_boolean or is_sparse_ok):
             raise TypeError(f"Data in X view {v_idx} must be numeric or boolean, but found {view_data.dtype}.")

        # Basic check if data looks binary (contains only 0s and 1s)
        if isinstance(view_data, scipy.sparse.spmatrix):
            # For sparse, check unique data values efficiently
            unique_vals = np.unique(view_data.data)
        else:
            unique_vals = np.unique(view_data)
        # Allow empty views (unique_vals would be empty)? No, should have N rows.
        # Check if all unique values present are either 0 or 1
        if not np.all(np.isin(unique_vals, [0, 1])):
             warnings.warn(f'generate_outliers:NonBinaryData - View {v_idx} does not appear to be strictly binary (0/1). Values found: {unique_vals}. Proceeding, but outlier generation assumes binary density.', UserWarning)

    print(f'Data loaded: {N} instances, {V} views.')

    # --- Calculate Original Densities (Proportion of 1s) for Each View ---
    print('Calculating original view densities...')
    original_densities = np.zeros(V)
    for v_idx, view_data_orig in enumerate(X_original):
        if isinstance(view_data_orig, scipy.sparse.spmatrix):
            num_ones = view_data_orig.nnz # Efficient way for sparse
        else:
            # Ensure boolean arrays are treated as 0/1
            num_ones = np.sum(view_data_orig.astype(float))

        total_elements = view_data_orig.shape[0] * view_data_orig.shape[1]

        if total_elements > 0:
            original_densities[v_idx] = num_ones / total_elements
        else:
            original_densities[v_idx] = 0.0 # Density is 0 for an empty view/matrix
        print(f'  View {v_idx} density: {original_densities[v_idx]:.4f}')


    # --- Calculate Number of Outliers ---
    num_attr_outliers = int(np.floor(N * a))
    num_class_outliers = int(np.floor(N * b))
    num_class_attr_outliers = int(np.floor(N * c))

    total_outliers_to_select = num_attr_outliers + num_class_outliers + num_class_attr_outliers

    print(f'Target outliers: Attribute={num_attr_outliers}, Class={num_class_outliers}, Class-Attribute={num_class_attr_outliers} (Total={total_outliers_to_select})')

    if total_outliers_to_select == 0:
        warnings.warn('generate_outliers:ZeroOutliers - Total number of outliers to generate is 0 based on proportions. Returning original data structure.', UserWarning)
        # Proceed, copy will happen, but no outliers added.
    if total_outliers_to_select > N:
        raise ValueError(f'Calculated total number of outliers ({total_outliers_to_select}) exceeds the number of instances ({N}). Adjust proportions.')

    # --- Select Non-Overlapping Indices for Outliers ---
    # Use 0-based indexing
    all_indices_shuffled = np.random.permutation(N)

    attr_idx = np.array([], dtype=int)
    class_idx = np.array([], dtype=int)
    class_attr_idx = np.array([], dtype=int)

    start = 0
    if num_attr_outliers > 0:
        attr_idx = all_indices_shuffled[start : start + num_attr_outliers]
        start += num_attr_outliers
    if num_class_outliers > 0:
        class_idx = all_indices_shuffled[start : start + num_class_outliers]
        start += num_class_outliers
    if num_class_attr_outliers > 0:
        class_attr_idx = all_indices_shuffled[start : start + num_class_attr_outliers]
        start += num_class_attr_outliers # Not strictly needed

    # Combine all selected outlier indices (ensure they are numpy arrays)
    all_outlier_indices = np.concatenate((attr_idx, class_idx, class_attr_idx)).astype(int)

    # --- Initialize Output Variables ---
    # Create a list of copies for modification, ensuring float type but values remain 0.0/1.0
    X_outlier = []
    for view in X_original:
        if isinstance(view, scipy.sparse.spmatrix):
            # Convert sparse to dense float for modification
            X_outlier.append(view.toarray().astype(float))
        else:
            # Copy dense array, ensure float
            X_outlier.append(view.copy().astype(float))

    Y_outlier = np.zeros((N, 1), dtype=int) # Initialize all instances as normal (0)

    # Mark all selected instances initially as outliers (1)
    if all_outlier_indices.size > 0:
        Y_outlier[all_outlier_indices] = 1

    # --- Generate Attribute Outliers ---
    if num_attr_outliers > 0:
        print(f'Generating {num_attr_outliers} attribute outliers (using original view density)...')
        for idx in attr_idx:
            for v_idx in range(V):
                try:
                    num_features = X_outlier[v_idx].shape[1]
                    density_v = original_densities[v_idx]
                    # Generate binary vector based on original view density
                    # Ensure output is float (0.0 or 1.0)
                    X_outlier[v_idx][idx, :] = (np.random.rand(num_features) < density_v).astype(float)
                except IndexError:
                     print(f"Warning: Could not access or modify X_outlier[{v_idx}][{idx}, :]. Skipping this instance/view for attribute outlier.")


    # --- Generate Class-Attribute Outliers ---
    if num_class_attr_outliers > 0:
        print(f'Generating {num_class_attr_outliers} class-attribute outliers (using original view density)...')
        num_views_to_modify_ca = int(np.floor(V / 2)) # Half views (floor)

        if num_views_to_modify_ca > 0:
            for idx in class_attr_idx:
                # Select random views (indices) for this instance
                if V > 0 : # Ensure V is positive before choosing
                    views_to_modify_indices = np.random.choice(V, min(num_views_to_modify_ca, V), replace=False)
                else:
                    views_to_modify_indices = []


                for v_idx in views_to_modify_indices:
                    try:
                        num_features = X_outlier[v_idx].shape[1]
                        density_v = original_densities[v_idx]
                        # Generate binary vector based on original view density
                        X_outlier[v_idx][idx, :] = (np.random.rand(num_features) < density_v).astype(float)
                    except IndexError:
                        print(f"Warning: Could not access or modify X_outlier[{v_idx}][{idx}, :]. Skipping this instance/view for class-attribute outlier.")

        else:
            print('Skipping class-attribute outlier generation as floor(V/2) = 0.')


    # --- Generate Class Outliers ---
    num_pairs_target = int(np.floor(num_class_outliers / 2)) # Target number of pairs to swap
    if num_class_outliers > 1 and num_pairs_target > 0:
        print(f'Generating class outliers (targeting {num_pairs_target} pairs by swapping data)...')

        num_views_to_swap = int(np.floor(V / 2)) # Half views (floor) to swap

        if num_views_to_swap > 0 and V > 0:
            shuffled_class_indices = np.random.permutation(class_idx) # Shuffle potential candidates
            paired_indices_flag = np.zeros(N, dtype=bool) # Track original indices already used
            pairs_formed = [] # Store the index pairs (tuples)
            Y_flat = Y.ravel() # Use flattened Y for easier label access

            # Try to form pairs with different classes
            for i in range(len(shuffled_class_indices)):
                if len(pairs_formed) >= num_pairs_target: # Stop if target reached
                    break

                idx1 = shuffled_class_indices[i]
                if paired_indices_flag[idx1]: # Skip if already paired
                    continue

                # Find a suitable partner (different class, not yet paired)
                for j in range(i + 1, len(shuffled_class_indices)):
                    idx2 = shuffled_class_indices[j]
                    # Check if not paired and classes are different
                    if not paired_indices_flag[idx2] and Y_flat[idx1] != Y_flat[idx2]:
                        # Found a pair
                        pairs_formed.append((idx1, idx2))
                        paired_indices_flag[idx1] = True
                        paired_indices_flag[idx2] = True
                        break # Partner found for idx1, move to next i

            print(f'Successfully formed {len(pairs_formed)} pairs for class outlier swaps.')

            # Perform the swaps for the formed pairs
            for p, (idx1, idx2) in enumerate(pairs_formed):
                 # Select random views (indices) for this pair
                 views_to_swap_indices = np.random.choice(V, min(num_views_to_swap, V), replace=False)

                 for v_idx in views_to_swap_indices:
                    try:
                        # Swap data rows in the selected view using .copy() to avoid view issues
                        temp_row = X_outlier[v_idx][idx1, :].copy()
                        X_outlier[v_idx][idx1, :] = X_outlier[v_idx][idx2, :].copy()
                        X_outlier[v_idx][idx2, :] = temp_row
                    except IndexError:
                         print(f"Warning: Could not access or swap X_outlier[{v_idx}] rows {idx1}/{idx2}. Skipping swap for this pair/view.")

        elif V == 0:
             print('Skipping class outlier generation as there are no views (V=0).')
        else: # num_views_to_swap == 0
            print('Skipping class outlier generation as floor(V/2) = 0.')

        # Note: Y_outlier for class_idx was already set to 1 earlier.
        # Instances intended for class outliers remain marked even if not paired/swapped.

    elif num_class_outliers <= 1:
         print(f'Skipping class outlier generation as fewer than 2 instances selected (num_class_outliers={num_class_outliers}).')
    elif num_pairs_target == 0: # Handles case when b is small but non-zero
         print('Skipping class outlier generation as target number of pairs is 0.')

    print('Outlier generation complete.')

    return X_outlier, Y_outlier

# --- Data Generation Function ---
def generate_multiview_outliers_py(mat_file_path, a, b, c):
    """
    Generates multi-view data with specified outlier types in Python. 
    (Docstring omitted for brevity)
    """
    # --- Input Validation ---
    if not isinstance(mat_file_path, str):
        raise ValueError('mat_file_path must be a string.')
    if not os.path.exists(mat_file_path):
        raise FileNotFoundError(f'MAT file not found: {mat_file_path}')
    
    for prop, name in [(a, 'a'), (b, 'b'), (c, 'c')]:
        if not isinstance(prop, (int, float)) or not (0 <= prop <= 1):
                 raise ValueError(f'Proportion {name} must be a numeric value between 0 and 1.')
                 
    if (a + b + c) > 1:
        raise ValueError('The sum of proportions (a + b + c) cannot exceed 1.')

    # --- Load Data ---
    print(f'Loading data from: {mat_file_path}')
    try:
        data = loadmat(mat_file_path)
        if 'X' not in data or 'Y' not in data:
            raise ValueError("The .mat file must contain variables named 'X' and 'Y'.")
        
        X_raw = data['X']
        if X_raw.ndim == 2 and X_raw.shape[0] == 1: 
            X = [np.array(arr, dtype=float) for arr in X_raw[0]]
        elif X_raw.ndim == 2 and X_raw.shape[1] == 1: 
            X = [np.array(arr, dtype=float) for arr in X_raw[:, 0]]
        elif isinstance(X_raw, list): 
             X = [np.array(arr, dtype=float) for arr in X_raw]
        else: 
             try:
                 X = [np.array(X_raw[i], dtype=float) for i in range(X_raw.size)]
             except Exception:
                  raise ValueError("Could not interpret 'X' from .mat file as a list of NumPy arrays.")

        Y = data['Y'].flatten().astype(int) 
        
    except Exception as e:
        raise IOError(f"Failed to load or process data from {mat_file_path}. Error: {e}")

    # --- Data Validation and Dimensions ---
    if not isinstance(X, list) or not X:
        raise ValueError("X must be loaded as a non-empty list of numpy arrays.")

    V = len(X) 
    if V == 0: raise ValueError("X list has no views.")
        
    try: N = X[0].shape[0] 
    except IndexError: raise ValueError("First view in X seems to be empty or not a 2D array.")
         
    if Y.shape[0] != N: raise ValueError(f"Y length ({Y.shape[0]}) must match number of instances in X[0] ({N}).")

    for v in range(V):
        if not isinstance(X[v], np.ndarray) or X[v].ndim != 2: raise ValueError(f"Element X[{v}] is not a 2D numpy array.")
        if X[v].shape[0] != N: raise ValueError(f"Number of instances in view {v} ({X[v].shape[0]}) does not match view 0 ({N}).")
        X[v] = X[v].astype(float) 

    print(f'Data loaded: {N} instances, {V} views.')

    # --- Calculate Number of Outliers ---
    num_attr_outliers = int(np.floor(N * a))
    num_class_outliers = int(np.floor(N * b)) 
    num_class_attr_outliers = int(np.floor(N * c))
    total_outliers_to_select = num_attr_outliers + num_class_outliers + num_class_attr_outliers
    print(f'Target outliers: Attribute={num_attr_outliers}, Class={num_class_outliers}, Class-Attribute={num_class_attr_outliers} (Total={total_outliers_to_select})')

    if total_outliers_to_select == 0: print('Warning: Total number of outliers to generate is 0.')
    elif total_outliers_to_select > N: raise ValueError(f'Calculated total outliers ({total_outliers_to_select}) > instances ({N}).')
       
    # --- Select Indices ---
    all_indices_shuffled = np.random.permutation(N)
    attr_idx, class_idx, class_attr_idx = np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    current_pos = 0
    if num_attr_outliers > 0: attr_idx = all_indices_shuffled[current_pos : current_pos + num_attr_outliers]; current_pos += num_attr_outliers
    if num_class_outliers > 0: class_idx = all_indices_shuffled[current_pos : current_pos + num_class_outliers]; current_pos += num_class_outliers
    if num_class_attr_outliers > 0: class_attr_idx = all_indices_shuffled[current_pos : current_pos + num_class_attr_outliers]; current_pos += num_class_attr_outliers
    all_outlier_indices = np.concatenate((attr_idx, class_idx, class_attr_idx)).astype(int)

    # --- Initialize Output ---
    X_outlier = copy.deepcopy(X) 
    Y_outlier = np.zeros(N, dtype=int) 
    if all_outlier_indices.size > 0: Y_outlier[all_outlier_indices] = 1

    # --- Normalize Data ---
    print('Normalizing data for each view to [0, 1]...')
    for v in range(V):
        view_data = X_outlier[v]
        min_vals, max_vals = np.min(view_data, axis=0), np.max(view_data, axis=0)
        range_vals = max_vals - min_vals
        zero_range_indices = (range_vals == 0)
        range_vals[zero_range_indices] = 1 
        X_outlier[v] = (view_data - min_vals) / range_vals
        X_outlier[v][np.isnan(X_outlier[v])] = 0 
        X_outlier[v] = np.clip(X_outlier[v], 0, 1)

    # --- Generate Attribute Outliers ---
    if num_attr_outliers > 0:
        print(f'Generating {num_attr_outliers} attribute outliers...')
        for idx in attr_idx:
            for v in range(V): X_outlier[v][idx, :] = np.random.rand(X_outlier[v].shape[1]) 
                
    # --- Generate Class-Attribute Outliers ---
    if num_class_attr_outliers > 0:
        print(f'Generating {num_class_attr_outliers} class-attribute outliers...')
        num_views_to_modify_ca = int(np.floor(V / 2)) 
        if num_views_to_modify_ca > 0:
            for idx in class_attr_idx:
                views_to_modify = np.random.choice(V, size=num_views_to_modify_ca, replace=False) 
                for v in views_to_modify: X_outlier[v][idx, :] = np.random.rand(X_outlier[v].shape[1])
        else: print('Skipping class-attribute outlier generation as floor(V/2) = 0.')
             
    # --- Generate Class Outliers ---
    num_pairs_target = int(np.floor(num_class_outliers / 2)) 
    if num_class_outliers > 1 and num_pairs_target > 0:
        print(f'Generating class outliers (targeting {num_pairs_target} pairs)...')
        num_views_to_swap = int(np.floor(V / 2)) 
        if num_views_to_swap > 0:
            shuffled_class_idx = np.random.permutation(class_idx) 
            paired_indices_flag = np.zeros(N, dtype=bool) 
            pairs_formed = [] 
            num_pairs_formed_count = 0
            for i in range(len(shuffled_class_idx)):
                if num_pairs_formed_count >= num_pairs_target: break
                idx1 = shuffled_class_idx[i]
                if paired_indices_flag[idx1]: continue 
                for j in range(i + 1, len(shuffled_class_idx)):
                     idx2 = shuffled_class_idx[j]
                     if not paired_indices_flag[idx2] and (Y[idx1] != Y[idx2]):
                         pairs_formed.append((idx1, idx2)); paired_indices_flag[idx1] = True
                         paired_indices_flag[idx2] = True; num_pairs_formed_count += 1; break 
            print(f'Successfully formed {num_pairs_formed_count} pairs for class outlier swaps.')
            for idx1, idx2 in pairs_formed:
                views_to_swap_indices = np.random.choice(V, size=num_views_to_swap, replace=False) 
                for v in views_to_swap_indices:
                    temp_row = X_outlier[v][idx1, :].copy()
                    X_outlier[v][idx1, :] = X_outlier[v][idx2, :].copy()
                    X_outlier[v][idx2, :] = temp_row
        else: print('Skipping class outlier swap generation as floor(V/2) = 0.')
    elif num_class_outliers <= 1: print(f'Skipping class outlier generation (num_class_outliers={num_class_outliers}).')
    elif num_pairs_target == 0: print('Skipping class outlier swap generation (target pairs = 0).')

    print('Outlier generation complete.')
    return X_outlier, Y_outlier 

# --- IAMOD Core Algorithm ---
def calculate_laplacian(adj_matrix):
    """Calculates the normalized Laplacian L=I-D^{-1/2}(A+I)D^{-1/2} from an adjacency matrix."""
    N = adj_matrix.shape[0]
    I = np.eye(N)
    A_with_self_loops = adj_matrix + I  
    D_diag = np.sum(A_with_self_loops, axis=1)
    D_diag[D_diag == 0] = 1.0 
    D_inv_sqrt_diag = np.power(D_diag, -0.5)
    D_inv_sqrt = np.diag(D_inv_sqrt_diag)
    Anorm = D_inv_sqrt.dot(A_with_self_loops).dot(D_inv_sqrt) 
    L = I - Anorm 
    return L

def graph_filtering(X_v, k_filt, s, m):
    """Applies graph filtering (I - sL)^m X to feature matrix X_v."""
    N = X_v.shape[0]
    try:
        nn = NearestNeighbors(n_neighbors=k_filt + 1, algorithm='auto').fit(X_v)
        _, indices = nn.kneighbors(X_v)
        adj = np.zeros((N, N))
        for i in range(N):
             adj[i, indices[i, 1:]] = 1 
        adj = np.maximum(adj, adj.T) 
    except Exception as e:
         print(f"Error in KNN for filtering view: {e}")
         adj = np.eye(N) 

    L_v = calculate_laplacian(adj)
    H_v = X_v.copy() 
    filter_op = np.eye(N) - s * L_v
    for _ in range(m):
        H_v = filter_op.dot(H_v) 
    return H_v

# MODIFIED run_cmod to calculate AUC per epoch
def run_cmod(X_list, Y_outlier, V, N, k_contrast, alpha, gamma, beta, s, m, learning_rate=0.001, epochs=50):
    """
    Runs IAMOD and calculates AUC per epoch.
    
    Args:
        X_list (list): List of input feature matrices per view.
        Y_outlier (np.ndarray): Ground truth outlier labels (0 or 1).
        ... (other parameters as before) ...

    Returns:
        tuple: (max_auc, best_epoch, final_scores, final_S)
    """
    I = np.eye(N)
    
    # --- Step 1: Graph Filtering ---
    print("Applying graph filtering...")
    start_time = time.time()
    H = [] 
    k_filter = k_contrast 
    for v in range(V):
        print(f"Filtering view {v+1}/{V}...")
        H_v = graph_filtering(X_list[v], k_filter, s, m)
        H.append(H_v)
    print(f"Graph filtering finished. Time: {time.time() - start_time:.2f}s")

    start_time = time.time()
    H_Ht = [Hv.dot(Hv.T) for Hv in H] 
    print(f"Precomputing H H^T finished. Time: {time.time() - start_time:.2f}s")

    # --- Step 2: Neighbor Identification ---
    print("Identifying neighbors for contrastive loss...")
    start_time = time.time()
    nbrs_inx = [] 
    for v in range(V):
        try:
            nn = NearestNeighbors(n_neighbors=k_contrast + 1, algorithm='auto').fit(H[v])
            _, indices = nn.kneighbors(H[v])
            nbrs_inx.append(indices[:, 1:].astype(int)) 
        except Exception as e:
            print(f"Error finding neighbors for view {v}: {e}")
            nbrs_inx.append(np.zeros((N, k_contrast), dtype=int))

    consistent_neighbors_idx = [] 
    min_views_for_consistency = math.floor(V / 2) + 1
    # min_views_for_consistency = V  ##### consistency across all views
    for i in range(N):
        neighbor_counts = {}
        for v in range(V):
            for neighbor_index in nbrs_inx[v][i]:
                neighbor_counts[neighbor_index] = neighbor_counts.get(neighbor_index, 0) + 1
        consistent_set = {neighbor_idx for neighbor_idx, count in neighbor_counts.items() 
                          if count >= min_views_for_consistency and neighbor_idx != i} 
        consistent_neighbors_idx.append(consistent_set)
    print(f"Identifying neighbors finished. Time: {time.time() - start_time:.2f}s")

    # --- Step 3: Initialization ---
    print("Initializing S and lambda_v...")
    S = np.eye(N); np.fill_diagonal(S, 0) 
    lambda_v = np.ones(V) / V 
    adam_cache = None 
    
    # Variables to track best AUC
    auc_per_epoch = []
    max_auc = -1.0
    best_epoch = -1
    best_S_for_max_auc = None # Store S when max AUC occurs

    # --- Step 4: Alternating Optimization ---
    print("Starting Alternating Optimization...")
    start_time_opt = time.time()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # --- 4a) Update Consensus Graph S ---
        grad_S = np.zeros((N, N))
        H_Ht_S = [Hv_Ht.dot(S) for Hv_Ht in H_Ht]

        for i in range(N):
            S_row_i_exp = np.exp(S[i, :])
            valid_p_indices = np.arange(N) != i
            k0 = np.sum(S_row_i_exp[valid_p_indices]); k0 = max(k0, 1e-9)
            exp_S_ij_divided_by_k0 = S_row_i_exp / k0; exp_S_ij_divided_by_k0[i] = 0 

            grad_recon_i, grad_contrast_i = np.zeros(N), np.zeros(N)
            for v in range(V):
                grad_recon_i += lambda_v[v] * (-H_Ht[v][i, :] + H_Ht_S[v][i, :])
                grad_contrast_v_i = exp_S_ij_divided_by_k0.copy() 
                if i < len(consistent_neighbors_idx):
                    consistent_idx_list = list(consistent_neighbors_idx[i])
                    if consistent_idx_list: grad_contrast_v_i[consistent_idx_list] -= 1
                grad_contrast_i += lambda_v[v] * grad_contrast_v_i

            grad_S[i, :] = 2 * grad_recon_i + alpha * grad_contrast_i; grad_S[i, i] = 0 

        S, adam_cache = adam(S, grad_S, adam_cache, lr=learning_rate)
        np.fill_diagonal(S, 0) 

        # --- 4b) Update View Weights lambda_v ---
        M = np.zeros(V)
        for v in range(V):
            loss_recon_v = np.linalg.norm(H[v].T - H[v].T.dot(S), 'fro')**2
            loss_contrast_v = 0
            for i in range(N):
                S_row_i_exp = np.exp(S[i, :])
                valid_p_indices = np.arange(N) != i
                k0 = np.sum(S_row_i_exp[valid_p_indices]); k0 = max(k0, 1e-9)
                if i < len(consistent_neighbors_idx): 
                    for j in consistent_neighbors_idx[i]:
                         log_arg = S_row_i_exp[j] / k0
                         if log_arg > 1e-9: loss_contrast_v -= math.log( log_arg )
            M[v] = loss_recon_v + alpha * loss_contrast_v
        
        if gamma <= 1: lambda_v = np.ones(V) / V 
        else:
             power_term = 1.0 / (gamma - 1.0); M_v_safe = M + 1e-9 
             lambda_v_new = np.power(M_v_safe / gamma, -power_term) 
             lambda_v_sum = np.sum(lambda_v_new)
             if lambda_v_sum > 1e-9 and not np.isnan(lambda_v_sum) and not np.isinf(lambda_v_sum):
                 lambda_v = lambda_v_new / lambda_v_sum
             else: lambda_v = np.ones(V) / V
                 
        # --- Calculate Outlier Scores and AUC for CURRENT epoch ---
        # Use current S from this epoch
        current_S = S 
        current_outlier_scores = np.zeros(N)
        
        # Precompute Reconstructed H based on current S
        S_no_diag_current = current_S - np.diag(np.diag(current_S))
        H_reconstructed_list_current = [S_no_diag_current.dot(Hv) for Hv in H] 

        # Calculate contrastive loss part for current S
        total_mod_contrast_loss_i_current = np.zeros(N)
        for i in range(N):
            S_row_i_exp = np.exp(current_S[i, :])
            valid_p_indices = np.arange(N) != i
            k0 = np.sum(S_row_i_exp[valid_p_indices]); k0 = max(k0, 1e-9)
            loss_i = 0
            if i < len(consistent_neighbors_idx): 
                for j in consistent_neighbors_idx[i]:
                     log_arg = S_row_i_exp[j] / k0
                     if log_arg > 1e-9: loss_i -= math.log(log_arg)
            total_mod_contrast_loss_i_current[i] = loss_i
            
        # Calculate final score for current epoch
        for i in range(N):
            total_recon_error_i = 0
            for v in range(V):
                if i < H[v].shape[0] and i < H_reconstructed_list_current[v].shape[0]: 
                    recon_error_i_v = np.linalg.norm(H[v][i, :] - H_reconstructed_list_current[v][i, :])**2 
                    total_recon_error_i += recon_error_i_v
                
            if i < len(total_mod_contrast_loss_i_current): 
                current_outlier_scores[i] = total_recon_error_i + beta * V * total_mod_contrast_loss_i_current[i]
            else:
                current_outlier_scores[i] = np.nan 
        
        # Calculate AUC for current epoch
        current_auc = np.nan # Default to NaN
        valid_score_indices = ~np.isnan(current_outlier_scores)
        num_valid_scores = np.sum(valid_score_indices)
        if num_valid_scores > 1:
            valid_scores = current_outlier_scores[valid_score_indices]
            valid_labels = Y_outlier[valid_score_indices]
            if len(np.unique(valid_labels)) > 1:
                try:
                    current_auc = roc_auc_score(valid_labels, valid_scores)
                except ValueError:
                    current_auc = np.nan # Handle cases where roc_auc_score fails
            
        auc_per_epoch.append(current_auc)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs} completed. Time: {epoch_time:.2f}s. AUC: {current_auc:.4f}")

        # Update max AUC and best S
        if not np.isnan(current_auc) and current_auc > max_auc:
            max_auc = current_auc
            best_epoch = epoch + 1
            best_S_for_max_auc = current_S.copy() # Store the S that gave max AUC
            
    # End of optimization loop
    print(f"Optimization finished. Total time: {time.time() - start_time_opt:.2f}s")
    
    final_S = S # S after the last epoch
    # Use best_S_for_max_auc if needed, otherwise calculate final scores with final_S
    final_scores = current_outlier_scores # Scores from the last epoch

    # --- Return results ---
    return max_auc, best_epoch, final_scores, final_S # Return max AUC info


# --- Main Execution Block ---
if __name__ == "__main__":
    
    # --- Configuration ---
    # mat_file_path = './dataset/Caltech101-7.mat'
    mat_file_path = './dataset/handwritten.mat'
    # mat_file_path = './dataset/bbcsport.mat'
    # mat_file_path = './dataset/Movies.mat'
    # mat_file_path = './dataset/Reuters.mat'
    # mat_file_path = './dataset/WebKB.mat'
    
    dataname = "Handwritten"
    if not os.path.exists(mat_file_path):
        print(f"Warning: MAT file not found at '{mat_file_path}'. Exiting.") 
        exit() 

    proportion_a = 0.05      # Proportion of attribute outliers
    proportion_b = 0.05      # Proportion of class outliers
    proportion_c = 0.05      # Proportion of class-attribute outliers
    
    # default parameters
    s = 0.5
    m = 2
    learning_rate = 0.001
    epochs = 20   # Increased epochs for monitoring
    
    f = open('config.yaml')
    config_data = yaml.safe_load(f)
    # k_contrast = config_data['{}'.format(dataname)]['k_contrast']
    # alpha = config_data['{}'.format(dataname)]['alpha']
    # gama = config_data['{}'.format(dataname)]['gama']
    # beta = config_data['{}'.format(dataname)]['beta']
    
    print("--- IAMOD Algorithm Execution with Epoch AUC Monitoring ---")

    # --- Generate Data with Outliers ---
    try:
        X_outlier, Y_outlier = generate_multiview_outliers_py(mat_file_path, proportion_a, proportion_b, proportion_c)
        V = len(X_outlier); N = X_outlier[0].shape[0]; D_list = [X_outlier[v].shape[1] for v in range(V)]
        print(f"\nData ready: {N} instances, {V} views, dimensions {D_list}")
        print(f"Ground truth labels: {np.sum(Y_outlier)} outliers, {N - np.sum(Y_outlier)} normals")
    except Exception as e:
        print(f"\nError during data generation: {e}\nExiting."); exit()

    # --- Run IAMOD Algorithm ---
    try:
        # Pass Y_outlier to run_cmod now
        max_auc_result, best_epoch_result, final_epoch_scores, final_S_matrix = run_cmod(
            X_list=X_outlier, 
            Y_outlier=Y_outlier, # Pass ground truth labels
            V=V, N=N, k_contrast=k_contrast, alpha=alpha, gamma=gamma, beta=beta, 
            s=s, m=m, learning_rate=learning_rate, epochs=epochs
        )
    except Exception as e:
        print(f"\nError during CMOD execution: {e}")
        import traceback; traceback.print_exc(); print("Exiting."); exit()

    # --- Final Results ---
    print("\n--- CMOD Final Results ---")
    if best_epoch_result != -1:
         print(f"Maximum AUC achieved during training: {max_auc_result:.4f} at Epoch {best_epoch_result}")
    else:
         print("No valid AUC score was calculated during training.")
         
    # Calculate AUC based on final epoch scores for reference
    print("\nAUC based on scores from the FINAL epoch:")
    valid_final_score_indices = ~np.isnan(final_epoch_scores)
    num_valid_final_scores = np.sum(valid_final_score_indices)
    if num_valid_final_scores > 1:
        valid_final_scores = final_epoch_scores[valid_final_score_indices]
        valid_final_labels = Y_outlier[valid_final_score_indices]
        if len(np.unique(valid_final_labels)) > 1:
            try:
                # final_auc = roc_auc_score(valid_final_labels, -valid_final_scores)
                final_auc = roc_auc_score(valid_final_labels, valid_final_scores)
                print(f"AUC Score (Final Epoch): {final_auc:.4f}")
            except ValueError as e_auc: print(f"Could not calculate final AUC: {e_auc}")
        else: print("Could not calculate final AUC: Only one class present.")
    else: print("Could not calculate final AUC: Not enough valid final scores.")
    
    print("\nFinal Consensus Graph S (sample 5x5):")
    print(np.round(final_S_matrix[:5, :5], 4))
    print("-----------------------------")
