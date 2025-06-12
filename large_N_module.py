# Put commonly reused functions here to be imported in other files
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
import random
from IPython.display import display
from IPython.display import display, Math
from matplotlib.colors import ListedColormap

# Create the one-qubit operations
Id =  sparse.csc_matrix(np.array([[1.,0.],[0.,1.]]))
P1 = sparse.csc_matrix(np.array([[0., 0.],[0., 1.]]))
P0 = sparse.csc_matrix(np.array([[1.,0.],[0.,0.]]))
Xop = sparse.csc_matrix(np.array([[0.,1.],[1.,0.]]))
Yop = sparse.csc_matrix(np.array([[0.,-1.j],[1.j,0.]]))


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
A function that acts with a one-qubit operator on a specified qubit. So, it takes in a 2x2 matrix and spits out a 2^n x 2^n matrix.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def one_qubit_gate( M: sparse._csc.csc_matrix, p: int, n: int) -> sparse._csc.csc_matrix:
    if (M.shape != (2,2)):
        print('Warning: Expected M argument to be a 2x2 sparse matrix.')
    if ( p<0 or p>=n ):
        print(f'Warning: Expected p argument to be within the range [0, {n}-1] inclusive.')

    result = sparse.eye(1, format='csc') # create a sparse 1x1 matrix to store result
    for i in range(n):
        if (i==p):
            result = sparse.kron(result, M, format='csc')
        else:
            result =sparse.kron(result, Id, format='csc')  
    return result

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    Define a function that returns an n-qubit gate that is in the form of a tensor product of single qubit gates.

    Expects the dictionary op_dict to be of the form { p1: M1, p2: M2, p3: M3 } where the p's are integers specifying
    qubits and the M's are 2x2 matrices in sparse format specifying one-qubit gates.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def multi_qubit_gate( op_dict: dict, n: int) -> sparse._csc.csc_matrix:
    result = sparse.eye(1, format='csc') # Create a 1x1 in sparse format to build our operator
    for i in range(n):
        if i in op_dict:
            result =sparse.kron(result, op_dict[i], format='csc')
        else:
            result =sparse.kron(result, Id, format='csc')
    return result

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Convert between the single-index p which enumerates qubits, and double-index (i,j) which enumerates lattice sites
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def map_to_circuit( i, j, lat_shape ):
    L, H = lat_shape
    if i >= L:
        print(f"[map_to_circuit] i index out of bounds: expected i<{L}, but i={i}")
    if j >= H:
        print(f"[map_to_circuit] j index out of bounds, expected j<{H}, but j={j}")
    
    return j*L + i


def map_to_lattice( p, lat_shape ):
    L, H = lat_shape
    if p >= (L*H):
        print(f"[map_to_lattice] p index out of bounds, expected p<{L*H}, but p={p}")

    return p % L, p // L


def get_neighbours( p, lat_shape, boundary='periodic' ):
    L, H = lat_shape
    i, j = map_to_lattice(p, lat_shape)
    if boundary == 'periodic':
        nl   = map_to_circuit((i - 1) % L, j, lat_shape)
        nd   = map_to_circuit(i, (j - 1) % H, lat_shape)
        nr   = map_to_circuit((i + 1) % L, j, lat_shape)
        nu   = map_to_circuit(i, (j + 1) % H, lat_shape)
        return nl, nd, nr, nu
    elif boundary == 'open':
        if i == 0:
            nl = None
        else:
            nl = map_to_circuit(i - 1, j, lat_shape)
        if j == 0:
            nd = None
        else:
            nd = map_to_circuit(i, j - 1, lat_shape)
        if i == L - 1:
            nr = None
        else:
            nr = map_to_circuit(i + 1, j, lat_shape)
        if j == H - 1:
            nu = None
        else:
            nu = map_to_circuit(i, j + 1, lat_shape)
        return (nl, nd, nr, nu)
    


# A function that applies the specified operator to every qubit in the lattice, and then returns the resulting matrix
def sigma(op_name, lat_shape, boundary='periodic'):
    L = lat_shape[0]
    H = lat_shape[1]
    n = L*H

    op = sparse.csc_matrix((2**n, 2**n))

    if boundary == "periodic":

        if op_name == "P1":
            for p in range(n):
                op += one_qubit_gate(P1, p, n)

        elif op_name == "P1P1":
            for p in range(n):
                i, j = map_to_lattice(p, lat_shape)
                nr   = map_to_circuit((i + 1) % L, j, lat_shape)
                nu   = map_to_circuit(i, (j + 1) % H, lat_shape)
                op += multi_qubit_gate( {p: P1, nr: P1}, n )
                op += multi_qubit_gate( {p: P1, nu: P1}, n )

        elif op_name == "P0P0P0P0X":
            for p in range(n):
                nl, nd, nr, nu = get_neighbours(p, lat_shape)
                op += multi_qubit_gate( {p: Xop, nl: P0, nd: P0, nr: P0, nu: P0}, n )

        elif op_name == "P0P0P0P1X":
            for p in range(n):
                nl, nd, nr, nu = get_neighbours(p, lat_shape)
                op += multi_qubit_gate( {p: Xop, nl: P1, nd: P0, nr: P0, nu: P0}, n )
                op += multi_qubit_gate( {p: Xop, nl: P0, nd: P1, nr: P0, nu: P0}, n )
                op += multi_qubit_gate( {p: Xop, nl: P0, nd: P0, nr: P1, nu: P0}, n )
                op += multi_qubit_gate( {p: Xop, nl: P0, nd: P0, nr: P0, nu: P1}, n )
        
        elif op_name == "P0P0P0P0Y":
            for p in range(n):
                nl, nd, nr, nu = get_neighbours(p, lat_shape)
                op += multi_qubit_gate( {p: Yop, nl: P0, nd: P0, nr: P0, nu: P0}, n )
        
        elif op_name == "P0P0P0P1Y":
            for p in range(n):
                nl, nd, nr, nu = get_neighbours(p, lat_shape)
                op += multi_qubit_gate( {p: Yop, nl: P1, nd: P0, nr: P0, nu: P0}, n )
                op += multi_qubit_gate( {p: Yop, nl: P0, nd: P1, nr: P0, nu: P0}, n )
                op += multi_qubit_gate( {p: Yop, nl: P0, nd: P0, nr: P1, nu: P0}, n )
                op += multi_qubit_gate( {p: Yop, nl: P0, nd: P0, nr: P0, nu: P1}, n )
        else:
            raise ValueError(f"Unknown operator name: {op_name}")
    
    elif boundary == "open":
        raise NotImplementedError("Open boundary conditions not implemented yet.")
    
    return op


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''' 
Create a function that produces the Hamiltonian matrix
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def make_electric_ham(g: float, lat_shape: list, boundary: str = 'periodic', N_colors: int = 3 ) -> sparse._csc.csc_matrix:
    H_E = sigma('P1', lat_shape, boundary)
    H_E -= 0.5 * sigma('P1P1', lat_shape, boundary)
    return (N_colors - 1/N_colors)*g*g * H_E


def make_magnetic_ham( g: float, lat_shape: tuple, boundary='periodic', N_colors: int = 3) -> sparse._csc.csc_matrix:
    coeff1 = -np.sqrt(2)/(2*g*g)
    coeff2 = -1/(2*g*g*N_colors)
    H_B1 = sigma('P0P0P0P0X', lat_shape, boundary)
    H_B2 = sigma('P0P0P0P1X', lat_shape, boundary)
    return coeff1 * H_B1  +   coeff2 * H_B2


def make_ham(g: float, lat_shape: tuple, boundary: str ='periodic', N_colors: int =3 ) -> sparse._csc.csc_matrix:
    HE = make_electric_ham(g, lat_shape, boundary, N_colors)
    HB = make_magnetic_ham(g, lat_shape, boundary, N_colors)
    return HE + HB


def get_ground_state_energy(g, lat_shape, boundary='periodic', N_colors=3):
    H = make_ham(g, lat_shape, boundary, N_colors)
    evals, evecs = sparse.linalg.eigsh( H, which='SA', k=1)
    return evals[0]

def get_ground_state(g, lat_shape, boundary='periodic', N_colors=3):
    H = make_ham(g, lat_shape, boundary, N_colors)
    _, evecs = sparse.linalg.eigsh( H, which='SA', k=1)
    return evecs[:,0] / np.linalg.norm(evecs[:,0]) # Normalize the ground state vector


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
ADAPT VQE CODE BELOW
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''




'''Helper functions'''


# These ones only require lattice size information

def get_operator_pool(op_names, lat_shape, boundary="periodic"):
    op_pool = [sigma(op_name, lat_shape, boundary) for op_name in op_names]
    return op_pool

def get_imag_operator_pool(op_names, lat_shape, boundary="periodic"):
    operator_pool = get_operator_pool(op_names, lat_shape, boundary)
    return [op.imag for op in operator_pool]

def construct_electric_vac(lat_shape):
    n = lat_shape[0]*lat_shape[1]
    return sparse.csc_matrix( ([1] + [0]*(2**n - 1)) ).T

def expectation_value(state, op):
    return (state.T.conj() @ (op @ state)).sum().real

# These ones require the coupling / hamiltonian

def get_commutator_pool(H, imag_operator_pool):
    return [-(H @ op - op @ H) for op in imag_operator_pool]


'''
Optimzation step
'''

def opt_step(psi_prev, epss_prev, indices_prev, H, lat_shape, commutator_pool, psi_0, imag_operator_pool):
    n = lat_shape[0]*lat_shape[1]

    ips = np.array([expectation_value(psi_prev, op) for op in commutator_pool])
    min_index = random.choice(np.where(ips == np.max(np.abs(ips)))[0])
    new_indices = np.append( indices_prev.astype(int), int(min_index) )
    new_epss = np.append(epss_prev, 0.01)

    def ansatz(epss):
        U = sp.sparse.eye(2**n, format="csc")
        for eps, idx in zip(epss, new_indices):
            U = sp.sparse.linalg.expm( eps * imag_operator_pool[idx]) @ U
        return U @ psi_0
    
    def objective_ev(epss):
        new_state = ansatz(epss)
        return expectation_value(new_state, H)
    
    sol = sp.optimize.minimize(objective_ev, new_epss)
    opt_epss = sol["x"]
    new_energy = sol["fun"]
    new_state = ansatz(opt_epss)

    return new_indices, new_state, opt_epss, new_energy


def adapt_vqe(g, lat_shape, boundary="periodic", N_colors=3, op_pool_names=["P0P0P0P0Y", "P0P0P0P1Y"], delta_E=0.0001):

    H = make_ham(g, lat_shape, boundary, N_colors)
    imag_operator_pool = get_imag_operator_pool(op_pool_names, lat_shape, boundary)
    commutator_pool = get_commutator_pool(H, imag_operator_pool)

    # store results
    psi_0 = construct_electric_vac(lat_shape)
    epss = []
    indices = []
    energies = []

    # use in loop
    new_state = psi_0
    opt_epss = np.array(epss)
    new_indices = np.array(indices)

    while True:
        new_indices, new_state, opt_epss, new_energy = opt_step(new_state, opt_epss, new_indices, H, lat_shape, commutator_pool, psi_0, imag_operator_pool)
        indices.append( new_indices[-1] )
        epss.append( opt_epss )
        energies.append( new_energy )
        if len(energies) > 2 and abs(energies[-1] - energies[-2]) <= delta_E:
            break
    
    return energies, indices, epss, new_state



'''
Trotterized version
'''


def opt_step_trotterized(psi_prev, epss_prev, indices_prev, H, lat_shape, commutator_pool, psi_0, op_pool_names):
    n = lat_shape[0]*lat_shape[1]
    ips = np.array([expectation_value(psi_prev, op) for op in commutator_pool])
    min_index = random.choice(np.where(ips == np.max(np.abs(ips)))[0])
    new_indices = np.append( indices_prev.astype(int), int(min_index) )
    new_epss = np.append(epss_prev, 0.01)

    def make_U_trotterized(eps, op_name, lat_shape):
        n = lat_shape[0]*lat_shape[1]
        RY = np.cos(eps)*Id + 1j*np.sin(eps)*Yop
        U = sparse.eye(2**n, format="csc")
        
        if op_name == "P0P0P0P0Y":
            for p in range(n):
                nl, nd, nr, nu = get_neighbours(p, lat_shape)
                P = multi_qubit_gate({nl: P0, nr: P0, nu: P0, nd: P0}, n)
                trotter_step = one_qubit_gate( RY, p, n )@P + (sp.sparse.eye(2**n, format='csc') - P)
                U = trotter_step @ U
        
        elif op_name == "P0P0P0P1Y":
            for p in range(n):
                nl, nd, nr, nu = get_neighbours(p, lat_shape)
                P =  multi_qubit_gate({nl: P1, nr: P0, nu: P0, nd: P0}, n)
                P += multi_qubit_gate({nl: P0, nr: P1, nu: P0, nd: P0}, n)
                P += multi_qubit_gate({nl: P0, nr: P0, nu: P1, nd: P0}, n)
                P += multi_qubit_gate({nl: P0, nr: P0, nu: P0, nd: P1}, n)
                trotter_step = one_qubit_gate( RY, p, n )@P + (sp.sparse.eye(2**n, format='csc') - P)
                U = trotter_step @ U
        else:
            raise ValueError(f"Unknown operator name: {op_name}")
        
        return U

    def ansatz(epss):
        U = sp.sparse.eye(2**n, format="csc")
        for eps, idx in zip(epss, new_indices):
            U = make_U_trotterized(eps, op_pool_names[idx], lat_shape) @ U
        return U @ psi_0
    
    def objective_ev(epss):
        new_state = ansatz(epss)
        return expectation_value(new_state, H)
    
    sol = sp.optimize.minimize(objective_ev, new_epss)
    opt_epss = sol["x"]
    new_energy = sol["fun"]
    new_state = ansatz(opt_epss)

    return new_indices, new_state, opt_epss, new_energy


def adapt_vqe_trotterized(g, lat_shape, boundary="periodic", N_colors=3, op_pool_names=["P0P0P0P0Y", "P0P0P0P1Y"], delta_E=0.0001):

    H = make_ham(g, lat_shape, boundary, N_colors)
    imag_operator_pool = get_imag_operator_pool(op_pool_names, lat_shape, boundary)
    commutator_pool = get_commutator_pool(H, imag_operator_pool)

    # store results
    psi_0 = construct_electric_vac(lat_shape)
    epss = []
    indices = []
    energies = []

    # use in loop
    new_state = psi_0
    opt_epss = np.array(epss)
    new_indices = np.array(indices)

    while True:
        new_indices, new_state, opt_epss, new_energy = opt_step_trotterized(new_state, opt_epss, new_indices, H, lat_shape, commutator_pool, psi_0, op_pool_names)
        indices.append( new_indices[-1] )
        epss.append( opt_epss )
        energies.append( new_energy )
        if len(energies) > 2 and abs(energies[-1] - energies[-2]) <= delta_E:
            break
    
    return energies, indices, epss, new_state





'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
CODE FOR DRAWING STATES
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Take in a vector and write it as a sum of kets, with the kets being the computational basis states.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def write_state_as_kets( vec, max_num_states=10 ):
    # Get the number of qubits from the length of the vector
    n = int(np.log2(len(vec)))

    # Trim the vector down to its k largest (absolute) values
    vec = keep_k_largest(vec, k=max_num_states)
    
    # Make a list of tuples of the form (coefficient, state) 
    terms = [ (vec[i], format(i, f"0{n}b") ) for i in range(len(vec)) if np.abs(vec[i]) > 0.00000001 ]
    # Sort them in descending order based on their coefficients
    terms = sorted( terms, key=lambda t: abs(t[0]), reverse=True )
    
    # Make this into a list of LaTeX formatted terms
    latex_terms = [ fr'{{{coef:.3f}}}\ket{{{state}}}' for coef, state in terms ]
    expr = " + ".join(latex_terms)
    
    # Write it
    display(Math(expr))
    return

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Take in a vector specifying a state, and the lattice shape (L,H), and plot the state.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def draw_state( vec, lat_shape, max_num_states=10 ):
    # Convert the lattice shape to an array shape
    arr_shape = (lat_shape[1], lat_shape[0])
    
    # Get the number of qubits from the lattice shape, and make sure the vector and lattice shapes are compatible
    L, H = lat_shape[0], lat_shape[1]
    n = L*H
    if n != int(np.log2(len(vec))):
        print(f"[draw_state]: The number of qubits in vec ({int(np.log2(len(vec)))}) and the lattice shape ({lat_shape}) are not compatible")

    # Trim the vector down to its k largest (absolute) values
    vec = keep_k_largest(vec, k=max_num_states)

    # Make a list of tuples of the form (coefficient, binary_string), sorted by size of coefficient
    terms = [ (vec[i], format(i, f"0{n}b") ) for i in range(len(vec)) if np.abs(vec[i]) > 0.0000001 ]
    terms = sorted( terms, key=lambda t: abs(t[0]), reverse=True )

    
    # Create side-by-side plots
    fig, axs = plt.subplots(1, len(terms), figsize=(len(terms)*1.5, 1.5))

    # Handle edge case
    if len(terms) == 1:
        axs = [axs]

    for ax, term in zip(axs, terms):
        create_binary_plot_on_ax(ax, term[1], arr_shape)

    plt.tight_layout()
    plt.show()
    return



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# Create a function that takes in a binary string and a lattice shape, and plots the corresponding state
# The binary string should be of length L*H, where L and H are the dimensions of the lattice.
# The function should create a 2D plot of the binary string, with 0s and 1s represented by different colors.
# The function should also allow for optional gridlines to be drawn, to indicate the borders of the lattice.
# The function should draw the state using a different color if the state is not physical.
# For an unphysical state, the cluster containing a cycle is drawn in red.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def create_binary_plot(binary_str, lat_shape, borders=True):
    # Start by converting the binary string into a binary array
    binary_array = binary_array_from_list( binary_str, lat_shape )
    vmax = 1 # Default max value for the colormap
    
    # Create custom colormap
    cmap = ListedColormap(["#003366", "#FFD700"])

    # Check if the state is physical
    is_physical, cluster = is_physical_state(binary_str, lat_shape)

    # If the state is not physical, color the cluster containing a cycle in red
    if not is_physical:
        # Create a copy of the binary array to modify
        binary_array = np.copy(binary_array)
        # Set the cluster containing a cycle to red
        for x, y in cluster:
            binary_array[x, y] = 2
        cmap = ListedColormap(["#003366", "#FFD700", "#FF0000"])
        vmax = 2 # Set max value for the colormap to include the red color


    # Create the plot
    fig, ax = plt.subplots(figsize=(lat_shape[0], lat_shape[1]))
    im = ax.imshow(binary_array, cmap=cmap, interpolation="none", vmin=0, vmax=vmax)

    # Style the plot
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # Optional: add gridlines if you want to see borders
    if borders==True:
        plt.grid(color='white', linewidth=1)
        plt.gca().set_xticks(np.arange(-.5, lat_shape[0], 1), minor=True)
        plt.gca().set_yticks(np.arange(-.5, lat_shape[1], 1), minor=True)
        plt.gca().grid(which="minor", color="white", linewidth=2)
        plt.gca().tick_params(which="minor", bottom=False, left=False)

    plt.close(fig)  # prevent automatic display

    # Return the figure and axis for later use
    return fig, ax


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Take in a binary string representing a computational basis state and determine if the state is physical or not.
In the (1,2,1) Hamiltonian, you can check if a computational basis state is physical in the following way:
1. Create a set of disjointed graphs by connecting all neighboring 1's to each other.
2. Check if any of the graphs have any cycles.
3. If any of the graphs have cycles, the state is not physical. Otherwise, it is physical.
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

def is_physical_state( binary_str, lat_shape ):
    # Convert the binary string into a binary array
    binary_array = binary_array_from_list( binary_str, lat_shape )

    arr_shape = (lat_shape[1], lat_shape[0])

    # Loop through the binary array and find all the clusters of 1's
    clusters = []
    visited = np.zeros_like(binary_array, dtype=bool)
    for i in range(arr_shape[0]):
        for j in range(arr_shape[1]):
            if binary_array[i, j] == 1 and not visited[i, j]:
                # Start a new cluster
                cluster = []
                stack = [(i, j)]
                while stack:
                    x, y = stack.pop()
                    if visited[x, y]:
                        continue
                    visited[x, y] = True
                    cluster.append((x, y))
                    # Check neighbors
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = (x + dx) % arr_shape[0], (y + dy) % arr_shape[1]
                        if binary_array[nx, ny] == 1 and not visited[nx, ny]:
                            stack.append((nx, ny))
                clusters.append(cluster)

    # Check if any of the clusters have cycles
    for cluster in clusters:
        # Create a set to keep track of visited nodes
        visited = set()
        stack = [cluster[0]]
        while stack:
            node = stack.pop()
            if node in visited:
                # Print the cycle for debugging
                # print("Cycle detected in cluster:", cluster)
                # Cycle detected
                return False, cluster
            # Mark the node as visited 
            visited.add(node)
            # Check neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = (node[0] + dx) % arr_shape[0], (node[1] + dy) % arr_shape[1]
                if (nx, ny) in cluster and (nx, ny) not in visited:
                    stack.append((nx, ny))
    # If no cycles were found, the state is physical
    return True, []



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Helper functions below
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


# Helper function, to turn a binary string into an array with the shape of the lattice
def binary_array_from_list( binary_str: str, lat_shape: tuple):
    arr_shape = (lat_shape[1], lat_shape[0])

    # Check if the length of the binary string matches the expected size
    if len(binary_str) != arr_shape[0] * arr_shape[1]:
        raise ValueError(f"Binary string length {len(binary_str)} does not match expected size {arr_shape[0] * arr_shape[1]}")
    # Check if the binary string contains only 0s and 1s
    if not all(bit in '01' for bit in binary_str):
        raise ValueError("Binary string must contain only 0s and 1s")
    # Convert the string into a list of integers (0s and 1s)
    binary_list = [int(bit) for bit in binary_str]

    # Convert the list into a NumPy array with the desired shape
    binary_array = np.array(binary_list).reshape(arr_shape)

    return binary_array


# The create_binary_plot function, now slightly tweaked to draw on an Axes object
# Helper function for draw_state()
# Like the create_binary_plot function, it will check if the state is physical or not, and color the cluster containing a cycle in red.
def create_binary_plot_on_ax(ax, binary_str, shape):
    binary_array = np.array([int(b) for b in binary_str]).reshape(shape)
    cmap = ListedColormap(["#003366", "#FFD700"])
    vmax = 1 # Default max value for the colormap
    # Check if the state is physical
    is_physical, cluster = is_physical_state(binary_str, shape)
    # If the state is not physical, color the cluster containing a cycle in red
    if not is_physical:
        # Set the cluster containing a cycle to red
        for x, y in cluster:
            binary_array[x, y] = 2
        cmap = ListedColormap(["#003366", "#FFD700", "#FF0000"])
        vmax = 2 # Set max value for the colormap to include the red color

    ax.imshow(binary_array, cmap=cmap, interpolation="none", vmin=0, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    
# Helper function, trim an array to only its k largest values
def keep_k_largest(arr, k):
    # Make a copy so we don't modify the original
    result = np.zeros_like(arr)
    
    # Get indices of the k largest absolute values
    if k > 0:
        idx = np.argpartition(np.abs(arr), -k)[-k:]
        result[idx] = arr[idx]
    
    return result


'''
Trotterized time evolution
'''

def HE1_make_U_trotterized(g, lat_shape, t, N_colors=3):
    L, H = lat_shape
    n = L * H
    C = g*g*(N_colors - 1/N_colors)

    result = sparse.eye(1, format='csc') # create a sparse 1x1 matrix to store result
    HE1_op = sparse.linalg.expm( -1j * C * P1 * t)

    for i in range(n):
        result = sp.sparse.kron(result, HE1_op, format='csc')
    
    return result


def HE2_make_U_trotterized(g, lat_shape, t, boundary = "periodic", N_colors=3):
    L, H = lat_shape
    n = L * H
    HE2_coeff = -0.5 * (N_colors - 1/N_colors)*g*g

    U = sparse.eye(2**n, format="csc")
    for p in range(n):
        _, _, nr, nu = get_neighbours(p, lat_shape, boundary)

        P = multi_qubit_gate({p: P1, nu: P1}, n)
        op = np.exp(-1j * t * HE2_coeff) * P + ( sparse.eye(2**n, format="csc") - P )
        
        U = op @ U

        P = multi_qubit_gate({p: P1, nr: P1}, n)
        op = np.exp(-1j * t * HE2_coeff) * P + ( sparse.eye(2**n, format="csc") - P )
        
        U = op @ U
    
    return U
    

# Create a function to make the trotterized time evolution for the first magnetic term

def HB1_make_U_trotterized(g, lat_shape, t, boundary = "periodic"):
    L, H = lat_shape
    n = L * H
    HB1_coeff = - 1 / ( np.sqrt(2) * g * g)

    # Create the 2x2 sparse rotation matrix needed for the magnetic term
    RX = sparse.linalg.expm(-1j * HB1_coeff * t * Xop)

    U = sparse.eye(2**n, format="csc") # For storing the result

    # loop over the lattice sites
    for p in range(n):
        # Get the neighbours of the current site
        nl, nd, nr, nu = get_neighbours(p, lat_shape, boundary)
        # Create the multi-qubit projection operator for the neighbors
        P = multi_qubit_gate({nl: P0, nd: P0, nr: P0, nu: P0}, n)
        # Efficiently compute the 5-qubit operator
        op = sparse.eye(2**n, format="csc") - P + one_qubit_gate(RX, p, n) @ P

        U = op @ U

    return U

# Create a function to make the trotterized time evolution for the second magnetic term

def HB2_make_U_trotterized(g, lat_shape, t, boundary = "periodic", N_colors=3):
    L, H = lat_shape
    n = L * H
    HB2_coeff = - 1 / ( 2 * g * g * N_colors)

    # Create the 2x2 sparse rotation matrix needed for the magnetic term
    RX = sparse.linalg.expm(-1j * HB2_coeff * t * Xop)

    U = sparse.eye(2**n, format="csc") # For storing the result

    # loop over the lattice sites
    for p in range(n):
        # Get the neighbours of the current site
        nl, nd, nr, nu = get_neighbours(p, lat_shape, boundary)

        # Create the multi-qubit projection operator for the neighbors
        P =  multi_qubit_gate({nl: P1, nd: P0, nr: P0, nu: P0}, n)
        P += multi_qubit_gate({nl: P0, nd: P1, nr: P0, nu: P0}, n)
        P += multi_qubit_gate({nl: P0, nd: P0, nr: P1, nu: P0}, n)
        P += multi_qubit_gate({nl: P0, nd: P0, nr: P0, nu: P1}, n)
        # Efficiently compute the 5-qubit operator
        op = sparse.eye(2**n, format="csc") - P + one_qubit_gate(RX, p, n) @ P
        U = op @ U
    return U

def U_trotterized(g, lat_shape, t, boundary='periodic', N_colors=3):
    """
    Create the trotterized time evolution operator for the Hamiltonian H = HE1 + HE2 + HB1 + HB2.
    """
    U_E1 = HE1_make_U_trotterized(g, lat_shape, t, N_colors)
    U_E2 = HE2_make_U_trotterized(g, lat_shape, t, boundary, N_colors)
    U_B1 = HB1_make_U_trotterized(g, lat_shape, t, boundary)
    U_B2 = HB2_make_U_trotterized(g, lat_shape, t, boundary, N_colors)
    return U_E1 @ U_E2 @ U_B1 @ U_B2