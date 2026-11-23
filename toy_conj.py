#Packages of interest
#Note to self: Change to make a network x version of this code for simplicity
import numpy as np
import networkx as nx
import random

#Denote the number of vertices
#CHANGED: 12 -> 19. At N=12 the star already scores 0 (the max), so there is nothing to find.
N = 19
#The number of possible edges for the bitstring
M = int(N*(N-1) / 2)
COUNTEREXAMPLE_FILE = "Counterexamples.txt"
#CHANGED: tolerance so floating-point noise on stars (~1e-16) is not counted as a counterexample
TOL = 1e-9


def empty_starting_point() -> str:
    ''' Create the empty starting string of all 0's for the graphs bitstring '''
    return "0" * M


def test_empty_starting_point():
    result = empty_starting_point()

    assert len(result) == M
    assert set(result) == {"0"}
    assert result == "0" * M

    print("empty_starting_point() passed")


def bitstring_to_adj(s: str) -> np.ndarray:
    """
    This function is used to read off a bitstring and convert it into a adjacency matrix
    format
    """
    n = N
    #Create the empty adjacency matrix
    A = np.zeros((n, n), dtype=np.float64)
    #Index of the string value
    k = 0
    #The rows of the matrix
    for i in range(n):
        #The entries of the columns
        for j in range(i+1, n):
            #Check the value of the string bit
            if s[k] == "1":
                #If 1, then the adjacency matrix is symmetrically denoted as 1
                A[i, j] = 1.0
                A[j, i] = 1.0
            #Update k to move to 1 to move into the next bit-string
            k += 1
    #Return the Adjacency matrix
    return A


def bitstring_to_adj_v2(s: str) -> np.ndarray:
    "Same as above but it was made by claude as a more efficient alternative if needed"
    A = np.zeros((N, N))
    iu = np.triu_indices(N, k=1)
    A[iu] = np.fromiter((c == "1" for c in s), dtype=float, count=M)
    return A + A.T


def is_valid(s: str) -> bool:
    #This checks if the length of the string is the exact match of the denoted M
    #This also checks if the string is made up of 0's and 1's.
    return len(s) == M and set(s) <= {"0", "1"}


def neighbors_edge_flip(obj: str):
    # This function flips one bit of the bitstring
    s = list(obj)
    for k in range(len(s)):
        s[k] = "1" if s[k] == "0" else "0"
        yield "".join(s)
        s[k] = "1" if s[k] == "0" else "0"


#non-trivial conj: sqrt(N-1) + 1 - (lambda_1 + mu)  > 0 means counterexample
def reward_calc(obj: str) -> float:
    #Get rid of whitespace
    obj = obj.strip()
    if not is_valid(obj):
        return -1e6
    A = bitstring_to_adj(obj)
    G = nx.from_numpy_array(A)
    #If it is not connected, then add a panalty
    if not nx.is_connected(G):
        return -1000.0 - nx.number_connected_components(G)

    mu = len(nx.max_weight_matching(G, maxcardinality=True))
    lower_bound = np.sqrt(A.shape[0] - 1) + 1
    lam1 = float(np.linalg.eigvalsh(A)[-1])

    score = lower_bound - (lam1 + mu)
    return score


def is_counterexample(obj: str) -> bool:
    #CHANGED: use the tolerance instead of score > 0
    return reward_calc(obj) > TOL


def _flip(s: str, k: int) -> str:
    '''Flip bit k of the bitstring (add or delete one edge).'''
    return s[:k] + ("1" if s[k] == "0" else "0") + s[k+1:]


#CHANGED: v1 only flipped single bits, so it stopped as soon as no flip helped
#(at N=19 the best graph is a near-tree with 18 edges and every flip makes it worse).
#The swap pass below moves one edge at a time without changing the number of edges.
#CHANGED: default max_steps M -> 10*M so search from dense random graphs can finish at N=19
def greedy_search_from_startpoint(db, obj: str, max_steps=10 * M,
                                  mode="two_flip", pair_samples=2000):
    '''
    mode="one_flip": single edge flips only (classic hill climb, weak here)
    mode="two_flip": single flips, then pairs of flips (stronger, slower)
    pair_samples: how many random pairs to try per stalled step (None = all)
    '''

    current = obj.strip()
    if not is_valid(current):
        current = empty_starting_point()
    current_score = reward_calc(current)

    for _ in range(max_steps):
        improved = False

        # --- every mode: single flip ---
        order = list(range(M))
        random.shuffle(order)
        for k in order:
            cand = _flip(current, k)
            sc = reward_calc(cand)
            if sc > current_score:
                current, current_score = cand, sc
                improved = True
                break
        if improved:
            continue

        if mode == "one_flip":        # nothing else to try -> local optimum
            break

        # --- two_flip mode: pairs of flips (adds the swap passes and more) ---
        if pair_samples is None:
            pairs = [(i, j) for i in range(M) for j in range(i + 1, M)]
            random.shuffle(pairs)
        else:
            pairs = (tuple(random.sample(range(M), 2)) for _ in range(pair_samples))

        for i, j in pairs:
            s = list(current)
            s[i] = "1" if s[i] == "0" else "0"
            s[j] = "1" if s[j] == "0" else "0"
            cand = "".join(s)
            if not nx.is_connected(nx.from_numpy_array(bitstring_to_adj(cand))):
                continue
            sc = reward_calc(cand)
            if sc > current_score:
                current, current_score = cand, sc
                improved = True
                break

        if not improved:
            break

    return [current]


#Keep the edge swap
def count_improving_swaps(obj: str, limit: int = None):
    '''
    Number of swaps from obj that improve the score, and how many break connectivity.
    A standalone check, so you can see whether swapping is worth anything on db[0].
    '''
    obj = obj.strip()
    base = reward_calc(obj)
    ones = [i for i in range(M) if obj[i] == "1"]
    zeros = [i for i in range(M) if obj[i] == "0"]
    checked = improving = disconnected = 0
    for off in ones:
        for on in zeros:
            if limit is not None and checked >= limit:
                return improving, disconnected, checked, base
            s = list(obj)
            s[off] = "0"
            s[on] = "1"
            cand = "".join(s)
            checked += 1
            A = bitstring_to_adj(cand)
            if not nx.is_connected(nx.from_numpy_array(A)):
                disconnected += 1
                continue
            if reward_calc(cand) > base:
                improving += 1
    return improving, disconnected, checked, base


def find_matching_number(adj_matrix):
    graph = nx.Graph(adj_matrix)
    matching = nx.max_weight_matching(graph, maxcardinality=True)
    return len(matching)


def find_largest_eigenvalue(adj_matrix):
    adj_matrix = np.array(adj_matrix)
    eigenvalues = np.linalg.eigvals(adj_matrix)
    return np.max(np.abs(eigenvalues))


#CHANGED: tests/prints only run when this file is executed directly, not on import
if __name__ == "__main__":
    test_empty_starting_point()
    for neighbor in neighbors_edge_flip("001"):
        print(neighbor)
    #Sanity check: the star should score ~0 (the conjecture's tight case)
    star = "1" * (N - 1) + "0" * (M - (N - 1))
    print("star reward:", reward_calc(star))







def random_tree_bits():
    '''Random spanning tree on N vertices, as a bitstring.'''
    A = np.zeros((N, N), dtype=int)
    order = list(range(N))
    random.shuffle(order)
    for k in range(1, N):
        #attach each new vertex to a random vertex already in the tree
        u, v = order[k], random.choice(order[:k])
        A[u, v] = A[v, u] = 1
    iu = np.triu_indices(N, k=1)   #same bit order as bitstring_to_adj
    return "".join(str(x) for x in A[iu])




def random_double_star():
    '''Two hubs joined by a bridge, remaining vertices split as leaves.'''
    A = np.zeros((N, N), dtype=int)
    verts = list(range(N))
    random.shuffle(verts)
    h1, h2 = verts[0], verts[1]          # the two hubs
    A[h1, h2] = A[h2, h1] = 1            # the bridge between them
    for v in verts[2:]:
        u = random.choice([h1, h2])      # attach each remaining vertex to a hub
        A[u, v] = A[v, u] = 1
    return "".join(str(x) for x in A[np.triu_indices(N, k=1)])




def random_graph_bits(p=None, connected=True, max_tries=100):
    '''Erdos-Renyi G(N, p) as a bitstring. p is drawn at random if not given.'''
    for _ in range(max_tries):
        q = p if p is not None else random.uniform(0.1, 0.4)
        s = "".join("1" if random.random() < q else "0" for _ in range(M))
        if not connected or nx.is_connected(nx.from_numpy_array(bitstring_to_adj(s))):
            return s
    return s   # fallback: may be disconnected; local search will still fix it
