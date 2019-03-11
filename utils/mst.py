import numpy as np

def find_cycles(heads):
    """
    Finds cycles in list of heads.
    Index in list determines the index of vertex. Value determines the head vertex index.
    Returns list of cycles as ordered list of cycle vertices indices.
    
    Eg.
    Heads 0 0 3 5 1 2
    Index 0 1 2 3 4 5
    
    There is one cycle:      2 -> 3 -> 5 -> 2
    so the function returns: [[2, 3, 5]]
    """
    cycles = []
    
    closed_ = np.full(heads.shape, False)
    closed_[0] = True # head is closed
    
    for i, head in enumerate(heads):
        if closed_[i]:
            continue
        
        open_ = []
        
        while not closed_[i]:
            if i in open_:
                cycle = list(reversed(open_[:open_.index(i) + 1]))
                cycles.append(cycle)
                break
            else:
                open_.insert(0, i)
                i = head
                head = heads[head]
            
        for i in open_:
            closed_[i] = True
            
    return cycles

def mst(x, enforce_single_head=True):
    """
    Chu, Liu & Edmonds Maximum Spanning Tree implementation.

    x is square matrix of head scores for each vertex.
    Returns a spanning tree maximizing the score
    in form of the list of heads for all vertices.
    """
    rows, cols = x.shape

    x = np.concatenate([np.zeros(rows).reshape(rows, 1), x[:, 1:]], axis=1)
    
    guard = np.zeros(x.shape)
    np.fill_diagonal(guard, np.inf)
    guard[0, 0] = 0

    if enforce_single_head:
        max_prob_head = x[0].argmax()
        guard[0, 1:max_prob_head] = np.inf
        guard[0, (max_prob_head + 1):] = np.inf
    
    x = x - guard
    x = x - x.max(axis=0)
    
    heads = x.argmax(axis=0)
    cycles = find_cycles(heads)
    
    if cycles:
        vtx_in_cycles = sum(cycles, [])
        vtx_no_cycles = sorted(v for v in range(rows) if v not in vtx_in_cycles)
        vtx_no_cycles_num = len(vtx_no_cycles)

        x_cont = x[vtx_no_cycles,:][:,vtx_no_cycles]
        x_cont = np.pad(x_cont, ((0, len(cycles)), (0, len(cycles))), 'constant', constant_values=0)

        for i_c, cycle in enumerate(cycles):
            i_c = i_c + len(vtx_no_cycles)
            for i_v, v in enumerate(vtx_no_cycles):
                x_cont[i_c, i_v] = max(x[cycle, v])
                x_cont[i_v, i_c] = max(x[v, cycle])

        heads_cont = mst(x_cont)
        
        heads.fill(0)
        for v_cont, h_cont in enumerate(heads_cont):
            if v_cont == 0:
                # head stays with 0 - skip it
                continue
            
            is_v_in_cycle = (v_cont >= vtx_no_cycles_num)
            is_h_in_cycle = (h_cont >= vtx_no_cycles_num)
            
            if is_v_in_cycle:
                v_cycle = cycles[v_cont - vtx_no_cycles_num]
                
                # 1. add whole cycle 
                for i in range(len(v_cycle)):
                    heads[v_cycle[i]] = v_cycle[(i + 1) % len(v_cycle)]
                    
                # 2. modify highest score vertex
                if is_h_in_cycle:
                    h_cycle = cycles[h_cont - vtx_no_cycles_num]
                    
                    max_ = -np.inf
                    for v_cycle_v in v_cycle:
                        for h_cycle_v in h_cycle:
                            if x[h_cycle_v, v_cycle_v] > max_:
                                max_ = x[h_cycle_v, v_cycle_v]
                                v = v_cycle_v
                                h = h_cycle_v
                    
                else:
                    h = vtx_no_cycles[h_cont]
                    v = max(v_cycle, key=lambda v_cycle_v: x[h, v_cycle_v])
            else:
                v = vtx_no_cycles[v_cont]
                if is_h_in_cycle:
                    h_cycle = cycles[h_cont - vtx_no_cycles_num]
                    # choose vertex from cycle with highest score as head
                    h = max(h_cycle, key=lambda h_cycle_v: x[h_cycle_v, v])
                else:
                    h = vtx_no_cycles[h_cont]
                
            heads[v] = h
            
    # there are no cycles in returned heads
    return heads
