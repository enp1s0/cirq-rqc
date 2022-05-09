import cirq
import cirq.experiments.google_v2_supremacy_circuit as supremacy_v2
import cirq.contrib.quimb as ccq
import quimb.tensor as qtn
import random

lattice_dim = 10
qubits = lattice_dim * lattice_dim

num_blocks = 5

block_length = 8
depth = block_length * num_blocks

num_samplings = 1

optimize = 'greedy'
backend = 'cupy'

circuit = supremacy_v2.generate_boixo_2018_supremacy_circuits_v2_grid(
        n_rows=lattice_dim,
        n_cols=lattice_dim,
        cz_depth=depth,
        seed=0
        )

circuit = cirq.drop_empty_moments(circuit)

tensors, qubit_frontier, fix = ccq.circuit_to_tensors(
        circuit,
        )
print("qubit_frontier", qubit_frontier)
print("fix", fix)

tn = qtn.TensorNetwork(tensors)

for s in range(num_samplings):
    bitstring = "".join(random.choice('01') for _ in range(qubits))
    psi_sample = qtn.MPS_computational_state(bitstring, tags='PSI_f').squeeze()
    open_edges = tn.all_inds()[-qubits:]
    end_edges = psi_sample.all_inds()
    rename_table = {end_edges[i] : open_edges[i] for i in range(len(end_edges))}
    psi_sample.reindex_(rename_table)
    circ_tn = tn & psi_sample
    print(circ_tn)
    #print(circ_tn)
    circ_tn = circ_tn.rank_simplify()
    circ_tn.astype_('complex64')

    width = circ_tn.contraction_width(optimize=optimize)

    prob = circ_tn.contract(all, optimize=optimize, backend=backend)
    print("width = ", width, ",prob = ", prob)

    #import matplotlib.pyplot as plt
    #psi_sample.graph()
    #plt.savefig('figure.pdf', bbox_inches='tight')
