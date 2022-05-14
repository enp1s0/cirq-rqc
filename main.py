import cirq
import cirq.experiments.google_v2_supremacy_circuit as supremacy_v2
import cirq.contrib.quimb as ccq
import quimb.tensor as qtn
import random
import numpy as np
import random
import cupy
from mpi4py import MPI

lattice_dim_row = 7
lattice_dim_col = 7
qubits = lattice_dim_row * lattice_dim_col

num_blocks = 3

block_length = 8
depth = 20
#depth = block_length * num_blocks

num_samplings = 10

optimize = 'greedy'
backend = 'cupy'


print("# lattice : ", lattice_dim_col, "x", lattice_dim_row)
print("# num qubits : ", qubits)
print("# depth : ", depth)
print("# num_samplings : ", num_samplings)
print("# optimize : ", optimize)
print("# backend : ", backend)

circuit = supremacy_v2.generate_boixo_2018_supremacy_circuits_v2_grid(
        n_rows=lattice_dim_row,
        n_cols=lattice_dim_col,
        cz_depth=depth,
        seed=0
        )

circuit = cirq.drop_empty_moments(circuit)

tensors, qubit_frontier, fix = ccq.circuit_to_tensors(
        circuit,
        )

tn = qtn.TensorNetwork(tensors)

comm = MPI.COMM_WORLD
random.seed(comm.Get_rank())
cupy.cuda.runtime.setDevice(comm.Get_rank())

samples = []
M = 10
for s in range(num_samplings * M):
    bitstring = "".join(random.choice('01') for _ in range(qubits))
    psi_sample = qtn.MPS_computational_state(bitstring, tags='PSI_f').squeeze()

    open_edges = tn.all_inds()[-qubits:]
    end_edges = psi_sample.all_inds()
    rename_table = {end_edges[i] : open_edges[i] for i in range(len(end_edges))}
    psi_sample.reindex_(rename_table)

    circ_tn = tn.rank_simplify() & psi_sample
    circ_tn.astype_('complex64')

    width = circ_tn.contraction_width(optimize=optimize)
    amplitude = circ_tn.contract(all, optimize=optimize, backend=backend)

    prob_tmp = (np.power(np.linalg.norm(amplitude), 2) * np.power(2, qubits, dtype=np.float64)) / M
    accept_prob = min(1, prob_tmp)

    accepted = False
    if random.random() < accept_prob:
        samples += [amplitude]
        accepted = True

    print("[", comm.Get_rank(), "] width = ", width, ",amplitude= ", amplitude, "accept_prob = ", accept_prob, "accepted = ", ("Yes" if accepted else "No"))

local_num_samplings = comm.allgather(len(samples))
total_num_samplings = np.sum(local_num_samplings)

recvbuf = np.zeros(total_num_samplings)
sep = [0] + [sum(local_num_samplings[:i]) for i in range(len(local_num_samplings) - 1)]
comm.Allgatherv(samples, [recvbuf, num_samplings * comm.Get_size(), sep, MPI.DOUBLE])
samples = recvbuf

if comm.Get_rank() == 0:
    print("Num sampled = ", len(samples), " / ", num_samplings)
    for b, a in samples:
        print("bitstring=", b, ", amplitude=", a)

    min_Np = -13
    max_Np = 8
    resolution_Np = 10
    prob_histogram = np.zeros((max_Np - min_Np + 1) * resolution_Np)

    for s in samples:
        _, amplitude = s
        prob = np.linalg.norm(amplitude) ** 2
        log_Np = np.log(prob * np.power(2, qubits, dtype=np.float64))

        x_index = int((log_Np - min_Np) * resolution_Np)

        if x_index > 0 and x_index < (max_Np - min_Np + 1) * resolution_Np:
            prob_histogram[x_index] += 1

    print("# num samples : ", len(samples), " / ", num_samplings)

    for i in range((max_Np - min_Np + 1) * resolution_Np):
        print("{:.2f},{:e}".format((i  / resolution_Np + min_Np), prob_histogram[i] / float(len(samples))))
