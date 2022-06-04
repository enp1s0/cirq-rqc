import cirq
import cirq.experiments.google_v2_supremacy_circuit as supremacy_v2
import cirq.contrib.quimb as ccq
import quimb.tensor as qtn
import random
import numpy as np
import random
import argparse
import cotengra as ctg
import optuna

# Cotengra optimizer
random.seed(0)
opt = ctg.ReusableHyperOptimizer(
    max_repeats=16,
    reconf_opts={},
    parallel=False,
    progbar=True,
    optlib='optuna',
    sampler=optuna.samplers.RandomSampler(0)
)

lattice_dim_row = 7
lattice_dim_col = 8
qubits = lattice_dim_row * lattice_dim_col

num_blocks = 3

block_length = 8
depth = 20
#depth = block_length * num_blocks

optimize = opt
backend = 'cupy'

parser = argparse.ArgumentParser(description='Quimb RQC')
parser.add_argument('--compute_type', type=str, default='complex64', help='Data type for computation')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--num_samplings', type=int, default=1, help='Random seed')
args = parser.parse_args()

num_samplings = args.num_samplings

print("# lattice : ", lattice_dim_col, "x", lattice_dim_row)
print("# num qubits : ", qubits)
print("# depth : ", depth)
print("# num_samplings : ", num_samplings)
print("# optimize : ", optimize)
print("# backend : ", backend)
print("# sampling_seed : ", args.seed)

np.set_printoptions(threshold=np.inf)

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

tn_simplified = tn.rank_simplify()

fig = tn.graph(color=['PSI0', 'H', 'X', 'Y', 'T', 'CZ'], return_fig=True)
fig.savefig('result.pdf')

for i in range(num_samplings):
    random.seed(i * 123867 + args.seed)
    bitstring = "".join(random.choice('01') for _ in range(qubits))
    psi_sample = qtn.MPS_computational_state(bitstring, tags='PSI_f').squeeze()

    open_edges = tn.all_inds()[-qubits:]
    end_edges = psi_sample.all_inds()
    rename_table = {end_edges[i] : open_edges[i] for i in range(len(end_edges))}
    psi_sample.reindex_(rename_table)

    circ_tn = tn_simplified & psi_sample
    circ_tn.astype_('complex64')

    width = circ_tn.contraction_width(optimize=optimize)
    amplitude = circ_tn.contract(all, optimize=optimize, backend=backend)
    prob = np.power(np.absolute(amplitude), 2) * np.power(2., qubits)

    print("[", i, "] bs=", bitstring, " width=", width, ",amplitude=", amplitude, ", prob=", prob)
