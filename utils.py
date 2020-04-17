import os
import pickle
import subprocess
import numpy as np
from tqdm import tqdm
from problems import TSP
import torch


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):
    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):
    with open(check_extension(filename), 'rb') as f:
        return np.array(pickle.load(f)).astype(np.float32)


def lkh_solve_tsp(graph):
    TSPLIB_FILE = '/dev/shm/tsplib.tsp'
    LKH_PAR_FILE = '/dev/shm/lkh.par'
    TOUR_FILE = '/dev/shm/lkh_tour'

    def make_tsplib_file(graph):
        head = [
            'NAME : tsp_problem', 'TYPE : TSP', f'DIMENSION : {len(graph)}',
            'EDGE_WEIGHT_TYPE : EUC_2D', 'NODE_COORD_SECTION'
        ]
        coords = [
            f'{i+1} {int(c[0] * 10000000 + 0.5)} {int(c[1] * 10000000 + 0.5)}'
            for i, c in enumerate(graph)
        ]
        lines = '\n'.join(head + coords + ['EOF\n'])
        with open(TSPLIB_FILE, 'w') as f:
            f.write(lines)
        return TSPLIB_FILE

    def make_lkh_par(tsplib_file_path):
        lines = [
            f'PROBLEM_FILE = {tsplib_file_path}',
            f'OUTPUT_TOUR_FILE = {TOUR_FILE}', 'RUNS = 1', 'SEED = 1234',
            'MAX_TRIALS = 10000', 'TRACE_LEVEL = 0\n'
        ]
        lines = '\n'.join(lines)
        with open(LKH_PAR_FILE, 'w') as f:
            f.write(lines)
        return LKH_PAR_FILE

    def read_tour(tour_file):
        with open(tour_file, 'r') as f:
            record = False
            tour = []
            for line in f:
                if record:
                    idx = int(line)
                    if idx == -1:
                        return tour
                    tour += [idx - 1]
                if line.startswith('TOUR_SECTION'):
                    record = True

    tsplib_file = make_tsplib_file(graph)
    lkh_par_file = make_lkh_par(tsplib_file)
    subprocess.run(['LKH/LKH', lkh_par_file], capture_output=True)
    tour = read_tour(TOUR_FILE)

    os.remove(TSPLIB_FILE)
    os.remove(LKH_PAR_FILE)
    os.remove(TOUR_FILE)

    return np.array(tour).astype(int)


def get_optimal_cost(graphs):
    tours = []
    for g in tqdm(graphs):
        tour = lkh_solve_tsp(g).reshape(1, -1)
        tours += [tour]
    tours = torch.from_numpy(np.concatenate(tours))
    graphs = torch.from_numpy(graphs).float()
    costs = TSP(graphs, tours)
    return costs
