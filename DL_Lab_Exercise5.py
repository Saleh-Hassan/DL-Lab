import pickle
import numpy as np
import copy
import matplotlib.pyplot as plt
import sys
import time

from robo.fmin import bayesian_optimization
np.random.seed(100)

rf = pickle.load(open("./rf_surrogate_cnn.pkl", "rb"))
cost_rf = pickle.load(open("./rf_cost_surrogate_cnn.pkl", "rb"))
lower = np.array([-6,32,4,4,4])
upper = np.array([0,512,10,10,10])

def objective_function(x, epoch=40):
    """
        Function wrapper to approximate the validation error of the hyperparameter configurations x by the prediction of a surrogate regression model,
        which was trained on the validation error of randomly sampled hyperparameter configurations.
        The original surrogate predicts the validation error after a given epoch. Since all hyperparameter configurations were trained for a total amount of
        40 epochs, we will query the performance after epoch 40.
    """
    # Normalize all hyperparameter to be in [0, 1]
    x_norm = copy.deepcopy(x)
    x_norm[0] = (x[0] - (-6)) / (0 - (-6))
    x_norm[1] = (x[1] - 32) / (512 - 32)
    x_norm[2] = (x[2] - 4) / (10 - 4)
    x_norm[3] = (x[3] - 4) / (10 - 4)
    x_norm[4] = (x[4] - 4) / (10 - 4)

    x_norm = np.append(x_norm, epoch)
    y = rf.predict(x_norm[None, :])[0]

    return y


def runtime(x, epoch=40):
    """
        Function wrapper to approximate the runtime of the hyperparameter configurations x.
    """

    # Normalize all hyperparameter to be in [0, 1]
    x_norm = copy.deepcopy(x)
    x_norm[0] = (x[0] - (-6)) / (0 - (-6))
    x_norm[1] = (x[1] - 32) / (512 - 32)
    x_norm[2] = (x[2] - 4) / (10 - 4)
    x_norm[3] = (x[3] - 4) / (10 - 4)
    x_norm[4] = (x[4] - 4) / (10 - 4)

    x_norm = np.append(x_norm, epoch)
    y = cost_rf.predict(x_norm[None, :])[0]

    return y


def get_random_hyperparameters():
    return [np.random.uniform(l, u) for l, u in zip(lower, upper)]

def bayesian_opt():
    bayesian_incumbent_list = []
    bayesian_incumbent_mean_list = []
    bayesian_runtime = []
    for i in range(0, 10):
        output_dict = bayesian_optimization(objective_function, lower, upper, num_iterations=50)
        bayesian_incumbent_mean_list.append(np.mean(output_dict['incumbent_values']))
        bayesian_incumbent_list.append(output_dict['incumbent_values'])
        bayesian_runtime.append(output_dict['runtime'])

    return np.mean(bayesian_incumbent_list, axis=0), np.cumsum(np.mean(bayesian_runtime, axis=0), axis=0), bayesian_incumbent_mean_list


def random_search_optimization():
    random_incumbent_list = []
    random_runtime_list = []
    random_incumbent_mean_list = []
    for i in range(0, 10):
        runtime_random = []
        incumbent = []
        current_best = sys.maxsize
        for j in range(0, 50):
            current_prediction = objective_function(get_random_hyperparameters())
            runtime_random.append(runtime(get_random_hyperparameters()))
            if current_prediction < current_best:
                current_best = current_prediction
            incumbent.append(current_best)
        random_incumbent_mean_list.append(np.mean(incumbent))
        random_runtime_list.append(runtime_random)
        random_incumbent_list.append(incumbent)

    return np.mean(random_incumbent_list, axis=0), np.cumsum(np.mean(random_runtime_list, axis=0), axis=0), random_incumbent_mean_list



def plot_graph_and_save(rs_opt, b_opt, xlabel_value, ylabel_value, output_file_name):
    plt.plot(range(len(rs_opt)), rs_opt, 'k--', color="blue", linewidth=2, label="Random Search")
    plt.plot(range(len(b_opt)), b_opt, 'k', color="black", linewidth=2, label="Bayesian Optimization")
    plt.legend(loc='upper right', frameon=False)
    plt.xlabel(xlabel_value)
    plt.ylabel(ylabel_value)
    plt.savefig(output_file_name)
    plt.show()

rs_validation_error, rs_runtime, _ = random_search_optimization()
bo_validation_error, bo_runtime, _ = bayesian_opt()
plot_graph_and_save(rs_validation_error, bo_validation_error, "# of Iterations", "Validation Error", "performance.png")
plot_graph_and_save(rs_runtime, bo_runtime, "# of Iterations", "Runtime", "runtime.png")