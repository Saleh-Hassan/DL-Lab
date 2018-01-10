import argparse
import logging
import pickle
import ConfigSpace as CS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
logging.basicConfig(level=logging.ERROR)
from copy import deepcopy
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
import hpbandster.distributed.utils
from hpbandster.distributed.worker import Worker


def create_config_space():
    cs = CS.ConfigurationSpace()
    adam_final_lr_fraction = CS.UniformFloatHyperparameter('Adam_final_lr_fraction',
                                                           lower=1e-4,
                                                           upper=1.0,
                                                           default_value=1e-2,
                                                           log=True)

    cs.add_hyperparameter(adam_final_lr_fraction)

    adam_initial_lr = CS.UniformFloatHyperparameter('Adam_initial_lr',
                                                    lower=1e-4,
                                                    upper=1e-2,
                                                    default_value=1e-3,
                                                    log=True)

    cs.add_hyperparameter(adam_initial_lr)

    sgd_final_lr_fraction = CS.UniformFloatHyperparameter('SGD_final_lr_fraction',
                                                          lower=1e-4,
                                                          upper=1.0,
                                                          default_value=1e-2,
                                                          log=True)

    cs.add_hyperparameter(sgd_final_lr_fraction)

    sgd_initial_lr = CS.UniformFloatHyperparameter('SGD_initial_lr',
                                                   lower=1e-3,
                                                   upper=0.5,
                                                   default_value=1e-1,
                                                   log=True)

    cs.add_hyperparameter(sgd_initial_lr)

    sgd_momentum = CS.UniformFloatHyperparameter('SGD_momentum',
                                                 lower=0.0,
                                                 upper=0.99,
                                                 default_value=0.9,
                                                 log=False)

    cs.add_hyperparameter(sgd_momentum)

    step_decay_epochs_per_step = CS.UniformIntegerHyperparameter('StepDecay_epochs_per_step',
                                                                lower=1,
                                                                default_value=16,
                                                                upper=128)

    cs.add_hyperparameter(step_decay_epochs_per_step)

    batch_size = CS.UniformIntegerHyperparameter('batch_size',
                                                 lower=8,
                                                 default_value=16,
                                                 upper=256)

    cs.add_hyperparameter(batch_size)

    dropout_0 = CS.UniformFloatHyperparameter('dropout_0',
                                              lower=0.0,
                                              upper=0.5,
                                              default_value=0.0,
                                              log=False)

    cs.add_hyperparameter(dropout_0)

    dropout_1 = CS.UniformFloatHyperparameter('dropout_1',
                                              lower=0.0,
                                              upper=0.5,
                                              default_value=0.0,
                                              log=False)

    cs.add_hyperparameter(dropout_1)

    dropout_2 = CS.UniformFloatHyperparameter('dropout_2',
                                              lower=0.0,
                                              upper=0.5,
                                              default_value=0.0,
                                              log=False)

    cs.add_hyperparameter(dropout_2)

    dropout_3 = CS.UniformFloatHyperparameter('dropout_3',
                                              lower=0.0,
                                              upper=0.5,
                                              default_value=0.0,
                                              log=False)

    cs.add_hyperparameter(dropout_3)

    l2_reg_0 = CS.UniformFloatHyperparameter('l2_reg_0',
                                             lower=1e-6,
                                             upper=1e-2,
                                             default_value=1e-4,
                                             log=True)

    cs.add_hyperparameter(l2_reg_0)

    l2_reg_1 = CS.UniformFloatHyperparameter('l2_reg_1',
                                             lower=1e-6,
                                             upper=1e-2,
                                             default_value=1e-4,
                                             log=True)

    cs.add_hyperparameter(l2_reg_1)

    l2_reg_2 = CS.UniformFloatHyperparameter('l2_reg_2',
                                             lower=1e-6,
                                             upper=1e-2,
                                             default_value=1e-4,
                                             log=True)

    cs.add_hyperparameter(l2_reg_2)

    l2_reg_3 = CS.UniformFloatHyperparameter('l2_reg_3',
                                             lower=1e-6,
                                             upper=1e-2,
                                             default_value=1e-4,
                                             log=True)

    cs.add_hyperparameter(l2_reg_3)

    learning_rate_schedule = CS.CategoricalHyperparameter('learning_rate_schedule', ['ExponentialDecay', 'StepDecay'],
                                                          default_value='ExponentialDecay')

    cs.add_hyperparameter(learning_rate_schedule)

    activation = CS.CategoricalHyperparameter('activation', ['relu', 'tanh'],
                                              default_value='relu')

    cs.add_hyperparameter(activation)

    loss_function = CS.CategoricalHyperparameter('loss_function', ['categorical_crossentropy'],
                                                 default_value='categorical_crossentropy')
    cs.add_hyperparameter(loss_function)

    num_layers = CS.UniformIntegerHyperparameter('num_layers',
                                                 lower=1,
                                                 default_value=2,
                                                 upper=4)
    cs.add_hyperparameter(num_layers)

    num_units_0 = CS.UniformIntegerHyperparameter('num_units_0',
                                                  lower=16,
                                                  default_value=32,
                                                  upper=256)

    cs.add_hyperparameter(num_units_0)

    num_units_1 = CS.UniformIntegerHyperparameter('num_units_1',
                                                  lower=16,
                                                  default_value=32,
                                                  upper=256)

    cs.add_hyperparameter(num_units_1)

    num_units_2 = CS.UniformIntegerHyperparameter('num_units_2',
                                                  lower=16,
                                                  default_value=32,
                                                  upper=256)

    cs.add_hyperparameter(num_units_2)

    num_units_3 = CS.UniformIntegerHyperparameter('num_units_3',
                                                  lower=16,
                                                  default_value=32,
                                                  upper=256)

    cs.add_hyperparameter(num_units_3)

    optimizer = CS.CategoricalHyperparameter('optimizer',
                                             ['Adam', 'SGD'],
                                             default_value='Adam')

    cs.add_hyperparameter(optimizer)

    output_activation = CS.CategoricalHyperparameter('output_activation',
                                                     ['softmax'],
                                                     default_value='softmax')

    cs.add_hyperparameter(output_activation)

    cond = CS.EqualsCondition(adam_final_lr_fraction, optimizer, 'Adam')
    cs.add_condition(cond)

    cond = CS.EqualsCondition(adam_initial_lr, optimizer, 'Adam')
    cs.add_condition(cond)

    cond = CS.EqualsCondition(sgd_final_lr_fraction, optimizer, 'SGD')
    cs.add_condition(cond)

    cond = CS.EqualsCondition(sgd_initial_lr, optimizer, 'SGD')
    cs.add_condition(cond)

    cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
    cs.add_condition(cond)

    cond = CS.EqualsCondition(step_decay_epochs_per_step, learning_rate_schedule, 'StepDecay')
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(dropout_1, num_layers, 1)
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(dropout_2, num_layers, 2)
    cs.add_condition(cond)

    cond = CS.EqualsCondition(dropout_3, num_layers, 4)
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(l2_reg_1, num_layers, 1)
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(l2_reg_2, num_layers, 2)
    cs.add_condition(cond)

    cond = CS.EqualsCondition(l2_reg_3, num_layers, 4)
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(num_units_1, num_layers, 1)
    cs.add_condition(cond)

    cond = CS.GreaterThanCondition(num_units_2, num_layers, 2)
    cs.add_condition(cond)

    cond = CS.EqualsCondition(num_units_3, num_layers, 4)
    cs.add_condition(cond)

    return cs


def objective_function(config, epoch=127, **kwargs):
    # Cast the config to an array such that it can be forwarded to the surrogate
    x = deepcopy(config.get_array())
    x[np.isnan(x)] = -1
    lc = rf.predict(x[None, :])[0]
    c = cost_rf.predict(x[None, :])[0]

    return lc[epoch], {"cost": c, "learning_curve": lc[:epoch].tolist()}


def plot_graph_and_save(wall_clock_time_plt, incumbent_validation_error,
                        x_label_value, y_label_value, output_file_name, title):
    plt.plot(wall_clock_time_plt, incumbent_validation_error, 'k--', color="blue", linewidth=2)
    plt.legend(loc='upper right', frameon=False)
    plt.xlabel(x_label_value)
    plt.ylabel(y_label_value)
    plt.savefig(output_file_name)
    plt.title(title)
    plt.show()


class WorkerWrapper(Worker):
    def compute(self, config, budget, *args, **kwargs):
        cfg = CS.Configuration(cs, values=config)
        loss, info = objective_function(cfg, epoch=int(budget))

        return ({
            'loss': loss,
            'info': {"runtime": info["cost"],
                     "lc": info["learning_curve"]}
        })


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--run_smac', action='store_true')
    parser.add_argument('--run_hyperband', action='store_true')
    parser.add_argument('--n_iters', default=50, type=int)
    args = vars(parser.parse_args())

    n_iters = args['n_iters']

    cs = create_config_space()
    rf = pickle.load(open("./rf_surrogate_paramnet_mnist.pkl", "rb"))
    cost_rf = pickle.load(open("./rf_cost_surrogate_paramnet_mnist.pkl", "rb"))

    if args["run_smac"]:
        print('coming in smac')
        scenario = Scenario({"run_obj": "quality",
                             "runcount-limit": n_iters,
                             "cs": cs,
                             "deterministic": "true",
                             "output_dir": ""})

        smac = SMAC(scenario=scenario, tae_runner=objective_function)
        smac.optimize()

        # The following lines extract the incumbent strategy and the estimated wall-clock time of the optimization
        rh = smac.runhistory
        incumbents = []
        incumbent_performance = []
        inc = None
        inc_value = 1
        idx = 1
        t = smac.get_trajectory()

        wall_clock_time = []
        cum_time = 0
        for d in rh.data:
            cum_time += rh.data[d].additional_info["cost"]
            wall_clock_time.append(cum_time)
        for i in range(n_iters):

            if idx < len(t) and i == t[idx].ta_runs - 1:
                inc = t[idx].incumbent
                inc_value = t[idx].train_perf
                idx += 1

            incumbents.append(inc)
            incumbent_performance.append(inc_value)

        plot_graph_and_save(wall_clock_time, incumbent_performance , "Wall Clock Time", "Incumbent Validation Error",
                            'smac.png', 'SMAC')

        lc_smac = []
        x = np.arange(10)
        ys = [i + x + (i * x) ** 2 for i in range(50)]
        colors = cm.rainbow(np.linspace(0, 1, len(ys)))
        smac_lc_index = 0
        for d in rh.data:
            container = rh.data[d].additional_info["learning_curve"]
            lc_smac.append(container)
            plt.plot(range(len(container)), container, label='Curve ' + str(smac_lc_index + 1))
            smac_lc_index = smac_lc_index + 1

        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.savefig("smac_learning.png")
        plt.legend(loc='lower right')
        plt.title("SMAC Learning Curves")
        plt.show()

    if args["run_hyperband"]:
        print('coming in hyperband')
        nameserver, ns_port = hpbandster.distributed.utils.start_local_nameserver()

        # starting the worker in a separate thread
        w = WorkerWrapper(nameserver=nameserver, ns_port=ns_port)
        w.run(background=True)

        CG = hpbandster.config_generators.RandomSampling(cs)

        # instantiating Hyperband with some minimal configuration
        HB = hpbandster.HB_master.HpBandSter(
            config_generator=CG,
            run_id='0',
            eta=2,  # defines downsampling rate
            min_budget=1,  # minimum number of epochs / minimum budget
            max_budget=127,  # maximum number of epochs / maximum budget
            nameserver=nameserver,
            ns_port=ns_port,
            job_queue_sizes=(0, 1),
        )
        # runs one iteration if at least one worker is available
        res = HB.run(10, min_n_workers=1)
        # shutdown the worker and the dispatcher
        HB.shutdown(shutdown_workers=True)

        # extract incumbent trajectory and all evaluated learning curves
        traj = res.get_incumbent_trajectory()
        wall_clock_time = []
        cum_time = 0

        for c in traj["config_ids"]:
            cum_time += res.get_runs_by_id(c)[-1]["info"]["runtime"]
            wall_clock_time.append(cum_time)

        lc_hyperband = []
        hb_lc_index=0
        for r in res.get_all_runs():
            c = r["config_id"]
            temp_curve = res.get_runs_by_id(c)[-1]["info"]["lc"]
            lc_hyperband.append(temp_curve)
            plt.plot(range(len(temp_curve)), temp_curve, label='Curve '+str(hb_lc_index + 1))
            hb_lc_index = hb_lc_index + 1

        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.savefig("hb_learning.png")
        plt.legend(loc='lower right')
        plt.title("Hyperband Learning Curves")
        plt.show()

        incumbent_performance = traj["losses"]
        plot_graph_and_save(wall_clock_time, incumbent_performance, "Wall Clock Time", "Incumbent Validation Error",
                            'HB.png', 'Hyperband')

