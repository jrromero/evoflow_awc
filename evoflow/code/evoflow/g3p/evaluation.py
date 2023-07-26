# -*- coding: utf-8 -*-

#    Rafael Barbudo Lunar, PhD Student
#    Knowledge and Discovery Systems (KDIS)
#    University of Cordoba, Spain

import numpy as np
from multiprocessing import Process, Manager
from os.path import join
import time
import warnings

from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def evaluate(ind, timeout, measure, X, y, cv, outdir, seed):

    start_time = time.time()
    fitness, predictions = _evaluate_cv(ind, measure, X, y, cv, timeout, seed)
    elapsed = time.time() - start_time

    with open(join(outdir, "individuals.tsv"), 'a') as f:

        if isinstance(fitness, str):
            f.write(str(ind) + '\t' + fitness + '\t' + fitness + '\n')
            return (0.0,), None, None
        else:
            #f.write(str(ind) + '\t' + str(fitness) + '\t' + str(elapsed) + '\t' + ",".join(str(v) for v in predictions.tolist()) + '\n')
            f.write(str(ind) + '\t' + str(fitness) + '\t' + str(elapsed) + '\n')
            return (fitness,), predictions, elapsed

"""
def _evaluate_cv(ind, measure, X, y, cv, timeout, seed, memory_limit=3072):
    mng = Manager()
    # Check for pipelines with duplicated operators
    pipe_str = str(ind).split(';')
    ops_str = [op[:op.index('(')] for op in pipe_str]
    if len(ops_str) != len(set(ops_str)):
        return "invalid_ind", None

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=seed)

    if timeout is not None:
        fold_timeout = int(timeout/cv)

    # Create a new process to control the timeout
    return_dict = mng.dict()
    p = Process(target=_fit_predict, args=(ind, x_train, y_train, x_test, return_dict, memory_limit))
    p.start()
    p.join(fold_timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return "time_err", None

    elif isinstance(return_dict['y_pred'], str):
        return return_dict['y_pred'], None

    y_pred = return_dict['y_pred']            

    # Compute the fitness
    fitness = measure(y_test, y_pred)

    # Return the evaluated individual
    return fitness, y_pred
"""


def _evaluate_cv(ind, measure, X, y, cv, timeout, seed, memory_limit=3072):

    # Check for pipelines with duplicated operators
    pipe_str = str(ind).split(';')
    ops_str = [op[:op.index('(')] for op in pipe_str]
    if len(ops_str) != len(set(ops_str)):
        return "invalid_ind", None

    # Generate the folds
    k_folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    
    # Check whether there is a timeout for an evaluation
    if timeout is not None:
        fold_timeout = int(timeout/cv)

    # Train and test the model for each fold
    fitneses = []
    predictions = np.empty(0)
    mng = Manager()

    for train_index, test_index in k_folds.split(X, y):

        # Get the train and test sets
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if timeout is None:

            try:
                if len(ind.pipeline) == 1: # only a classifier
                    ind.pipeline.fit(x_train, y_train)
                else:
                    # apply only the preprocessing
                    for step in ind.pipeline[:-1]:

                        if getattr(step, "fit_transform", None) is not None:
                            x_train = step.fit_transform(x_train, y_train)
                        elif getattr(step, "fit_resample", None) is not None:
                            x_train, y_train = step.fit_resample(x_train, y_train)

                    # raise an exception if the new dataset is too large
                    if x_train.nbytes/1024**2 > memory_limit:
                        raise MemoryError

                    ind.pipeline[-1].fit(x_train, y_train)

                y_pred = ind.pipeline.predict(x_test)

            except ValueError:
                return "invalid_ind", None
            except MemoryError:
                return "mem_err", None
        else:
            # Create a new process to control the timeout
            return_dict = mng.dict()
            p = Process(target=_fit_predict, args=(ind, x_train, y_train, x_test, return_dict, memory_limit))
            p.start()
            p.join(fold_timeout)

            if p.is_alive():
                p.terminate()
                p.join()
                return "time_err", None

            elif isinstance(return_dict['y_pred'], str):
                return return_dict['y_pred'], None

            y_pred = return_dict['y_pred']            

        # Store the fitness of the fold
        predictions = np.concatenate([predictions, y_pred])
        fitness = measure(y_test, y_pred)
        fitneses.append(fitness)

    # Return the evaluated individual
    fitness = sum(fitneses)/len(fitneses)
    return fitness, predictions



def _fit_predict(ind, x_train, y_train, x_test, ret_dict, memory_limit):

    try:
        if len(ind.pipeline) == 1:
            ind.pipeline.fit(x_train, y_train)
        else:
            for step in ind.pipeline[:-1]:

                if getattr(step, "fit_transform", None) is not None:
                    x_train = step.fit_transform(x_train, y_train)
                elif getattr(step, "fit_resample", None) is not None:
                    x_train, y_train = step.fit_resample(x_train, y_train)

            if x_train.nbytes/1024**2 > memory_limit:
                raise MemoryError

            ind.pipeline[-1].fit(x_train, y_train)

        ret_dict['y_pred'] = ind.pipeline.predict(x_test)

    except ValueError:
        ret_dict['y_pred'] = "invalid_ind"
    except MemoryError:
        ret_dict['y_pred'] = "mem_err"
    except np.linalg.LinAlgError:
        ret_dict['y_pred'] = "invalid_ind"
