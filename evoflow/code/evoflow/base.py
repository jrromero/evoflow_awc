# -*- coding: utf-8 -*-

#    Rafael Barbudo Lunar, PhD Student
#    Knowledge and Discovery Systems (KDIS)
#    University of Cordoba, Spain

import numpy as np
from os.path import join, exists
import pathlib
import random as rand
import shutil
import sys
import time

from deap import base, creator, tools

from sklearn.metrics import balanced_accuracy_score

from evoflow.g3p.crossover import cx_multi
from evoflow.g3p.encoding import SyntaxTreeSchema, SyntaxTreePipeline
from evoflow.g3p.evaluation import evaluate
from evoflow.g3p.grammar import parse_pipe_grammar
from evoflow.g3p.mutation import mut_multi
from evoflow.g3p.support import DiverseElite


class EvoFlow:

    """ Class responsible for optimizing the machine learning pipelines """

    def __init__(self, grammar, pop_size=100, generations=100,
                 fitness=balanced_accuracy_score, nderiv=13,
                 crossover=cx_multi, mutation=mut_multi, cxpb=0.8,
                 mutpb=0.2, elite_size=10, timeout=3600,
                 eval_timeout=360, seed=None, outdir=None):

        # Set the parameters for training
        self.grammar = grammar
        self.pop_size = pop_size
        self.generations = generations
        self.fitness = fitness
        self.nderiv = nderiv
        self.crossover = getattr(sys.modules["evoflow.g3p.crossover"], crossover)
        self.mutation = getattr(sys.modules["evoflow.g3p.mutation"], mutation)
        print(self.crossover)
        print(self.mutation)
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.elite_size = elite_size
        self.timeout = timeout
        self.eval_timeout = eval_timeout
        self.seed = seed
        self.outdir = outdir

        # Create the logging file for the individual
        if exists(self.outdir):
            shutil.rmtree(self.outdir)
        pathlib.Path(join(self.outdir)).mkdir(parents=True, exist_ok=True)
        with open(join(self.outdir, "individuals.tsv"), 'a') as f:
            f.write("pipeline\tfitness\tfit_time\n")

        # Configure the G3P logging
        stat_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stat_size = tools.Statistics(lambda ind: len(ind.pipeline))
        stats = tools.MultiStatistics(fitness=stat_fit, size=stat_size)
        stats.register("min     ", np.min)  # For better visualization
        stats.register("max     ", np.max)
        stats.register("avg     ", np.mean)
        stats.register("std     ", np.std)
        self.stats = stats

        # Load the grammar
        root, terms, non_terms, self.pset, terms_families = parse_pipe_grammar(grammar, seed)
        self.schema = SyntaxTreeSchema(nderiv, root, terms, non_terms, self.pset, terms_families)

        # Configure the creator (from DEAP)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", SyntaxTreePipeline, fitness=creator.FitnessMax)

        # Configure the toolbox (from DEAP)
        toolbox = base.Toolbox()
        toolbox.register("expr", self.schema.createSyntaxTree)
        toolbox.register("ind", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.ind)
        toolbox.register("select", tools.selTournament, tournsize=2)
        toolbox.register("mate", self.crossover, schema=self.schema)
        toolbox.register("mutate", self.mutation, schema=self.schema)
        self.toolbox = toolbox        


    def fit(self, X, y):

        # Create the evolutionary process
        start_time = time.time()
        self._evolve(X, y)

        # Save the final execution time
        exec_time = str(time.time() - start_time)
        print("--- " + exec_time + " seconds ---")
        with open(join(self.outdir, "evolution.txt"), "a") as log:
            log.write("--- " + exec_time + " sec ---\n")

        # Stop the execution if there are not individuals in the elite
        if self.elite is None or len(self.elite) == 0:
            print('ERR: The elite is empty.')
            sys.exit()


    def _reset_ind(self, ind):

        del ind.fitness.values
        del ind.pipeline


    def _varAnd(self, population):

        # Clone the parents
        offspring = [self.toolbox.clone(ind) for ind in population]

        # Apply the crossover
        for i in range(1, len(offspring), 2):
            if rand.random() < self.cxpb:
                offspring[i - 1], offspring[i], modified = self.toolbox.mate(offspring[i-1], offspring[i])
                if modified:
                    self._reset_ind(offspring[i-1])
                    self._reset_ind(offspring[i])

        # Apply the mutation
        for i, _ in enumerate(offspring):
            if rand.random() < self.mutpb:
                offspring[i], modified = self.toolbox.mutate(offspring[i])
                # Properties may do not exist if they have been deleted during crossover
                if modified and hasattr(offspring[i], "pipeline"):
                    self._reset_ind(offspring[i])

        return offspring


    def _evolve(self, X, y):

        # To control the timeout
        start = time.time()

        # Set the seed if it is not None
        if self.seed is not None:
            rand.seed(self.seed)
            np.random.seed(self.seed)

        # Configure the rest of the evolutionary algorithm
        self.toolbox.register("evaluate", evaluate, measure=self.fitness, 
                              seed=self.seed, X=X, y=y, outdir=self.outdir, cv=5)

        # Configure additional logging
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (self.stats.fields if self.stats else [])
        for category in self.stats.fields:
            logbook.chapters[category].header = self.stats[category].fields

        # Elite with the best individuals
        self.elite = DiverseElite(self.elite_size, div_weight=0.2)
        print("div_weight is 0.2")

        # Create the initial population
        population = self.toolbox.population(n=self.pop_size)

        # Evaluate the inivial population
        for ind in population:
            ind.create_sklearn_pipeline(self.pset)
            if (time.time() - start) < self.timeout:
                ind.fitness.values, ind.prediction, ind.runtime = self.toolbox.evaluate(ind, self.eval_timeout)
            else:
                ind.fitness.values = (0,)

        # Update the elite
        pop_valid = [ind for ind in population if ind.fitness.values != (0,)]
        self.elite.update(pop_valid)

        # Append the current generation statistics to the logbook
        record = self.stats.compile(population) if self.stats else {}
        logbook.record(gen=0, nevals=len(population), **record)
        report = logbook.stream
        print(report)
        with open(join(self.outdir, "evolution.txt"), "a") as log:
            log.write(report + "\n")

        # Begin the generational process
        for gen in range(1, self.generations + 1):

            # Stop the evaluation if timeout has been reached
            if (time.time() - start) > self.timeout:
                return

            # Apply the genetic operators
            offspring = self.toolbox.select(population, self.pop_size)
            offspring = self._varAnd(offspring)

            # Evaluate the offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]                
            for ind in invalid_ind:
                ind.create_sklearn_pipeline(self.pset)
                if (time.time() - start) < self.timeout:
                    ind.fitness.values, ind.prediction, ind.runtime = self.toolbox.evaluate(ind, self.eval_timeout)
                else:
                    ind.fitness.values = (0,)

            # Update the elite
            off_valid = [ind for ind in offspring if ind.fitness.values != (0,)]
            self.elite.update(off_valid)

            # Replace the current population ensuring that the best individual is kept
            off_fits = [ind.fitness.values[0] for ind in offspring]
            idx_min = off_fits.index(min(off_fits))
            if self.elite.best_ind() not in offspring:
                offspring[idx_min] = self.elite.best_ind()
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = self.stats.compile(population) if self.stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            report = logbook.stream
            print(report)
            with open(join(self.outdir, "evolution.txt"), "a") as log:
                log.write(report + "\n")
