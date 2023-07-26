# -*- coding: utf-8 -*-

#    Rafael Barbudo Lunar, PhD Student
#    Knowledge and Discovery Systems (KDIS)
#    University of Cordoba, Spain

"""
The :mod:`mutation` for Grammar Guided Genetic Programming (G3P).

This module provides the methods to mutate SyntaxTree.
"""

from copy import deepcopy
import random as rand
from evoflow.g3p.encoding import TerminalNode, NonTerminalNode


def rebuild_branch(ind, schema, start):

    # Make a copy just in case
    ind_bak = deepcopy(ind)

    # Get the subtree to mutate and its end
    tree_slice = ind.search_subtree(start)
    p0_branchEnd = tree_slice.stop

    # Get branch depth (to check maximum size)
    p0_branchDepth = sum(1 for node in ind if isinstance(node, NonTerminalNode))

    # Determine the maximum size to fill (to check maximum size)
    p0_swapBranch = sum(1 for node in ind[tree_slice] if isinstance(node, NonTerminalNode))

    # Get the symbol
    symbol = ind[start].symbol

    # Save the fragment at the right of the subtree
    aux = ind[p0_branchEnd:]

    # Remove the subtree and the fragment at its right
    del ind[start:]

    # Create the son (second fragment) controlling the number of derivations
    max_derivations = schema.maxDerivSize - p0_branchDepth + p0_swapBranch
    min_derivations = schema.minDerivations(symbol)
    try:
        derivations = rand.randint(min_derivations, max_derivations)
    except ValueError:
        return ind_bak
    schema.fillTreeBranch(ind, symbol, derivations)

    # Restore the fragment at the right of the subtree
    ind += aux
    return ind


def mut_multi(ind, schema):

    # get the string representing the parent
    par_str = str(ind)

    # apply one of the mutators
    randval = rand.random()
    if randval < 0.2:
        son = _mut_struct(ind, schema)
    else:
        son = _mut_hps(ind, schema)

    # check whether the new individual is equal
    if str(son) == par_str:
        return son, False
    return son, True


def _mut_hps(ind, schema):

    # get the start position of the classifier
    pos_classifier = [idx for idx, node in enumerate(ind) if node.symbol == "classifier"][0]

    # get the number of hyper-parameters for the preprocessing and the classifier
    num_prep_hps = sum([1 for node in ind[:pos_classifier] if "::" in node.symbol])
    num_class_hps = sum([1 for node in ind[pos_classifier:] if "::" in node.symbol])

    # compute the probability of changing one hyper parameter
    mutpb_class = 0.0
    mutpb_prep = 0.0

    if num_prep_hps > 0:
        mutpb_prep = 1.0/num_prep_hps

    if num_class_hps > 0:
        mutpb_class = 1.0/num_class_hps

    # mutate each hyper-parameter with its corresponding probability
    for idx, node in enumerate(ind):

        if idx < pos_classifier:
            mutpb = mutpb_prep
        else:
            mutpb = mutpb_class

        if "::" in node.symbol and rand.random() < mutpb:
            term = schema.terminals_map.get(node.symbol)
            ind[idx] = TerminalNode(node.symbol, term.code())

    # return the individual after modifying it
    return ind


def _mut_struct(ind, schema):
    
    # we want to mutate high level non terminals
    target_symbols = ['workflow', 'preprocessingBranch']
    # get the possible start points to start the mutation
    possible_starts = [index for index, node in enumerate(ind) if node.symbol in target_symbols]
    # get one start point at random
    start = rand.choice(possible_starts)
    # perform the mutation
    return rebuild_branch(ind, schema, start)


def mut_branch(ind, schema):

    # get the string representing the parent
    par_str = str(ind)
    # get the possible start points to start the mutation
    possible_starts = [index for index, node in enumerate(ind) if isinstance(node, NonTerminalNode)]
    # get one start point at random
    start = rand.choice(possible_starts)
    # perform the mutation
    son = rebuild_branch(ind, schema, start)
    # check whether the new individual is equal
    if str(son) == par_str:
        return son, False
    return son, True
