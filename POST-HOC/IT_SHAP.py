# BSD 2-Clause License
#
# Copyright (c) 2024, Óscar Escudero-Arnanz
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
This file is based on the original SHAP implementation:
https://github.com/slundberg/shap/blob/master/shap/explainers/_kernel.py
"""


import numpy as np
import pandas as pd
import scipy as sp
import logging
import copy
import itertools
from typing import Tuple, Union


from shap.utils._legacy import convert_to_link, IdentityLink
from shap.utils._legacy import convert_to_instance, convert_to_model
from shap.explainers._kernel import Kernel

from scipy.special import binom
from scipy.sparse import issparse

log = logging.getLogger('shap')

import os
import re
import csv
from pathlib import Path
from timeshap.utils import convert_to_indexes, convert_data_to_3d
from timeshap.explainer import temp_coalition_pruning
from timeshap.utils import get_tolerances_to_test
from typing import Callable, List, Union

from shap.utils._legacy import Instance, Model, Data

def time_shap_match_model_to_data(model, data):
    assert isinstance(model, Model), "model must be of type Model!"
    data = data.data
    returns_hs = False
    try:
        out_val = model.f(data)
        if len(out_val) == 2:
            # model returns the hidden state aswell.
            # We can use this hidden state to make the algorithm more efficent
            # as we reduce the computation of all pruned events to a single hidden state
            out_val, _ = out_val
            returns_hs = True
    except:
        raise

    if model.out_names is None:
        if len(out_val.shape) == 1:
            model.out_names = ["output value"]
        else:
            model.out_names = ["output value " + str(i) for i in range(out_val.shape[0])]

    return out_val, returns_hs


class TimeShapDenseData(Data):
    def __init__(self, data, mode, group_names, *args):
        
        self.groups = args[0] if len(args) > 0 and args[0] is not None else [np.array([i]) for i in range(len(group_names))]

        if mode in ["event", "feature", "pruning", "cell"]:
            self.weights = args[2] if len(args) > 1 else np.ones(data.shape[0])
            self.weights /= np.sum(self.weights)

            self.transposed = False
            self.group_names = group_names
            self.data = data
            self.groups_size = len(self.groups)

        else:
            raise ValueError("TimeShapDenseData - mode not supported")


def time_shap_match_instance_to_data(instance, data):
    assert isinstance(instance, Instance), "instance must be of type Instance!"

    if isinstance(data, TimeShapDenseData):
        if instance.group_display_values is None:
            instance.group_display_values = [instance.x[0, :, group[0]] if len(group) == 1 else "" for group in data.groups]
        assert len(instance.group_display_values) == len(data.groups)
        instance.groups = data.groups
    else:
        raise NotImplementedError("Type of data not supported")


def time_shap_convert_to_data(val, mode, pruning_idx, varying=None):
    if type(val) == np.ndarray:
        if mode == 'event':
            event_names = ["Event: {}".format(i) for i in np.arange(val.shape[1], pruning_idx, -1)]
            if pruning_idx > 0:
                event_names += ["Pruned Events"]
            return TimeShapDenseData(val, mode, event_names)
        elif mode == 'feature':
            feature_names = ["Feat: {}".format(i) for i in np.arange(val.shape[2])]
            if pruning_idx > 0:
                feature_names += ["Pruned Events"]
            return TimeShapDenseData(val, mode, feature_names)
        elif mode == 'cell':
            group_names = []
            for event_idx in varying[0]:
                for feat_idx in varying[1]:
                    group_names += ["({}, {})".format(event_idx, feat_idx)]

            # OEA ==========>
            pruned_events = False
            all_other = False
            other_event_rel_feat = False
            other_feat_rel_event = False
            # END OEA <=============
        

            return TimeShapDenseData(val, mode, group_names), [other_feat_rel_event, other_event_rel_feat, all_other, pruned_events]
        elif mode == 'pruning':
            return TimeShapDenseData(val, mode, ["x", "hidden"])
        else:
            raise ValueError("`convert_to_data` - mode not supported")

    elif isinstance(val, Data):
        return val
    else:
        assert False, "Unknown type passed as data object: " + str(type(val))


class TimeShapKernel(Kernel):
    """Uses TimeSHAP Kernel SHAP method to explain the output of any function.

    TimeSHAP extends KernelSHAP to explain sequences of features on several axis.
    TimeSHAP calculates, event, feature, and cell level explanations.
    Due to sequences being arbitrarily long, TimeSHAP also implements a pruning
    algorithm based on Shapley values, to select the most relevant, recent,
    consecutive events.

    Parameters
    ----------
    model: function
        User supplied function that takes a 3D array (# samples x # sequence length x # features)
        and computes the output of the model for those samples. The output can be a vector
        (# samples) or a matrix (# samples x # model outputs).
        In order to use TimeSHAP in an optimized way, this model can also return the explained
        model's hidden state.

    background : numpy.array or pd.DataFrame
        The background event/sequence to use for integrating out features. To determine the impact
        of a feature, that feature is set to "missing" and the change in the model output
        is observed. Since most models aren't designed to handle arbitrary missing data at test
        time, we simulate "missing" by replacing the feature with the values it takes in the
        background dataset. So if the background dataset is a simple sample of all zeros, then
        we would approximate a feature being missing by setting it to zero.
        In TimeSHAP you can use an average event or average sequence.
        When using average events, consider using `timeshap.calc_avg_event` method to obtain it.
        When using average sequence, considering using `timeshap.calc_avg_sequence` method to obtain it.
        Note that when using the average sequence, all sequences of the dataset need to be the same length.

    rs: int
        Random seed for timeshap algorithm

    mode: str
        This method indicates what kind of explanations should be calculated.
        Possible values: ["pruning", "event", "feature", "cell"]
            - "pruning" - used for pruning algorithm
            - "event" - used for event explanations
            - "feature" - used for feature explanations
            - "cell" -used for cell explanations

    varying: Tuple
        index of varying indexes on cell level
        If mode == "cell": varying needs to be of len 2, the first the idx of
            events to preturb, and the second the idx of features

    link : "identity" or "logit"
        A generalized linear model link to connect the feature importance values to the model
        output. Since the feature importance values, phi, sum up to the model output, it often makes
        sense to connect them to the output with a link function where link(output) = sum(phi).
        If the model output is a probability then the LogitLink link function makes the feature
        importance values have log-odds units.
    """
    def __init__(self, model, background, rs, mode, varying=None, link=IdentityLink(), **kwargs):
        self.background = background
        self.random_seed = rs
        self.mode = mode
        self.data = None
        self.varyingInds = None
        self.pruning_idx = None
        self.varying = varying
        self.returns_hs = None
        self.background_hs = None
        self.instance_hs = None
        if mode in 'cell':
            if varying is None:
                # The algorithm supports the calculation using all events and features
                # but its computation is very expensive as the number of cells is very large
                raise ValueError("Cell level needs to receive which cells to calculate")
            self.varying = varying
            cell_idx_keys = []
            for event_idx in self.varying[0]:
                for feat_idx in self.varying[1]:
                    cell_idx_keys += [[event_idx, feat_idx]]
                    

            self.cell_idx_keys = np.array(cell_idx_keys)
            # OEA =======>
            self.len_groups = len(self.cell_idx_keys)
            event_idx = self.cell_idx_keys[:, 0]
            # END OEA <===============
            self.considered_cells = {}
            for event in np.unique(event_idx):
                self.considered_cells[event] = self.cell_idx_keys[self.cell_idx_keys[:, 0] == event][:, 1]
                
        # convert incoming inputs to standardized iml objects
        self.link = convert_to_link(link)
        self.model = convert_to_model(model)
        self.keep_index = kwargs.get("keep_index", False)
        self.keep_index_ordered = kwargs.get("keep_index_ordered", False)
        
    def set_variables_up(self, X: np.ndarray):
        """Sets variables up for explanations

        Parameters
        ----------
        X: Union[pd.DataFrame, np.ndarray]
            Instance being explained
        """
        if len(self.background.shape) == 2:
            # 2D background needs to be expanded
            if self.background.shape[0] > 1 and not self.background.shape[0] == X.shape[1]:
                raise ValueError(
                    "When using background events, you can only pass one average event."
                    "When using background sequence, your background must be the same sequence length of the explained sequence")
            elif self.background.shape[0] > 1 and self.background.shape[0] == X.shape[1]:
                # average sequence
                sequence = copy.deepcopy(self.background)
            elif self.background.shape[0] == 1:
                # average event
                sequence = np.tile(self.background, (X.shape[1], 1))
            else:
                raise ValueError("Unknown combination of background and sequence sizes. Please open a ticket on github")
            sequence = np.expand_dims(sequence.copy(), axis=0)
        elif len(self.background.shape) == 3:
            if self.background.shape[1] > 1 and not self.background.shape[1] == X.shape[1]:
                raise ValueError(
                    "When using background events, you can only pass one average event."
                    "When using background sequence, your background must be the same sequence length of the explained sequence")
            elif self.background.shape[1] > 1 and self.background.shape[1] == X.shape[1]:
                # average sequence
                sequence = copy.deepcopy(self.background)
            elif self.background.shape[1] == 1:
                # average event
                sequence = np.tile(self.background, (1, X.shape[1], 1))
            else:
                raise ValueError("Unknown combination of background and sequence sizes. Please open a ticket on github")

        if self.mode == 'cell':
            self.data, self.special_cells = time_shap_convert_to_data(sequence, self.mode, self.pruning_idx, self.varying)
        else:
            self.data = time_shap_convert_to_data(sequence, self.mode, self.pruning_idx, self.varying)

        model_null, returns_hs = time_shap_match_model_to_data(self.model, self.data)
        self.returns_hs = returns_hs

        if not self.mode == 'pruning' and self.returns_hs:
            if self.pruning_idx == 0:
                # obtain the HS format
                _, example_hs = self.model.f(X[:, -1:, :])
                if isinstance(example_hs, tuple):
                    if isinstance(example_hs[0], tuple):
                        self.instance_hs = tuple(tuple(np.zeros_like(example_hs[y_i][i_x]) for i_x, x in enumerate(y)) for y_i, y in enumerate(example_hs))
                        self.background_hs = tuple(tuple(np.zeros_like(example_hs[y_i][i_x]) for i_x, x in enumerate(y)) for y_i, y in enumerate(example_hs))
                    else:
                        self.instance_hs = tuple(np.zeros_like(example_hs[i]) for i, x in enumerate(example_hs))
                        self.background_hs = tuple(np.zeros_like(example_hs[i]) for i, x in enumerate(example_hs))
                else:
                    self.instance_hs = np.zeros_like(example_hs)
                    self.background_hs = np.zeros_like(example_hs)
            else:
                _, self.background_hs = self.model.f(sequence[:, :self.pruning_idx, :])
                _, self.instance_hs = self.model.f(X[:, :self.pruning_idx, :])
                assert isinstance(self.background_hs, (np.ndarray, tuple)), "Hidden states are required to be numpy arrays or tuple "
                if isinstance(self.background_hs, tuple):
                    if isinstance(self.background_hs[0], tuple):
                        # working with LSTM
                        assert np.array([len(x) == 2 for x in self.background_hs]).all()
                        assert np.array([[isinstance(y, np.ndarray) for y in x] for x in self.background_hs]).all()
                    else:
                        assert np.array([isinstance(x, np.ndarray) for x in self.background_hs]).all()
        # enforce our current input type limitations
        assert isinstance(self.data, TimeShapDenseData), \
            "Shap explainer only supports the DenseData input currently."
        assert not self.data.transposed, "Shap explainer does not support transposed DenseData or SparseData currently."

        # warn users about large background data sets
        if len(self.data.weights) > 100:
            log.warning("Using " + str(len(
                self.data.weights)) + " background data samples could cause " +
                        "slower run times. Consider using shap.sample(data, K) or shap.kmeans(data, K) to " +
                        "summarize the background as K samples.")

        # init our parameters
        self.N = self.data.data.shape[0]
        # seq len total
        self.S = self.data.data.shape[1]
        self.P = self.data.data.shape[2]
        self.linkfv = np.vectorize(self.link.f)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

        # find E_x[f(x)]
        if isinstance(model_null, (pd.DataFrame, pd.Series)):
            model_null = np.squeeze(model_null.values)
        self.fnull = np.sum((model_null.T * self.data.weights).T, 0)
        self.expected_value = self.linkfv(self.fnull)
        
        # OEA: Adapt self.fnull and self.expected_value as single vector
        # =====================================================================>
        self.fnull = np.array([np.mean(self.fnull[:self.rows_filtered])])
        self.expected_value = np.array([np.mean(self.expected_value[:self.rows_filtered])])
        self.S = self.rows_filtered
        # <======================================================================

        # see if we have a vector output
        self.vector_out = True
        if len(self.fnull.shape) == 0:
            self.vector_out = False
            self.fnull = np.array([self.fnull])
            self.D = 1
            self.expected_value = float(self.expected_value)
        else:
            self.D = self.fnull.shape[0]

    def shap_values(self, X, pruning_idx=None, **kwargs):
        """ Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array
            A 3D matrix (#samples x #events x #features) on which to explain the model's output.

        nsamples : "auto" or int
            Number of times to re-evaluate the model when explaining each prediction. More samples
            lead to lower variance estimates of the SHAP values. The "auto" setting uses
            `nsamples = 2 * X.shape[1] + 2048`.

        l1_reg : "num_features(int)", "auto" (default for now, but deprecated), "aic", "bic", or float
            The l1 regularization to use for feature selection (the estimation procedure is based on
            a debiased lasso). The auto option currently uses "aic" when less that 20% of the possible sample
            space is enumerated, otherwise it uses no regularization. THE BEHAVIOR OF "auto" WILL CHANGE
            in a future version to be based on num_features instead of AIC.
            The "aic" and "bic" options use the AIC and BIC rules for regularization.
            Using "num_features(int)" selects a fix number of top features. Passing a float directly sets the
            "alpha" parameter of the sklearn.linear_model.Lasso model used for feature selection.

        Returns
        -------
        For models with a single output this returns a matrix of SHAP values
        (# samples x # features). Each row sums to the difference between the model output for that
        sample and the expected value of the model output (which is stored as expected_value
        attribute of the explainer). For models with vector outputs this returns a list
        of such matrices, one for each output.
        """
        assert isinstance(X, np.ndarray), "Instance must be 3D numpy array"
        if self.mode == "pruning":
            assert pruning_idx is not None
        else:
            assert pruning_idx < X.shape[1], "Pruning idx must be smaller than the sequence length. If not all events are pruned"
        assert pruning_idx % 1 == 0, "Pruning idx must be integer"
        self.pruning_idx = int(pruning_idx)
        

        ### OEA: My code adaptation fosr temporal output
        rows_filtered = list(np.any(X == 666, axis=-1)[0])
        self.rows_filtered = rows_filtered.count(False)
        # END OEA <======================================
        self.set_variables_up(X)

        if sp.sparse.issparse(X) and not sp.sparse.isspmatrix_lil(X):
            X = X.tolil()
    
        
        if X.shape[0] == 1:            
            explanation = self.explain(X, **kwargs)
            explanation = np.mean(explanation, axis=1)
            explanation = explanation.reshape(-1, 1)
            out = np.zeros(explanation.shape[0])

            if isinstance(explanation.shape, tuple) and len(explanation.shape) == 2:
                assert explanation.shape[1] == 1
                out[:] = explanation[:, 0]
            else:
                out[:] = explanation

            return out
        
    def explain(self, incoming_instance, **kwargs):
        # convert incoming input to a standardized iml object
        instance = convert_to_instance(incoming_instance)
        instance.group_display_values = self.data.group_names
        time_shap_match_instance_to_data(instance, self.data)

        # Find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        if self.mode == "event":
            # OEA ============================================================================>
            self.varyingInds = np.array([x for x in np.arange(self.rows_filtered-1, -1, -1)])
            # <================================================================================
        elif self.mode == 'pruning':
            self.varyingInds = [0, 1]
        elif self.mode == "feature":
            if self.pruning_idx > 0:
                self.varyingInds = self.varying_groups(instance.x, self.data.groups_size - 1)
                # add an index for pruned events
                self.varyingInds = np.concatenate((self.varyingInds, np.array([self.data.groups_size - 1])))
            else:
                self.varyingInds = self.varying_groups(instance.x, self.data.groups_size)
        elif self.mode == 'cell':
            self.varyingInds = np.arange(len(self.data.groups))
        else:
            raise ValueError("`explain` -> mode not suported")

        if self.data.groups is None:
            self.varyingFeatureGroups = np.array([i for i in self.varyingInds])
            self.M = self.varyingFeatureGroups.shape[0]
        else:
            if self.mode in ['event']:
                self.varyingFeatureGroups = self.varyingInds
                self.M = len(self.varyingFeatureGroups)
                if self.pruning_idx > 0:
                    self.M += 1
            elif self.mode in ['feature']:
                if self.pruning_idx > 0:
                    self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds[:-1]]
                else:
                    self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
                self.M = len(self.varyingFeatureGroups)
                if self.pruning_idx > 0:
                    self.M += 1
            else:
                self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
                self.M = len(self.varyingFeatureGroups)

            groups = self.data.groups
            # convert to numpy array as it is much faster if not jagged array (all groups of same length)
            if isinstance(self.varyingFeatureGroups, list) and all(len(groups[i]) == len(groups[0]) for i in range(len(self.varyingFeatureGroups))):
                self.varyingFeatureGroups = np.array(self.varyingFeatureGroups)
                # further performance optimization in case each group has a single value
                if self.varyingFeatureGroups.shape[1] == 1:
                    self.varyingFeatureGroups = self.varyingFeatureGroups.flatten()

        if self.returns_hs:
            # Removed the input variability to receive pd.series and DataFrame
            model_out, _ = self.model.f(instance.x)
        else:
            model_out = self.model.f(instance.x)
            
        ### OEA =============>
        model_out = model_out[:, :self.rows_filtered]
        model_out = np.array([[np.mean(model_out)]])
        self.fx = model_out[0]
        # END OEA <===========


        if not self.vector_out:
            self.fx = np.array([self.fx])

        # if no features vary then no feature has an effect
        if self.M == 0:
            phi = np.zeros((self.data.groups_size, self.D))

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            phi = np.zeros((self.data.groups_size, self.D))
            diff = self.link.f(self.fx) - self.link.f(self.fnull)
            for d in range(self.D):
                phi[self.varyingInds[0], d] = diff[d]

        # if more than one feature varies then we have to do real work
        else:
            self.l1_reg = kwargs.get("l1_reg", "auto")

           # pick a reasonable number of samples if the user didn't specify how many they wanted
            self.nsamples = kwargs.get("nsamples", "auto")
            if self.nsamples == "auto":
                self.nsamples = 2 * self.M + 2**11

            # if we have enough samples to enumerate all subsets then ignore the unneeded samples
            self.max_samples = 2 ** 30
            if self.M <= 30:
                self.max_samples = 2 ** self.M - 2
                if self.nsamples > self.max_samples:
                    self.nsamples = self.max_samples

            # reserve space for some of our computations
            self.allocate()

            # weight the different subset sizes
            num_subset_sizes = int(np.ceil((self.M - 1) / 2.0))
            num_paired_subset_sizes = int(np.floor((self.M - 1) / 2.0))
            weight_vector = np.array([(self.M - 1.0) / (i * (self.M - i)) for i in range(1, num_subset_sizes + 1)])
            weight_vector[:num_paired_subset_sizes] *= 2
            weight_vector /= np.sum(weight_vector)
            log.debug("weight_vector = {0}".format(weight_vector))
            log.debug("num_subset_sizes = {0}".format(num_subset_sizes))
            log.debug("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
            log.debug("M = {0}".format(self.M))
        

            # fill out all the subset sizes we can completely enumerate
            # given nsamples*remaining_weight_vector[subset_size]
            num_full_subsets = 0
            num_samples_left = self.nsamples
            group_inds = np.arange(self.M, dtype='int64')
            mask = np.zeros(self.M)
            remaining_weight_vector = copy.copy(weight_vector)
            for subset_size in range(1, num_subset_sizes + 1):

                # determine how many subsets (and their complements) are of the current size
                nsubsets = binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes: nsubsets *= 2
                log.debug("subset_size = {0}".format(subset_size))
                log.debug("nsubsets = {0}".format(nsubsets))
                log.debug("self.nsamples*weight_vector[subset_size-1] = {0}".format(
                    num_samples_left * remaining_weight_vector[subset_size - 1]))
                log.debug("self.nsamples*weight_vector[subset_size-1]/nsubsets = {0}".format(
                    num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets))

                # see if we have enough samples to enumerate all subsets of this size
                if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                    num_full_subsets += 1
                    num_samples_left -= nsubsets

                    # rescale what's left of the remaining weight vector to sum to 1
                    if remaining_weight_vector[subset_size - 1] < 1.0:
                        remaining_weight_vector /= (1 - remaining_weight_vector[subset_size - 1])

                    # add all the samples of the current subset size
                    w = weight_vector[subset_size - 1] / binom(self.M, subset_size)
                    if subset_size <= num_paired_subset_sizes: w /= 2.0
                    for inds in itertools.combinations(group_inds, subset_size):
                        mask[:] = 0.0
                        mask[np.array(inds, dtype='int64')] = 1.0
                        
                        self.add_sample(instance.x, mask, w)
                        if subset_size <= num_paired_subset_sizes:
                            mask[:] = np.abs(mask - 1)
                            self.add_sample(instance.x, mask, w)
                else:
                    break

            log.info("num_full_subsets = {0}".format(num_full_subsets))
            # add random samples from what is left of the subset space
            nfixed_samples = self.nsamplesAdded
            samples_left = self.nsamples - self.nsamplesAdded
            log.debug("samples_left = {0}".format(samples_left))
            np.random.seed(self.random_seed)
            if num_full_subsets != num_subset_sizes:
                remaining_weight_vector = copy.copy(weight_vector)
                remaining_weight_vector[:num_paired_subset_sizes] /= 2 # because we draw two samples each below
                remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
                remaining_weight_vector /= np.sum(remaining_weight_vector)
                
                log.info("remaining_weight_vector = {0}".format(remaining_weight_vector))
                log.info("num_paired_subset_sizes = {0}".format(num_paired_subset_sizes))
                ind_set = np.random.choice(len(remaining_weight_vector), 4 * samples_left, p=remaining_weight_vector)
                ind_set_pos = 0
                used_masks = {}

                while samples_left > 0 and ind_set_pos < len(ind_set):
                    mask.fill(0.0)
                    ind = ind_set[ind_set_pos] # we call np.random.choice once to save time and then just read it here
                    ind_set_pos += 1
                    subset_size = ind + num_full_subsets + 1
                    mask[np.random.permutation(self.M)[:subset_size]] = 1.0

                    # only add the sample if we have not seen it before, otherwise just
                    # increment a previous sample's weight
                    mask_tuple = tuple(mask)
                    new_sample = False
                    if mask_tuple not in used_masks:
                        new_sample = True
                        used_masks[mask_tuple] = self.nsamplesAdded
                        samples_left -= 1
                        self.add_sample(instance.x, mask, 1.0)
                    else:
                        self.kernelWeights[used_masks[mask_tuple]] += 1.0

                    # add the compliment sample
                    if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)

                        # only add the sample if we have not seen it before, otherwise just
                        # increment a previous sample's weight
                        if new_sample:
                            samples_left -= 1
                            self.add_sample(instance.x, mask, 1.0)
                        else:
                            # we know the compliment sample is the next one after the original sample, so + 1
                            self.kernelWeights[used_masks[mask_tuple] + 1] += 1.0

                # normalize the kernel weights for the random samples to equal the weight left after
                # the fixed enumerated samples have been already counted

                weight_left = np.sum(weight_vector[num_full_subsets:])
                log.info("weight_left = {0}".format(weight_left))
                self.kernelWeights[nfixed_samples:] *= weight_left / self.kernelWeights[nfixed_samples:].sum()


            # execute the model on the synthetic samples we have created
            self.run()

            # solve then expand the feature importance (Shapley value) vector to contain the non-varying features
            # OEA =======================>
            if self.mode == "event":
                self.data.groups_size = self.rows_filtered
            # END OEA <===================
            
            phi = np.zeros((self.data.groups_size, self.D))
            for d in range(self.D):
                vphi, _ = self.solve(self.nsamples / self.max_samples, d)
                if self.mode == 'event':
                    phi[:, d] = vphi
                elif self.mode == 'cell':
                    phi[:, d] = vphi
                else:
                    phi[self.varyingInds, d] = vphi

        if not self.vector_out:
            phi = np.squeeze(phi, axis=1)
            
        return phi

    @staticmethod
    def not_equal(i, j):
        if isinstance(i, str) or isinstance(j, str):
            return 0 if i == j else 1
        return 0 if np.isclose(i, j, equal_nan=True) else 1

    def varying_groups(self, x, group_size):
        if not sp.sparse.issparse(x):
            varying = np.zeros(group_size)
            for i in range(0, group_size):
                inds = self.data.groups[i]
                x_group = x[0, :, inds]
                if sp.sparse.issparse(x_group):
                    if all(j not in x.nonzero()[1] for j in inds):
                        varying[i] = False
                        continue
                    x_group = x_group.todense()
                num_mismatches = np.sum(np.frompyfunc(self.not_equal, 2, 1)(x_group, self.data.data[:, 0, inds]))
                varying[i] = num_mismatches > 0
            varying_indices = np.nonzero(varying)[0]
            return varying_indices
        else:
            # go over all nonzero columns in background and evaluation data
            # if both background and evaluation are zero, the column does not vary
            varying_indices = np.unique(np.union1d(self.data.data.nonzero()[1], x.nonzero()[1]))
            remove_unvarying_indices = []
            for i in range(0, len(varying_indices)):
                varying_index = varying_indices[i]
                # now verify the nonzero values do vary
                data_rows = self.data.data[:, [varying_index]]
                nonzero_rows = data_rows.nonzero()[0]

                if nonzero_rows.size > 0:
                    background_data_rows = data_rows[nonzero_rows]
                    if sp.sparse.issparse(background_data_rows):
                        background_data_rows = background_data_rows.toarray()
                    num_mismatches = np.sum(np.abs(background_data_rows - x[0, varying_index]) > 1e-7)
                    # Note: If feature column non-zero but some background zero, can't remove index
                    if num_mismatches == 0 and not \
                        (np.abs(x[0, [varying_index]][0, 0]) > 1e-7 and len(nonzero_rows) < data_rows.shape[0]):
                        remove_unvarying_indices.append(i)
            mask = np.ones(len(varying_indices), dtype=bool)
            mask[remove_unvarying_indices] = False
            varying_indices = varying_indices[mask]
            return varying_indices

    def allocate(self):
        if sp.sparse.issparse(self.data.data):
            # We tile the sparse matrix in csr format but convert it to lil
            # for performance when adding samples
            shape = self.data.data.shape
            nnz = self.data.data.nnz
            data_rows, data_cols = shape
            rows = data_rows * self.nsamples
            shape = rows, data_cols
            if nnz == 0:
                self.synth_data = sp.sparse.csr_matrix(shape, dtype=self.data.data.dtype).tolil()
            else:
                data = self.data.data.data
                indices = self.data.data.indices
                indptr = self.data.data.indptr
                last_indptr_idx = indptr[len(indptr) - 1]
                indptr_wo_last = indptr[:-1]
                new_indptrs = []
                for i in range(0, self.nsamples - 1):
                    new_indptrs.append(indptr_wo_last + (i * last_indptr_idx))
                new_indptrs.append(indptr + ((self.nsamples - 1) * last_indptr_idx))
                new_indptr = np.concatenate(new_indptrs)
                new_data = np.tile(data, self.nsamples)
                new_indices = np.tile(indices, self.nsamples)
                self.synth_data = sp.sparse.csr_matrix((new_data, new_indices, new_indptr), shape=shape).tolil()
        else:
            if self.returns_hs and self.mode != 'pruning':
                self.synth_data = np.tile(self.data.data[:, self.pruning_idx:, :], (self.nsamples, 1, 1))
                if isinstance(self.background_hs, tuple):
                    if isinstance(self.background_hs[0], tuple):
                        self.synth_hidden_states = tuple(tuple(np.tile(y, (1, self.nsamples, 1)) for y in x) for x in self.background_hs)
                    else:
                        self.synth_hidden_states = tuple(np.tile(x, (1, self.nsamples, 1)) for x in self.background_hs)
                else:
                    self.synth_hidden_states = np.tile(self.background_hs, (1, self.nsamples, 1))

            else:
                self.synth_data = np.tile(self.data.data, (self.nsamples, 1, 1))

        self.maskMatrix = np.zeros((self.nsamples, self.M))
        self.kernelWeights = np.zeros(self.nsamples)
        
        self.y = np.zeros((self.nsamples * self.N, self.D))
        self.ey = np.zeros((self.nsamples, self.D))
        self.lastMask = np.zeros(self.nsamples)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0
        if self.keep_index:
            self.synth_data_index = np.tile(self.data.index_value, self.nsamples)

    def add_sample(self, x, m, w):
        offset = self.nsamplesAdded * self.N
        mask = m == 1.0
        if self.mode == "event":
            self.event_add_sample(x, mask, offset)
        elif self.mode == "feature":
            self.feat_add_sample(x, mask, offset)
        elif self.mode == 'pruning':
            self.pruning_add_sample(x, mask, offset)
        elif self.mode == "cell":
            self.cell_add_sample(x, mask, offset)
        else:
            raise ValueError("`add_sample` - Mode not supported")

        self.maskMatrix[self.nsamplesAdded, :] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def activate_background(self, x, offset):
        # in case self.pruning_idx == sequence length, we dont prune anything.
        if not self.pruning_idx == self.S:
            if self.returns_hs:
                # in case of using hidden state optimization, the background is the instance one
                if isinstance(self.synth_hidden_states, tuple):
                    if isinstance(self.synth_hidden_states[0], tuple):
                        for i, i_layer_state in enumerate(self.synth_hidden_states):
                            i_layer_state[0][:, offset:offset + self.N, :] = self.instance_hs[i][0]
                            i_layer_state[1][:, offset:offset + self.N, :] = self.instance_hs[i][1]
                    else:
                        for i, i_layer_state in enumerate(self.synth_hidden_states):
                            i_layer_state[:, offset:offset + self.N, :] = self.instance_hs[i]
                else:
                    self.synth_hidden_states[:, offset:offset + self.N, :] = self.instance_hs
            else:
                # in case of not using hidden state optimization, we need to set the whole background to the original sequence
                evaluation_data = x[0:1, :self.pruning_idx, :]
                self.synth_data[offset:offset + self.N, :self.pruning_idx, :] = evaluation_data

    def cell_add_sample(self, x, mask, offset):

        x = x[:,:self.rows_filtered]
        cells_to_preturb = self.cell_idx_keys[mask[: self.cell_idx_keys.shape[0]], :]
        relevent_events = np.unique(self.cell_idx_keys[:, 0])
        relevent_feats = np.unique(self.cell_idx_keys[:, 1])

        feats_by_event = {}
        for event in np.unique(cells_to_preturb[:, 0]):
            feats_by_event[event] = cells_to_preturb[cells_to_preturb[:, 0] == event][:, 1]

        # BACKGROUND IS ACTIVE
        if self.special_cells[3] and mask[-self.special_cells[3]]:
            self.activate_background(x, offset)

        # other cells are active (no relevant events or feats)
        if self.special_cells[2] and mask[-self.special_cells[2]]:
            other_events = [x for x in np.arange(self.pruning_idx, x.shape[1]) if x not in relevent_events]
            other_feats = [x for x in range(x.shape[2]) if x not in relevent_feats]

            for event in other_events:
                evaluation_data = x[0:1, event, other_feats]
                if self.returns_hs:
                    event = event - self.pruning_idx
                self.synth_data[offset:offset + self.N, event, other_feats] = evaluation_data

        mask_pointer = self.cell_idx_keys.shape[0]
        if self.special_cells[0]:
            # other feats in relevant events
            perturb_events = relevent_events[mask[mask_pointer: mask_pointer + len(self.varying[0])]]
            mask_pointer += len(self.varying[0])
            for event in perturb_events:
                other_feats = [x for x in range(x.shape[2]) if x not in relevent_feats]
                evaluation_data = x[0:1, event, other_feats]
                if self.returns_hs:
                    event = event - self.pruning_idx
                self.synth_data[offset:offset + self.N, event, other_feats] = evaluation_data

        if self.special_cells[1]:
            # other events in relevant feats
            perturb_feats = relevent_feats[mask[mask_pointer: mask_pointer + len(self.varying[1])]]
            mask_pointer += len(self.varying[1])
            other_events = [x for x in np.arange(self.pruning_idx, x.shape[1]) if x not in relevent_events]
            for event in other_events:
                evaluation_data = x[0:1, event, perturb_feats]
                if self.returns_hs:
                    event = event - self.pruning_idx
                self.synth_data[offset:offset + self.N, event, perturb_feats] = evaluation_data

        # activate individual cells
        for event, feats in feats_by_event.items():
            evaluation_data = x[0:1, event, feats]
            if self.returns_hs:
                event = event - self.pruning_idx
            self.synth_data[offset:offset + self.N, event, feats] = evaluation_data

    def event_add_sample(self, x, mask, offset):
        # there is a background and it is active
        if self.pruning_idx > 0 and mask[-1]:
            self.activate_background(x, offset)

        if self.pruning_idx > 0:
            # there is a background, so the last position of the mask is for it
            groups = self.varyingFeatureGroups[mask[:-1]]
        else:
            groups = self.varyingFeatureGroups[mask]

        evaluation_data = x[0:1, groups, :]
        if self.returns_hs:
            # re-align indexes to the truncated sequence
            groups = [x-self.pruning_idx for x in groups]
        self.synth_data[offset:offset + self.N, groups, :] = evaluation_data

    def feat_add_sample(self, x, mask, offset):
        #BACKGROUND IS ACTIVE
        if self.pruning_idx > 0 and mask[-1]:
            self.activate_background(x, offset)

        if self.pruning_idx > 0:
            # there is a background, so the last position of the mask is for it
            groups = self.varyingFeatureGroups[mask[:-1]]
        else:
            groups = self.varyingFeatureGroups[mask]

        evaluation_data = x[0:1, self.pruning_idx:, groups]
        if self.returns_hs:
            self.synth_data[offset:offset+self.N, :, groups] = evaluation_data
        else:
            self.synth_data[offset:offset+self.N, self.pruning_idx:, groups] = evaluation_data

    def pruning_add_sample(self, x, mask, offset):
        if not len(mask) == 2:
            raise ValueError("For pruning mode, masks must have size 2")
        if mask[0]:
            # cur active
            evaluation_data = x[0:1, self.pruning_idx:, :]
            self.synth_data[offset:offset + self.N, self.pruning_idx:, :] = evaluation_data
        if mask[1]:
            # background active
            evaluation_data = x[0:1, :self.pruning_idx, :]
            self.synth_data[offset:offset + self.N, :self.pruning_idx, :] = evaluation_data

    def run(self):
        num_to_run = self.nsamplesAdded * self.N - self.nsamplesRun * self.N
        data = self.synth_data[self.nsamplesRun * self.N:self.nsamplesAdded * self.N, :, :]

        if not self.mode == 'pruning' and self.returns_hs:
            if isinstance(self.synth_hidden_states, tuple):
                if isinstance(self.synth_hidden_states[0], tuple):
                    hidden_sates = tuple(tuple(y[:, self.nsamplesRun * self.N: self.nsamplesAdded * self.N,:] for y in x) for x in self.synth_hidden_states)
                else:
                    hidden_sates = tuple(x[:, self.nsamplesRun * self.N: self.nsamplesAdded * self.N,:] for x in self.synth_hidden_states)
            else:
                hidden_sates = self.synth_hidden_states[:, self.nsamplesRun * self.N: self.nsamplesAdded * self.N,:]

            modelOut, _ = self.model.f(data, hidden_sates)
            modelOut = modelOut[:, :self.rows_filtered]
        elif self.returns_hs:
            modelOut, _ = self.model.f(data)
            modelOut = modelOut[:, :self.rows_filtered]
        else:
            modelOut = self.model.f(data)
            # OEA =======================================>
            modelOut = modelOut[:, :self.rows_filtered]
            # END OEA <===================================

        if isinstance(modelOut, (pd.DataFrame, pd.Series)):
            modelOut = modelOut.values

        # OEA =====================================================================>
        modelOut = np.mean(modelOut, axis=1)
        # <=========================================================================

        self.y[self.nsamplesRun * self.N:self.nsamplesAdded * self.N, :] = np.reshape(modelOut, (num_to_run, self.D))

        # find the expected value of each output
        for i in range(self.nsamplesRun, self.nsamplesAdded):
            eyVal = np.zeros(self.D)
            for j in range(0, self.N):
                eyVal += self.y[i * self.N + j, :] * self.data.weights[j]

            self.ey[i, :] = eyVal
            self.nsamplesRun += 1
            
           
def local_feat(f: Callable[[np.ndarray], np.ndarray],
               data: np.array,
               feature_dict: dict,
               entity_uuid: Union[str, int, float],
               entity_col: str,
               baseline: Union[pd.DataFrame, np.array],
               pruned_idx: int,
               ) -> pd.DataFrame:
    """Method to calculate event level explanations or load them if path is provided

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: np.array
        Sequence to explain.

    feature_dict: dict
        Information required for the feature level explanation calculation

    entity_uuid: Union[str, int, float]
        The indentifier of the sequence that is being pruned.
        Used when fetching information from a csv of explanations

    entity_col: str
        Entity column to identify sequences

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.
        The baseline can be an average event or an average sequence

    pruned_idx: int
        Index to prune the sequence. All events up to this point are grouped

    Returns
    -------
    pd.DataFrame
    """
    if feature_dict.get("path") is None or not os.path.exists(feature_dict.get("path")):
        feat_data = feature_level(f, data, baseline, pruned_idx, feature_dict.get("rs"), feature_dict.get("nsamples"), model_feats=feature_dict.get("feature_names"))
        if feature_dict.get("path") is not None:
            # create directory
            if '/' in feature_dict.get("path"):
                Path(feature_dict.get("path").rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
            feat_data.to_csv(feature_dict.get("path"), index=False)
    elif feature_dict.get("path") is not None and os.path.exists(feature_dict.get("path")):
        feat_data = pd.read_csv(feature_dict.get("path"))
        if len(feat_data.columns) == 5 and entity_col is not None:
            feat_data = feat_data[feat_data[entity_col] == entity_uuid]
        elif len(feat_data.columns) == 4:
            pass
        else:
            # TODO
            # the provided csv should be generated by timeshap, by either
            # explaining the whole dataset with TODO or just the instance in question
            raise ValueError
    else:
        raise ValueError
    return feat_data


def feature_level(f: Callable,
                  data: np.ndarray,
                  baseline: np.ndarray,
                  pruned_idx: int,
                  random_seed: int,
                  nsamples: int,
                  model_feats: List[Union[int, str]] = None,
                  ) -> pd.DataFrame:
    """Method to calculate event level explanations

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: np.array
        Sequence to explain.

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.
        The baseline can be an average event or an average sequence

    pruned_idx: int
        Index to prune the sequence. All events up to this index are grouped

    random_seed: int
        Used random seed for the sampling process.

    nsamples: int
        The number of coalitions for TimeSHAP to sample.

    model_feats: List[Union[int, str]]
        The list of feature names.
        If none is provided, "Feature 1" format is used

    Returns
    -------
    pd.DataFrame
    """
    if pruned_idx == -1:
        pruned_idx = 0

    explainer = TimeShapKernel(f, baseline, random_seed, "feature")
    shap_values = explainer.shap_values(data, pruning_idx=pruned_idx, nsamples=nsamples)

    if model_feats is None:
        model_feats = ["Feature {}".format(i) for i in np.arange(data.shape[2])]

    model_feats = copy.deepcopy(model_feats)
    if pruned_idx > 0:
        model_feats += ["Pruned Events"]

    ret_data = []
    for exp, feature in zip(shap_values, model_feats):
        ret_data += [[random_seed, nsamples, feature, exp]]
    return pd.DataFrame(ret_data, columns=['Random seed', 'NSamples', 'Feature', 'Shapley Value'])


def local_cell_level(f: Callable[[np.ndarray], np.ndarray],
                     data: Union[pd.DataFrame, np.array],
                     cell_dict: dict,
                     event_data: pd.DataFrame,
                     feat_data: pd.DataFrame,
                     entity_uuid,
                     entity_col,
                     baseline,
                     pruned_idx,
                     ) -> pd.DataFrame:
    """Method to calculate event level explanations or load them if path is provided

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: pd.DataFrame
        Sequence to explain.

    cell_dict: dict
        Information required for the cell level explanation calculation

    event_data: pd.DataFrame
        Event level explanations.

    feat_data: pd.DataFrame
        Feature level explanations.

    entity_uuid: Union[str, int, float]
        The indentifier of the sequence that is being pruned.
        Used when fetching information from a csv of explanations

    entity_col: str
        Entity column to identify sequences

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.

    pruned_idx: int
        Index to prune the sequence. All events up to this point are grouped

    Returns
    -------
    pd.DataFrame
    """

    if cell_dict.get("path") is None or not os.path.exists(cell_dict.get("path")):
        ######print("No path to cell data provided. Calculating data")
        cell_data = cell_level(f, data, baseline, event_data, feat_data, cell_dict.get("rs"), cell_dict.get("nsamples"), cell_dict, pruned_idx)
        if cell_dict.get("path") is not None:
            # create directory
            if '/' in cell_dict.get("path"):
                Path(cell_dict.get("path").rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
            cell_data.to_csv(cell_dict.get("path"), index=False)
    elif cell_dict.get("path") is not None and os.path.exists(cell_dict.get("path")):
        cell_data = pd.read_csv(cell_dict.get("path"))
        if len(cell_data.columns) == 5 and entity_col is not None:
            cell_data = cell_data[cell_data[entity_col] == entity_uuid]
        elif len(cell_data.columns) == 4:
            pass
        else:
            # TODO
            # the provided csv should be generated by timeshap, by either
            # explaining the whole dataset with TODO or just the instance in question
            raise ValueError
    else:
        raise ValueError
    return cell_data


def cell_top_feats(feat_data: pd.DataFrame,
                   feat_threshold: float = None,
                   top_x_feats: int = None,
                   **kwargs,
                   ) -> Tuple[list, list]:
    """Calculates the indexes and names of events to participate in the cell level
    given the conditions

    Parameters
    ----------
    feat_data: pd.DataFrame
        Feature level explanations.

    feat_threshold: float
        The threshold to consider a feature relevant.

    top_x_feats: float
        Number of features to include as relevant

    Returns
    -------
    Tuple[list, list]
        Tuple containing two lists.
        The first list contains the feature indexes ordered ascendingly.
        The second list contains the correspondent feature names.
    """
    ordered_feats = feat_data.iloc[(-feat_data['Shapley Value'].abs()).argsort()].reset_index()

    top_feats_idx = []
    for _, row in ordered_feats.iterrows():
        if row['Feature'] == 'Pruned Events':
            # we want to skip previous events
            continue
        if feat_threshold is not None and abs(row['Shapley Value']) < feat_threshold:
            # we have reached the sopping point
            break
        top_feats_idx += [[row['index'], row['Feature']]]
        if top_x_feats is not None and len(top_feats_idx) == top_x_feats:
            # we have added enough events
            break

    df = pd.DataFrame(top_feats_idx, columns=['feat', 'name']).sort_values('feat')
    return list(df['feat'].values), list(df['name'].values)

def cell_top_events(event_data: pd.DataFrame,
                    event_threshold: float = None,
                    top_x_events: int = None,
                    **kwargs,
                    ) -> Tuple[list, list]:
    """Calculates the indexes and names of events to participate in the cell level
    given the conditions

    Parameters
    ----------
    event_data: pd.DataFrame
        Event level explanations.

    event_threshold: float
        The threshold to consider an event relevant.

    top_x_events: float
        Number of events to include as relevant

    Returns
    -------
    Tuple[list, list]
        Tuple containing two lists.
        The first list contains the event indexes ordered ascendingly.
        The second list contains the correspondent event names.
    """
    # order explanations by absolute contribution
    ordered_exp = event_data.iloc[(-event_data['Shapley Value'].abs()).argsort()].reset_index()

    top_events_idx = []
    for _, row in ordered_exp.iterrows():
        if row['Feature'] == 'Pruned Events':
            # we want to skip previous events
            continue

        top_events_idx += [[-row['index'] -1, row['Feature']]]

    df = pd.DataFrame(top_events_idx, columns=['idx', 'name']).sort_values('idx')
    
    return list(df['idx'].values), list(df['name'].values)



def considered_cells(event_data: pd.DataFrame,
                     feat_data: pd.DataFrame,
                     **kwargs,
                     ) -> Tuple[Tuple[List, List], Tuple[List, List]]:
    """Calculates the indexes and names of events to participate in the cell level
    given the conditions

    Parameters
    ----------
    event_data: pd.DataFrame
        Event level explanations.

    feat_data: pd.DataFrame
        Feature level explanations.

    Returns
    -------
    Tuple[Tuple[List, List], Tuple[List, List]]
        Events and features to include in cell level computations
        and their respective name
    """
    top_events_idx, top_events_names = cell_top_events(event_data, **kwargs)
    top_feats_idx, top_feats_names = cell_top_feats(feat_data, **kwargs)

    return (top_events_idx, top_feats_idx), (top_events_names,top_feats_names)


def cell_level(f: Callable,
               data: np.ndarray,
               baseline: Union[pd.DataFrame, np.ndarray],
               event_data: pd.DataFrame,
               feat_data: pd.DataFrame,
               random_seed: int,
               nsamples: int,
               cell_dict: dict,
               pruned_idx: int,
               model_feats=None,
               ) -> pd.DataFrame:
    """Cell level given relevant events and features

    Parameters
    ----------
    f : Callable
        Prediction method of model being explained.
        Will be called with a 3-D input

    data: numpy.ndarray
        Input matrix to use. First element of the first dimension is explained,
        using the rest of the elements as context/hidden state.

    baseline: numpy.ndarray
        Baseline event to use. Median of numerical and mode for categorical.

    event_data: pd.DataFrame
        Event level explanations.

    feat_data: pd.DataFrame
        Feature level explanations.

    random_seed: int
        Used random seed for the sampling process.

    nsamples: int
        The number of coalitions for TimeSHAP to sample.

    cell_dict: dict
        Information required for the cell level explanation calculation

    pruned_idx: int
        Index to prune the sequence. All events up to this point are grouped\

    model_feats: List
        The list of feature names.
        If none is provided, "Feature 1" format is used

    Returns
    -------
    pd.DataFrame
    """
    kwargs = {}
    if cell_dict.get('threshold', False):
        # single threshold for everything
        kwargs['event_threshold'] = cell_dict.get('threshold')
        kwargs['feat_threshold'] = cell_dict.get('threshold')
    elif cell_dict.get('event_threshold', False) and cell_dict.get('feat_threshold', False):
        kwargs['event_threshold'] = cell_dict.get('event_threshold')
        kwargs['feat_threshold'] = cell_dict.get('feat_threshold')
    elif cell_dict.get('top_x', False):
        kwargs['top_x_events'] = cell_dict.get('top_x')
        kwargs['top_x_feats'] = cell_dict.get('top_x')
    elif cell_dict.get('top_x_events', False) and cell_dict.get('top_x_feats', False):
        kwargs['top_x_events'] = cell_dict.get('top_x_events')
        kwargs['top_x_feats'] = cell_dict.get('top_x_feats')
    else:
        raise ValueError("No threshold condition provided for cell level")

    varying_cells, names = considered_cells(event_data, feat_data, **kwargs)

    negative_indexes = np.array([False if x >= 0 else True for x in varying_cells[0]])

    if any(negative_indexes):
        if not all(negative_indexes):
            raise ValueError("All indexes must be positive or negative. Not both")

        rows_filtered = list(np.any(data == 666, axis=-1)[0])
        max_T = rows_filtered.count(False)
        varying_cells = ([max_T + x for x in varying_cells[0]], varying_cells[1])


    explainer = TimeShapKernel(f, baseline, random_seed, "cell", varying=varying_cells)
    explanation = explainer.shap_values(data, pruning_idx=pruned_idx, nsamples=nsamples)

    ret_df_data = []
    i = 0
    for event in names[0]:
        for feat in names[1]:
            row = [event, feat, explanation[i]]
            i += 1
            ret_df_data += [row]

    if explainer.special_cells[0]:
        for event in names[0]:
            row = [event, 'Other Features', explanation[i]]
            i += 1
            ret_df_data += [row]

    if explainer.special_cells[1]:
        for feat in names[1]:
            row = ['Other Events', feat, explanation[i]]
            i += 1
            ret_df_data += [row]

    if explainer.special_cells[2]:
        ret_df_data += [["Other Events", "Other Features", explanation[i]]]
        i += 1
    if explainer.special_cells[3]:
        ret_df_data += [["Pruned Events", "Pruned Events", explanation[i]]]
    return pd.DataFrame(ret_df_data, columns=['Event', 'Feature',
                                              'Shapley Value']).sort_values(
        'Shapley Value', ascending=False)



def event_level(f: Callable,
                data: np.array,
                baseline: Union[np.ndarray, pd.DataFrame],
                pruned_idx: int,
                random_seed: int,
                nsamples: int,
                display_events: List[str] = None,
                ) -> pd.DataFrame:
    """Method to calculate event level explanations

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: np.array
        Sequence to explain.

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.
        The baseline can be an average event or an average sequence

    pruned_idx: int
        Index to prune the sequence. All events up to this index are grouped

    random_seed: int
        Used random seed for the sampling process.

    nsamples: int
        The number of coalitions for TimeSHAP to sample.

    display_events: List[str]
        In-order list of event names to be displayed

    Returns
    -------
    pd.DataFrame
    """
    explainer = TimeShapKernel(f, baseline, random_seed, "event")
    shap_values = explainer.shap_values(data, pruning_idx=pruned_idx, nsamples=nsamples)


    if display_events is None:
        display_events = ["Event {}".format(str(-int(i))) for i in np.arange(1, data.shape[1]-pruned_idx+1)]
    else:
        display_events = display_events[-len(shap_values)+1:]
    if pruned_idx > 0:
        display_events += ["Pruned Events"]

    ret_data = []
    for exp, event in zip(shap_values, display_events):
        ret_data += [[random_seed, nsamples, event, exp]]
    return pd.DataFrame(ret_data, columns=['Random seed', 'NSamples', 'Feature', 'Shapley Value'])

def local_event(f: Callable[[np.ndarray], np.ndarray],
                data: np.array,
                event_dict: dict,
                entity_uuid: Union[str, int, float],
                entity_col: str,
                baseline: Union[pd.DataFrame, np.array],
                pruned_idx: int,
                ) -> pd.DataFrame:
    """Method to calculate event level explanations or load them if path is provided

    Parameters
    ----------
    f: Callable[[np.ndarray], np.ndarray]
        Point of entry for model being explained.
        This method receives a 3-D np.ndarray (#samples, #seq_len, #features).
        This method returns a 2-D np.ndarray (#samples, 1).

    data: np.array
        Sequence to explain.

    event_dict: dict
        Information required for the event level explanation calculation

    entity_uuid: Union[str, int, float]
        The indentifier of the sequence that is being pruned.
        Used when fetching information from a csv of explanations

    entity_col: str
        Entity column to identify sequences

    baseline: Union[np.ndarray, pd.DataFrame],
        Dataset baseline. Median/Mean of numerical features and mode of categorical.
        In case of np.array feature are assumed to be in order with `model_features`.
        The baseline can be an average event or an average sequence

    pruned_idx: int
        Index to prune the sequence. All events up to this index are grouped

    Returns
    -------
    pd.DataFrame
    """
    if event_dict.get("path") is None or not os.path.exists(event_dict.get("path")):
        event_data = event_level(f, data, baseline, pruned_idx, event_dict.get("rs"), event_dict.get("nsamples"))
        if event_dict.get("path") is not None:
            # create directory
            if '/' in event_dict.get("path"):
                Path(event_dict.get("path").rsplit("/", 1)[0]).mkdir(parents=True, exist_ok=True)
            event_data.to_csv(event_dict.get("path"), index=False)
    elif event_dict.get("path") is not None and os.path.exists(event_dict.get("path")):
        event_data = pd.read_csv(event_dict.get("path"))
        if len(event_data.columns) == 5 and entity_col is not None:
            event_data = event_data[event_data[entity_col] == entity_uuid]
        elif len(event_data.columns) == 4:
            pass
        else:
            # TODO
            # the provided csv should be generated by timeshap, by either
            # explaining the whole dataset with TODO or just the instance in question
            raise ValueError
    else:
        raise ValueError
    
    return event_data

def calc_local_report(f: Callable[[np.ndarray], np.ndarray],
                      data: Union[pd.DataFrame, np.array],
                      pruning_dict: dict,
                      event_dict: dict,
                      feature_dict: dict,
                      cell_dict: dict = None,
                      baseline: Union[pd.DataFrame, np.ndarray] = None,
                      model_features: List[Union[int, str]] = None,
                      entity_col=None,
                      entity_uuid=None,
                      time_col=None,
                      verbose=False,
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:


    if isinstance(data, pd.DataFrame):
        if time_col is not None:
            data[time_col] = data[[time_col]].apply(pd.to_numeric)
            data = data.sort_values(time_col)
        if model_features is not None:
            data = data[model_features]
        else:
            data = data.values
        data = np.expand_dims(data.to_numpy().copy(), axis=0).astype(float)


    pruning_idx = 0
    coal_plot_data = None
    event_data = local_event(f, data, event_dict, entity_uuid, entity_col, baseline, pruning_idx)
    feature_data = local_feat(f, data, feature_dict, entity_uuid, entity_col, baseline, pruning_idx)

    if cell_dict:
        cell_data = local_cell_level(f, data, cell_dict, event_data, feature_data, entity_uuid, entity_col, baseline, pruning_idx)
    else:
        cell_data = None

    return coal_plot_data, event_data, feature_data, cell_data

def local_report(f: Callable[[np.ndarray], np.ndarray],
                 data: Union[pd.DataFrame, np.array],
                 pruning_dict: dict,
                 event_dict: dict,
                 feature_dict: dict,
                 cell_dict: dict = None,
                 baseline: Union[pd.DataFrame, np.array] = None,
                 model_features: List[Union[str, int]] = None,
                 entity_col: str = None,
                 entity_uuid: str = None,
                 time_col: str = None,
                 verbose=False,
                 ):

    pruning_data, event_data, feature_data, cell_level = \
        calc_local_report(f, data, pruning_dict, event_dict, feature_dict,
                          cell_dict, baseline, model_features, entity_col,
                          entity_uuid, time_col, verbose
                          )

    return event_data, feature_data, cell_level
