# This file encapsulates the various ensemble algorithms that take a trip and a list of primary predictions and return a label data structure

import copy
import logging
import typing

import emission.core.wrapper.labelprediction as ecwl
import emission.core.wrapper.inferredtrip as ecwi

# This placeholder ensemble simply returns the first prediction that was run
def ensemble_first_prediction(trip, predictions):
    # Since this is not a real ensemble yet, we will not mark it as such
    # algorithm_id = ecwl.AlgorithmTypes.ENSEMBLE
    algorithm_id = ecwl.AlgorithmTypes(predictions[0]["algorithm_id"]);
    prediction = copy.copy(predictions[0]["prediction"])
    return algorithm_id, prediction

# If we get a real prediction, use it, otherwise fallback to the placeholder
def ensemble_real_and_placeholder(trip, predictions):
        if predictions[0]["prediction"] != []:
            sel_prediction = predictions[0]
            logging.debug(f"Found real prediction {sel_prediction}, using that preferentially")
            # assert sel_prediction.algorithm_id == ecwl.AlgorithmTypes.TWO_STAGE_BIN_CLUSTER
        else:
            sel_prediction = predictions[1]
            logging.debug(f"No real prediction found, using placeholder prediction {sel_prediction}")
            # Use a not equal assert since we may want to change the placeholder
            assert sel_prediction.algorithm_id != ecwl.AlgorithmTypes.TWO_STAGE_BIN_CLUSTER

        algorithm_id = ecwl.AlgorithmTypes(sel_prediction["algorithm_id"])
        prediction = copy.copy(sel_prediction["prediction"])
        return algorithm_id, prediction

# This is meant to be a general-purpose ensemble that works on any number and type of input predictions
def array_multiply_ensemble(trip:ecwi.Inferredtrip, predictions:typing.List[ecwl.Labelprediction], added_uncertainty:typing.Optional[float]=None, significance_threshold:typing.Optional[float]=None, max_entries:typing.Optional[float]=None) -> typing.Tuple[ecwl.AlgorithmTypes, dict]:
    '''
        Takes any number of predictions, pours the data into a big pile of array arithmetic, and hopefully produces some sort of "average" output.
        I made this procedure up; it might not be the best way to do this.
        
        Steps:
        1. Construct an ndarray representing probabilities for all possible label tuples, with all entries initialized to the same (nonzero) value
        2. For each input prediction:
            a. Multiply all p-values by (1-added_uncertainty)
            b. Construct an ndarray of the same format as above for it; distribute the remaining uncertainty (1-sum(p)) equally among the remaining entries
            c. Multiply this into the ndarray from step 1
        3. Normalize the ndarray so the sum is 1
        4. Discard entries below SIGNIFICANCE_THRESHOLD and redistribute their p to dispose of much of the added uncertainty
        5. Convert back to the list-dictionary format and limit the length of the list to MAX_ENTRIES
        
        This method is feasible when we have only three label categories, but generating every single possibility might become prohibitively expensive if many more dimensions are added.
        
        The concept of "uncertainty" refers to the difference between the sum of the p-values explicitly included in a given prediction and 1.
        Here, we interpret uncertainty to mean that the label tuples not referred to in the prediction are all of equal probability and sum to make up that difference.

        :param trip: the trip object -- not currently used, but part of the standard interface for ensemble functions
        :param predictions: the list of primary predictions we will assemble into an ensemble
        :param added_uncertainty: forces a certain "open-mindedness," e.g., preventing tuples from being completely ruled out just because one prediction doesn't allow for them. If None, uses a sensible default.
        :param significance_threshold: filters out possibilities that are vanishingly unlikely, speeds up the algorithm, and allows us to mostly eliminate the negative side effects of ADDED_UNCERTAINTY. If None, uses a sensible default.
        :param max_entries: a final safeguard against the output prediction data structure being enormous. A three-label prediction list entry represented as JSON in ASCII is roughly 120 bytes. If None, uses a sensible default.
    '''

    # Defaults
    if added_uncertainty is None: added_uncertainty = 0.01
    if significance_threshold is None: significance_threshold = 0.002
    if max_entries is None: max_entries = 200

    # Imports
    import numpy as np
    import emission.analysis.configs.label_config as eacl

    # Initialization
    eacl._test_options["use_sample"] = True  # While our placeholder algorithms only use two label categories, we should here too
    eacl.reload_config()
    labels = eacl.labels
    name2index = {label: {name: i for i,name in enumerate(labels[label])} for label in labels}
    nd_shape = tuple(len(labels[label]) for label in labels)
    n_entries = np.prod(nd_shape)
    p_matrix = np.ones(nd_shape)

    uncertainty_multiplier = 1-added_uncertainty
    for pred_obj in predictions:
        pred_list = pred_obj["prediction"]

        # Add uncertainty
        for e in pred_list: e["p"] = e["p"]*uncertainty_multiplier

        # Initialize ndarray with leftover uncertainty (whether added or preexisting)
        leftover_uncertainty = 1-sum([e["p"] for e in pred_list])
        assert leftover_uncertainty >= added_uncertainty-0.00001  # Should be == in the case of zero preexisting uncertainty, > otherwise; allow for floating-point error
        pred_matrix = np.ones(nd_shape)*(leftover_uncertainty/n_entries)

        # Figure out which part of the ndarray each item of the prediction refers to and distribute the item's p evenly over those entries
        for pred_item in pred_list:
            slicer = []
            for label in labels:
                if label in pred_item["labels"]:
                    index = name2index[label][pred_item["labels"][label]]
                    slicer.append(slice(index, index+1))
                else:  # If pred_item omits one of the label categories, apply it to everything in that category
                    slicer.append(slice(None, None))
            slicer = tuple(slicer)
            n_sliced = np.prod(np.shape(pred_matrix[slicer]))
            assert n_sliced != 0  # All our slices should refer to something
            pred_matrix[slicer] = pred_item["p"]/n_sliced

        # Multiply the current prediction onto the ensemble
        p_matrix *= pred_matrix
    
    # Normalize
    p_matrix /= np.sum(p_matrix)
    # Zero out entries under the threshold and renormalize to dispose of some (typically, most) of the added uncertainty
    p_matrix = np.where(p_matrix >= significance_threshold, p_matrix, 0)
    p_matrix /= np.sum(p_matrix)
    # Get all entries that pass the threshold
    sig_indices = np.argwhere(np.asarray(p_matrix >= significance_threshold))  # Works regardless of whether the zeroing out step is included

    # Convert to the list-dictionary format
    ensemble = []
    for s_index in sig_indices:
        label_dict = {label: labels[label][s_index[i]] for i,label in enumerate(labels)}
        p = p_matrix[tuple(s_index)]
        ensemble.append({"labels": label_dict, "p": p})

    # Report only the MAX_ENTRIES most likely entries
    ensemble.sort(key = lambda x: x["p"], reverse=True)
    ensemble = ensemble[0:max_entries]

    return (ecwl.AlgorithmTypes.ENSEMBLE, ensemble)
