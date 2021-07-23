# This file encapsulates the various prediction algorithms that take a trip and return a label data structure
# Named "inferrers.py" instead of "predictors.py" to avoid a name collection in our abbreviated import convention

import logging
import random

import emission.analysis.modelling.tour_model.load_predict as lp

# A set of placeholder predictors to allow pipeline development without a real inference algorithm.
# For the moment, the system is configured to work with two labels, "mode_confirm" and
# "purpose_confirm", so I'll do that.

# The first placeholder scenario represents a case where it is hard to distinguish between
# biking and walking (e.g., because the user is a very slow biker) and hard to distinguish
# between work and shopping at the grocery store (e.g., because the user works at the
# grocery store), but whenever the user bikes to the location it is to work and whenever
# the user walks to the location it is to shop (e.g., because they don't have a basket on
# their bike), and the user bikes to the location four times more than they walk there.
# Obviously, it is a simplification.
def placeholder_predictor_0(trip):
    return [
        {"labels": {"mode_confirm": "bike", "purpose_confirm": "work"}, "p": 0.8},
        {"labels": {"mode_confirm": "walk", "purpose_confirm": "shopping"}, "p": 0.2}
    ]


# The next placeholder scenario provides that same set of labels in 75% of cases and no
# labels in the rest.
def placeholder_predictor_1(trip):
    return [
        {"labels": {"mode_confirm": "bike", "purpose_confirm": "work"}, "p": 0.8},
        {"labels": {"mode_confirm": "walk", "purpose_confirm": "shopping"}, "p": 0.2}
    ] if random.random() > 0.25 else []


# This third scenario provides labels designed to test the soundness and resilience of
# the client-side inference processing algorithms.
def placeholder_predictor_2(trip):
    # Timestamp2index gives us a deterministic way to match test trips with labels
    # Hardcoded to match "test_july_22" -- clearly, this is just for testing
    timestamp2index = {494: 5, 565: 4, 795: 3, 805: 2, 880: 1, 960: 0}
    timestamp = trip["data"]["start_local_dt"]["hour"]*60+trip["data"]["start_local_dt"]["minute"]
    index = timestamp2index[timestamp] if timestamp in timestamp2index else 0
    return [
        [

        ],
        [
            {"labels": {"mode_confirm": "bike", "purpose_confirm": "work"}, "p": 0.8},
            {"labels": {"mode_confirm": "walk", "purpose_confirm": "shopping"}, "p": 0.2}
        ],
        [
            {"labels": {"mode_confirm": "drove_alone"}, "p": 0.8},
        ],
        [
            {"labels": {"mode_confirm": "bike", "purpose_confirm": "work"}, "p": 0.8},
            {"labels": {"mode_confirm": "walk", "purpose_confirm": "shopping"}, "p": 0.2}
        ],
        [
            {"labels": {"mode_confirm": "walk", "purpose_confirm": "shopping"}, "p": 0.45},
            {"labels": {"mode_confirm": "walk", "purpose_confirm": "entertainment"}, "p": 0.35},
            {"labels": {"mode_confirm": "drove_alone", "purpose_confirm": "work"}, "p": 0.15},
            {"labels": {"mode_confirm": "shared_ride", "purpose_confirm": "work"}, "p": 0.05}
        ],
        [
            {"labels": {"mode_confirm": "walk", "purpose_confirm": "shopping"}, "p": 0.45},
            {"labels": {"mode_confirm": "walk", "purpose_confirm": "entertainment"}, "p": 0.35},
            {"labels": {"mode_confirm": "drove_alone", "purpose_confirm": "work"}, "p": 0.15},
            {"labels": {"mode_confirm": "shared_ride", "purpose_confirm": "work"}, "p": 0.05}
        ]
    ][index]


# This fourth scenario provides labels designed to test the expectation and notification system.
def placeholder_predictor_3(trip):
    timestamp2index = {494: 5, 565: 4, 795: 3, 805: 2, 880: 1, 960: 0}
    timestamp = trip["data"]["start_local_dt"]["hour"]*60+trip["data"]["start_local_dt"]["minute"]
    index = timestamp2index[timestamp] if timestamp in timestamp2index else 0
    return [
        [
            {"labels": {"mode_confirm": "bike", "purpose_confirm": "work"}, "p": 0.80},
            {"labels": {"mode_confirm": "walk", "purpose_confirm": "shopping"}, "p": 0.20}
        ],
        [
            {"labels": {"mode_confirm": "bike", "purpose_confirm": "work"}, "p": 0.80},
            {"labels": {"mode_confirm": "walk", "purpose_confirm": "shopping"}, "p": 0.20}
        ],
        [
            {"labels": {"mode_confirm": "drove_alone", "purpose_confirm": "entertainment"}, "p": 0.70},
        ],
        [
            {"labels": {"mode_confirm": "bike", "purpose_confirm": "work"}, "p": 0.96},
            {"labels": {"mode_confirm": "walk", "purpose_confirm": "shopping"}, "p": 0.04}
        ],
        [
            {"labels": {"mode_confirm": "walk", "purpose_confirm": "shopping"}, "p": 0.45},
            {"labels": {"mode_confirm": "walk", "purpose_confirm": "entertainment"}, "p": 0.35},
            {"labels": {"mode_confirm": "drove_alone", "purpose_confirm": "work"}, "p": 0.15},
            {"labels": {"mode_confirm": "shared_ride", "purpose_confirm": "work"}, "p": 0.05}
        ],
        [
            {"labels": {"mode_confirm": "walk", "purpose_confirm": "shopping"}, "p": 0.60},
            {"labels": {"mode_confirm": "walk", "purpose_confirm": "entertainment"}, "p": 0.25},
            {"labels": {"mode_confirm": "drove_alone", "purpose_confirm": "work"}, "p": 0.11},
            {"labels": {"mode_confirm": "shared_ride", "purpose_confirm": "work"}, "p": 0.04}
        ]
    ][index]

# Placeholder that is suitable for a demo.
# Finds all unique label combinations for this user and picks one randomly
def placeholder_predictor_demo(trip):
    import random

    import emission.core.get_database as edb
    user = trip["user_id"]
    unique_user_inputs = edb.get_analysis_timeseries_db().find({"user_id": user}).distinct("data.user_input")
    random_user_input = random.choice(unique_user_inputs) if random.randrange(0,10) > 0 else []

    logging.debug(f"In placeholder_predictor_demo: found {len(unique_user_inputs)} for user {user}, returning value {random_user_input}")
    return [{"labels": random_user_input, "p": random.random()}]

# Non-placeholder implementation. First bins the trips, and then clusters every bin
# See emission.analysis.modelling.tour_model for more details
# Assumes that pre-built models are stored in working directory
# Models are built using evaluation_pipeline.py and build_save_model.py
def predict_two_stage_bin_cluster(trip):
    return lp.predict_labels(trip)

# Helper function for section_to_trip_mode predictor
# Transforms a dictionary of PredictedModeTypes : probability to a dictionary of trip_confirm_options.MODE : probability
def map_raw_mode_to_rich_mode(trip, raw_dict):
    import emission.core.wrapper.modeprediction as ecwm
    # TODO calculate this based on the user's prior trip distribution (that's why we pass in the trip object)
    transform = {
        ecwm.PredictedModeTypes.UNKNOWN: {},
        ecwm.PredictedModeTypes.WALKING: {"walk": 1},
        ecwm.PredictedModeTypes.BICYCLING: {"bike": 0.4, "bikeshare": 0.2, "pilot_ebike": 0.4},  # Maybe we could distinguish between bike and ebike by speed or variability in speed or something?
        ecwm.PredictedModeTypes.BUS: {"bus": 1},
        ecwm.PredictedModeTypes.TRAIN: {"train": 1},
        ecwm.PredictedModeTypes.CAR: {"drove_alone": 0.6, "shared_ride": 0.3, "taxi": 0.05, "free_shuttle": 0.05},
        ecwm.PredictedModeTypes.AIR_OR_HSR: {}  # Currently we don't seem to have a MODE option for this
    }

    rich_dict = {}
    for raw_mode,prob in raw_dict.items():
        transformed = transform[ecwm.PredictedModeTypes[raw_mode]]
        for rich_mode in transformed: transformed[rich_mode] *= prob
        rich_dict.update(transformed)
    return rich_dict


# Predict trip mode by finding the most significant section and using the previously inferred section modes
def section_to_trip_mode(trip):
    import numpy as np
    import pandas as pd

    import emission.storage.decorations.trip_queries as esdt
    import emission.storage.decorations.section_queries as esds

    # Formula to determine how significant a given section is compared to another
    # TODO tune this (by adjusting the coefficients or otherwise) or replace with an existing work
    sorter = lambda section: 1.0*section["data"]["duration"]+1.0*section["data"]["distance"]

    # Confirmedtrip has a "primary_section" field that is currently unused.
    # For now, I don't care about it; I'm trying to get this implemented very quickly.
    # TODO Later, we might apply sorter elsewhere and store the result in primary_section.
    sections = esdt.get_cleaned_sections_for_trip(trip["user_id"], trip["data"]["cleaned_trip"])
    sections.sort(key=sorter, reverse=True)

    # We calculate how sure we are that a given section is actually the primary section by first calculating
    # how significant it is by the metric defined above, applying an exponent to increase the larger values
    # and decrease the smaller values, and normalizing so the sum is 1. For instance, if our raw significances
    # by sorter are [5000, 4000, 1000], an exponent of 5.0 gives us confidences of [0.75, 0.25, 0.00].
    # An exponent of 2.7 would give us [0.64, 0.35, 0.01].
    # I made this procedure up; it might not be the best way to do this.
    confidences = np.array([[sorter(section)] for section in sections])  # This being a column vector will help us later
    confidences **= 5.0  # TODO the exponent can be tuned
    confidences /= np.sum(confidences)

    # Now we get a dictionary of predicted modes : probabilities for all sections, weight the mode probabilities by the section probabilities, and sum
    prediction_dicts = [esds.get_inferred_mode_entry(trip["user_id"], section.get_id())["data"]["predicted_mode_map"] for section in sections]
    prediction_matrix = pd.DataFrame(prediction_dicts)
    prediction_matrix *= confidences  # Weight each mode's probabilities by our confidence that that is the primary mode
    raw_predictions = prediction_matrix.sum(axis=0)  # Sum across sections
    np.testing.assert_almost_equal(raw_predictions.sum(), 1)

    # Convert PredictedModeTypes to trip_confirm_options.MODE and assemble into the prediction data structure
    rich_modes = map_raw_mode_to_rich_mode(trip, raw_predictions.to_dict())
    rich_modes_sorted = list(rich_modes.items())
    rich_modes_sorted.sort(key = lambda x: x[1], reverse=True)  # Sort descending by probability
    prediction = [{"labels": {"mode_confirm": mode}, "p": p} for mode,p in rich_modes_sorted]
    return prediction
