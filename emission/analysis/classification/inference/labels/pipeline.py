# Standard imports
import logging
import copy

# Our imports
import emission.storage.pipeline_queries as epq
import emission.storage.timeseries.abstract_timeseries as esta
import emission.storage.decorations.analysis_timeseries_queries as esda
import emission.core.wrapper.labelprediction as ecwl
import emission.core.wrapper.entry as ecwe
import emission.analysis.classification.inference.labels.inferrers as eacili
import emission.analysis.classification.inference.labels.ensembles as eacile

# Select which type of logging predictions get sent to
PREDICTIONS_LOGGER = logging.info

# For each algorithm in ecwl.AlgorithmTypes that runs on a trip (e.g., not the ensemble, which
# runs on the results of other algorithms), primary_algorithms specifies a corresponding
# function in eacili to run. This makes it easy to plug in additional algorithms later.
primary_algorithms = {
    ecwl.AlgorithmTypes.SECTION_TO_TRIP_MODE: eacili.section_to_trip_mode,
    ecwl.AlgorithmTypes.PLACEHOLDER_2: eacili.placeholder_predictor_2,
}

# ensemble specifies which algorithm in eacile to run.
# This makes it easy to test various ways of combining various algorithms.
ensemble = eacile.array_multiply_ensemble


# Does all the work necessary for a given user
def infer_labels(user_id):
    time_query = epq.get_time_range_for_label_inference(user_id)
    try:
        lip = LabelInferencePipeline()
        lip.user_id = user_id
        lip.run_prediction_pipeline(user_id, time_query)
        if lip.last_trip_done is None:
            logging.debug("After run, last_trip_done == None, must be early return")
        epq.mark_label_inference_done(user_id, lip.last_trip_done)
    except:
        logging.exception("Error while inferring labels, timestamp is unchanged")
        epq.mark_label_inference_failed(user_id)

# Code structure based on emission.analysis.classification.inference.mode.pipeline
# and emission.analysis.classification.inference.mode.rule_engine
class LabelInferencePipeline:
    def __init__(self):
        self._last_trip_done = None
    
    @property
    def last_trip_done(self):
        return self._last_trip_done

    # For a given user and time range, runs all the primary algorithms and ensemble, saves results
    # to the database, and records progress
    def run_prediction_pipeline(self, user_id, time_range):
        self.ts = esta.TimeSeries.get_time_series(user_id)
        self.toPredictTrips = esda.get_entries(
            esda.CLEANED_TRIP_KEY, user_id, time_query=time_range)
        
        n = 0  # For debugging
        for cleaned_trip in self.toPredictTrips:
            # Create an inferred trip
            cleaned_trip_dict = copy.copy(cleaned_trip)["data"]
            inferred_trip = ecwe.Entry.create_entry(user_id, "analysis/inferred_trip", cleaned_trip_dict)
            inferred_trip["data"]["cleaned_trip"] = cleaned_trip.get_id()
            
            # Run the algorithms and the ensemble, store results
            results = self.compute_and_save_algorithms(inferred_trip)
            ensemble = self.compute_and_save_ensemble(inferred_trip, copy.deepcopy(results))

            # Log predictions to the specified destination
            if PREDICTIONS_LOGGER is not None:
                PREDICTIONS_LOGGER(f"Trip {n}, inferred trip ID={inferred_trip.get_id()}:")
                for i,k in enumerate(primary_algorithms): PREDICTIONS_LOGGER(f"{k}:\n{results[i]['prediction']}")
                PREDICTIONS_LOGGER("ensemble:")
                for entry in ensemble['prediction']: PREDICTIONS_LOGGER(entry)
                PREDICTIONS_LOGGER("")
            n += 1

            # Put final results into the inferred trip and store it
            inferred_trip["data"]["inferred_labels"] = ensemble["prediction"]
            self.ts.insert(inferred_trip)

            if self._last_trip_done is None or self._last_trip_done["data"]["end_ts"] < cleaned_trip["data"]["end_ts"]:
                self._last_trip_done = cleaned_trip
    
    # This is where the labels for a given trip are actually predicted.
    # Though the only information passed in is the trip object, the trip object can provide the
    # user_id and other potentially useful information.
    def compute_and_save_algorithms(self, trip):
        predictions = []
        for algorithm_id, algorithm_fn in primary_algorithms.items():
            prediction = algorithm_fn(trip)
            lp = ecwl.Labelprediction()
            lp.trip_id = trip.get_id()
            lp.algorithm_id = algorithm_id
            lp.prediction = prediction
            lp.start_ts = trip["data"]["start_ts"]
            lp.end_ts = trip["data"]["end_ts"]
            self.ts.insert_data(self.user_id, "inference/labels", lp)
            predictions.append(lp)
        return predictions

    # Combine all our predictions into a single ensemble prediction
    def compute_and_save_ensemble(self, trip, predictions):
        il = ecwl.Labelprediction()
        il.trip_id = trip.get_id()
        il.start_ts = trip["data"]["start_ts"]
        il.end_ts = trip["data"]["end_ts"]
        (il.algorithm_id, il.prediction) = ensemble(trip, predictions)
        self.ts.insert_data(self.user_id, "analysis/inferred_labels", il)
        return il
