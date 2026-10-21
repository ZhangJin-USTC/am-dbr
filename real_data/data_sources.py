"""Public data-source registry for the OCC empirical study.

Raw data are not stored in this repository. The experiment loaders in the
framework download or query these public sources into a user-provided local
cache when reruns are requested.
"""

DATA_SOURCES = {
    "metro_traffic": {
        "label": "Metro Interstate Traffic Volume",
        "source": "UCI Machine Learning Repository, dataset 492",
        "url": "https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume",
    },
    "jena_climate": {
        "label": "Jena climate temperature",
        "source": "TensorFlow/Keras public copy of MPI-BGC Jena climate data",
        "url": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
    },
    "open_meteo_phoenix_daily": {
        "label": "Phoenix daily weather",
        "source": "Open-Meteo Historical Weather API",
        "url": "https://open-meteo.com/en/docs/historical-weather-api",
        "query": "latitude=33.45, longitude=-112.07, daily weather, 1980-2025",
    },
    "open_meteo_seattle_daily": {
        "label": "Seattle daily weather",
        "source": "Open-Meteo Historical Weather API",
        "url": "https://open-meteo.com/en/docs/historical-weather-api",
        "query": "latitude=47.61, longitude=-122.33, daily weather, 1980-2025",
    },
    "opsd_germany_daily": {
        "label": "German daily electricity consumption",
        "source": "Open Power System Data",
        "url": "https://github.com/jenfly/opsd/blob/master/opsd_germany_daily.csv",
        "note": "The executable loader uses the public daily mirror; cite OPSD and record the downloaded file hash.",
    },
}
