import h5py
from postprocessing import trim, despeckle, normalize, threshold
from analysis import porosity, two_point_probability
from store import to_json, plot_s2, plot_images

import os
import logging

import numpy as np

#@delayed
def load(filename):
    img = None
    with h5py.File(filename, "r") as f:
        img = f['data'][()][0, 0, :, :, :]
    return img


#@delayed
def process(img, tasks):
    for task in tasks:
        img = task(img)
    return img


#@delayed
def analyze(img, tasks):
    results = []
    for task in tasks:
        results.append(task(img))
    return results, img


#@delayed
def store(sample_results, tasks, dir, orig_dir):
    cov = [results[0][1] for results in sample_results]
    porosities = [results[0][0] for results in sample_results]
    imgs = [results[1] for results in sample_results]
    orig_por = plot_s2(dir, os.path.join(orig_dir, 'orig.csv'), cov)
    plot_images(dir, imgs)
    avg_por = np.mean(porosities)
    if 0.95*orig_por < avg_por < 1.05*orig_por:
        return False
    else:
        return True

def run_analysis_pipeline(dir, orig_dir):
    logging.basicConfig(filename=os.path.join(dir, 'process.log'), level=logging.DEBUG)
    logging.debug('Entered pipeline...')
    files = []
    for file in os.listdir(dir):
        if file.endswith(".hdf5"):
            files.append(os.path.join(dir, file))
    logging.debug('Got file names...')

    post_processing_tasks = [trim, despeckle, normalize, threshold]
    analysis_tasks = [porosity, two_point_probability]
    store_tasks = [to_json, plot_images, plot_s2]

    loaded = [load(f) for f in files]
    logging.debug('Loaded files...')
    processed = [process(img, post_processing_tasks) for img in loaded]
    logging.debug('Post-processed files')
    analyzed = [analyze(img, analysis_tasks) for img in processed]
    logging.debug('Analyzed files...')
    stored = store(analyzed, store_tasks, dir, orig_dir)
    logging.debug('Wrote data...')

    for f in files[1::]:
        os.remove(f)
    logging.debug('Wrote data...')
    #returns true if needs to be deleted.
	
    del loaded
    del processed
    del analyzed 

    return stored

    