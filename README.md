This code can be used to reproduce the results presented in the NIPS ML4Audio workshop paper: https://arxiv.org/abs/1712.00254.
Feel free to get in touch if you spot errors, have questions, or would like to discuss follow-up results (tt@cortilabs.com).

To get training:
- download the esc50 dataset from: https://github.com/karoldvl/ESC-50 or https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YDEPUT.
- install the necessary dependencies as listed below.
- clone the GitHub repository, and install the library by running ‘pip install .’ in the folder containing the setup.py file.
- process the ESC50 dataset by running (./) the ‘make_esc50_data.sh’ file in the ‘Experiments’ folder after making it executable (chmod +x) and specifying the necessary paths.
- train the MSTmodel for each of the folds by running the mst_learning.sh file in the ‘Experiments’ folder (…). Provide the path to the ‘features’ subfolder that is created during processing of the dataset.
- train the classifiers for each of the folds by running the classifier_training.sh file in the ‘Experiments’ folder (…). Provide the path to the ‘features’ subfolder that is created during processing of the dataset.
- use the ‘retrieve_fold_results’ function in the ‘Utilities’ folder to collect and average the results over the folds. Plotting was done using standard matplotlib functions.

Dependencies
- librosa (tested with version 0.5.0), pyaudio, pydub, numpy, pandas, scipy, tensorflow (tested with version 1.2).

Notes:
1. This is a work in progress and it’s not focussed on production environments (i.e. before using in any non-research application I would strongly advice to spend time on optimization).
2. Without access to a GPU, training these models might be prohibitively expensive. I have not tried this.
3. Multi-GPU training is not supported at the moment.
4. The code has been kept somewhat modular so that you can change individual components (model, trainer, dataset) relatively easily for running new experiments.
