run_to_analyse = f"Run4"
data_directory = f"/jupyter-workspace/cnaf-storage/cygno-analysis/RECO/{run_to_analyse}"

wrong_to_wright_position_dict={  3.5:5,
                                10.5:15,
                                17.5:25,
                                24.5:35,
                                32.5:46.5}

correct_position_to_step_dict = { 5    :'step1',   # converts step to cm https://github.com/CYGNUS-RD/WIKI-documentation/wiki/Detector-General
                                  15   :'step2',
                                  25   :'step3',
                                  35   :'step4',
                                  46.5 :'step5'}

# matches the Run number to the start and stop numbers of the datasets.
lime_underground_run_numbers_dict = {'Run 1': [ 3000,   5291],
                                     'Run 2': [ 7792,  11175],
                                     'Run 3': [17362,  39385],
                                     'Run 4': [40784,  55101],
                                     'Run 5': [56894, 120893]}

steps_of_interest = [
    'Daily Calibration, step 5',
    'Daily Calibration, step 4',
    'Daily Calibration, step 3',
    'Daily Calibration, step 2',
    'Daily Calibration, step 1',
    'Daily Calibration, step 5, Low Gain',
    'Daily Calibration, step 4, Low Gain',
    'Daily Calibration, step 3, Low Gain',
    'Daily Calibration, step 2, Low Gain',
    'Daily Calibration, step 1, Low Gain',
    'Daily Calibration, step5, Low Gain',
    'Daily Calibration, step4, Low Gain',
    'Daily Calibration, step3, Low Gain',
    'Daily Calibration, step2, Low Gain',
    'Daily Calibration, step1, Low Gain',
    'Daily Calibration - LOW Gain, step 5',
    'Daily Calibration - LOW Gain, step 4',
    'Daily Calibration - LOW Gain, step 3',
    'Daily Calibration - LOW Gain, step 2',
    'Daily Calibration - LOW Gain, step 1',
    'S008:DATA:Fe Calibration, Fe Step 5',
    'S008:DATA:Fe Calibration, Fe Step 4',
    'S008:DATA:Fe Calibration, Fe Step 3',
    'S008:DATA:Fe Calibration, Fe Step 2',
    'S008:DATA:Fe Calibration, Fe Step 1',
    'S008:DATA:BKG Calibration, Fe Step 5',
    'S008:DATA:BKG Calibration, Fe Step 4',
    'S008:DATA:BKG Calibration, Fe Step 3',
    'S008:DATA:BKG Calibration, Fe Step 2',
    'S008:DATA:BKG Calibration, Fe Step 1',
]
