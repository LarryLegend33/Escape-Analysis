import csv
import numpy as np

directory = 'C:/Users/Deadpool/Desktop/EscapeExperimentGenerator/'
freerun_mins = "2"

dark_tap = "dark,tap"
light_tap_barrier = "light,tap"
light_tap_nobarrier = "nb_light,tap"
loom_virtual = "light,v_loom"
loom_barrier = "light,b_loom"
loom_left = "light,l_loom"
loom_left_tap = "light,L_loom"
loom_right_tap = "light,R_loom"
loom_right = "light,r_loom"
freerun_light_barrier = "light,freerun," + freerun_mins
freerun_light_nobarrier = "nb_light,freerun," + freerun_mins
freerun_dark = "dark,freerun," + freerun_mins
light_tap_virtualbarrier = "v_light,tap"
light_tap_virtualbarrier_inverted = "iv_light,tap"

#append a trial number based on the function you write


# switch keys to actual final names of stimuli for clarity
exp_dict = {"dark_tap": dark_tap,
            "light_tap_barrier": light_tap_barrier, 
            "light_tap_nobarrier": light_tap_nobarrier,
            "light_tap_virtualbarrier": light_tap_virtualbarrier, 
            "light_tap_virtualbarrier_inverted": light_tap_virtualbarrier_inverted,
            "loom_virtual": loom_virtual,
            "loom_barrier": loom_barrier,
            "loom_left": loom_left, 
            "loom_right": loom_right,
            "freerun_light_barrier": freerun_light_barrier,
            "freerun_light_nobarrier": freerun_light_nobarrier,
            "freerun_dark": freerun_dark,
            "loom_left_tap": loom_left_tap,
            "loom_right_tap": loom_right_tap}
            
def generate_minefield_control(dic, numtrials, stimtype):
    if stimtype == 'loom':
        stim = ""
    elif stimtype == 'tap':
        stim = "light_tap_nobarrier"
    with open(directory + 'control_experiment.csv', 'wb') as csvfile:
        exp_file = csv.writer(csvfile, delimiter=' ', quotechar='|', escapechar=' ', quoting=csv.QUOTE_NONE)
        exp_file.writerow([dic["freerun_light_nobarrier"]])
        for i in range(numtrials):
            exp_file.writerow([dic[stim] + "," + str(i)])


def generate_biasing_looms(dic, numlooms, numtaps, direction):
    with open(directory + 'loom_training.csv', 'wb') as csvfile:
        exp_file = csv.writer(csvfile, delimiter=' ', quotechar='|', escapechar=' ', quoting=csv.QUOTE_NONE)
  #      exp_file.writerow([dic[8]])
        for i in range(numlooms):
            if direction == "l":
                exp_file.writerow([dic["loom_left"] + "," + str(i)])
            elif direction == "r":
                exp_file.writerow([dic["loom_right"] + "," + str(i)])
            elif direction == "L":
                exp_file.writerow([dic["loom_left_tap"] + "," + str(i)])
            elif direction == "R":
                exp_file.writerow([dic["loom_right_tap"] + "," + str(i)])
        for j in range(numtaps):
            exp_file.writerow([dic["light_tap_nobarrier"] + "," + str(j)])

def generate_lightdark_barriers(dic, numtrials, freeruns):
    with open(directory + 'lightdark_experiment.csv', 'wb') as csvfile:
        num_trialtypes = 2
        trialcounter = np.zeros(num_trialtypes).astype(np.int)
        exp_file = csv.writer(csvfile, delimiter=' ', quotechar='|', escapechar=' ', quoting=csv.QUOTE_NONE)
        if freeruns:
            exp_file.writerow([dic["freerun_light_barrier"]])
            exp_file.writerow([dic["freerun_dark"]])
        triallist = randomize_trials(numtrials, 2)
        for trial in triallist:
            if trial == 0:
                stim = "dark_tap"                
            elif trial == 1:
                stim = "light_tap_barrier"                
            exp_file.writerow([dic[stim] + "," + str(trialcounter[trial])])
            trialcounter[trial] += 1

def generate_virtualbarriertrials(dic, numtrials, stimtype, inversions): 
    if stimtype == 'loom':
        stim = "loom_virtual"
    num_trialtypes = 3
    triallist = randomize_trials(numtrials, num_trialtypes)    
    trialcounter = np.zeros(num_trialtypes).astype(np.int)
    with open(directory + 'virtualbarrier_experiment.csv', 'wb') as csvfile:
        exp_file = csv.writer(csvfile, delimiter=' ', quotechar='|', escapechar=' ', quoting=csv.QUOTE_NONE)
        for trial in triallist:
            if trial == 0:
                stim = "light_tap_virtualbarrier"
            elif trial == 1: 
                stim = "light_tap_virtualbarrier_inverted"
            elif trial == 2:
                stim = "light_tap_nobarrier"
            exp_file.writerow([dic[stim] + "," + str(trialcounter[trial])])
            trialcounter[trial] += 1

def randomize_trials(total_trials, num_different_trials):
    trial_list = []
    typecounts = np.zeros(num_different_trials)
    trialcounter = 0
    while True:
        if trialcounter == total_trials:
            break
        trial = np.random.randint(0, num_different_trials)
        if typecounts[trial] != total_trials / num_different_trials:
            trial_list.append(trial)
            typecounts[trial] += 1
            trialcounter += 1
    return trial_list

generate_minefield_control(exp_dict, 10, "tap")
generate_lightdark_barriers(exp_dict, 10, True)
generate_biasing_looms(exp_dict, 10, 2, "L")
generate_virtualbarriertrials(exp_dict, 24, 'tap', False)
