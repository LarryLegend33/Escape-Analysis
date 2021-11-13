import os
import numpy as np
import math
from matplotlib import pyplot as pl
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
import seaborn as sb
import copy
import cv2
import toolz
import scipy.ndimage
import imageio
import pickle
import itertools
from collections import deque
from toolz.itertoolz import sliding_window, partition
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelmin, argrelmax, argrelextrema
from scipy.stats import ttest_ind, ttest_rel
from astropy.convolution import convolve, Gaussian1DKernel
import itertools as itz
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.image import AxesImage
from matplotlib.colors import Colormap
import pdb
import matplotlib


# This is the main class for this program.
# Escape objects take in barrier locations, XY coords of the fish,
# movies during taps, raw stim files (i.e. the position of the
# tapper light over time), and relevant bacgkrounds for each
# tap trial. The Escape class contains methods for finding the fish's
# orientation relative to a barrier, finding the timing and angle
# of the c-start by calculating the exact moment of the tap,
# and plotting the escape trajectory relative to a barrier position


# initial results computed with cstart_time_thresh = 150, angle thresh = 30, std = 1





matplotlib.rcParams['pdf.fonttype'] = 42


def ts_plot(list_of_lists, ax):
#    fig, ax = pl.subplots(1, 1)
    index_list = list(
        itertools.chain.from_iterable(
            [range(len(arr)) for arr in list_of_lists]))
    id_list = [(ind*np.ones(len(arr))).tolist() for ind, arr in enumerate(list_of_lists)]
    ids_concatenated = list(itertools.chain.from_iterable(id_list))
    #this works if passed np.arrays instead of lists
    value_list = list(itertools.chain.from_iterable(list_of_lists))
    df_dict = {'x': index_list, 
               'y': value_list}
    df = pd.DataFrame(df_dict)
    sb.lineplot(data=df, x='x', y='y', ax=ax)
   # pl.show()
    return df


class Condition_Collector:
    def __init__(self, condition):
        self.condition = condition
        self.trajectory_stat_start = 20
        self.escape_data = {'Heading vs Barrier': [],
                            'Distance From Barrier After Escape': [],
                            'Collision Percentage': [],
                            'Total Valid Trials': 0,
                            'Total Collisions': 0,
                            'CStart Latency': [],
                            'CStart Angle': [],
                            'No Escape': 0,
                            'Phototaxis to Tap Time': [],
                            'Correct CStart Percentage': [],
                            'Total Correct Trajectories': [0, 0],
                            'Correct Trajectory Percentage': [],
                            'Correct Trajectory Percentage BLeft': [],
                            'Correct Trajectory Percentage BRight': [],
                            'Collision Trials BLeft': [],
                            'Collision Trials BRight': [],
                            'Total Correct CStarts': 0,
                            'Total CStarts': 0,
                            'Left vs Right CStarts': [],
                            'Left Vs Right Trajectories': [0, 0],
                            'Left Traj Percentage': [],
                            'CStart Rel to Prevbout': [],
                            'Taps Per Entry Into Arena': [],
                            'Total Time In Center': [],
                            'Barrier On Left Trajectories': [],
                            'Barrier On Right Trajectories': [],
                            'CStarts Per Trial': [0, 0]}
        self.timerange = []
        self.filter_index = 0
        self.filter_by_led = True
        # here change filter index to 1 for mag, 0 for hvsb.
        # set heading vs barrier to pos or neg.
        # filter_index 0 is h_vs_b. if neg, barrier is on left.
        # self.filter_range = [90, 200]
        self.rejected_fish = []
        self.included_fish = []
        self.filter_cstart = np.nan
        # 0 to 180 for barrier on right, -180 to 0 for barrier on left. empty for no filter.
        self.filter_function = lambda x: x
        self.velocity_threshold = 0
        self.pre_c = 10
        self.all_velocities = []

    def update_ddict(self, escape_obj):

        def wallfilter(xycoord):
            distance_from_center = magvector(
                [xycoord[0] - 640, xycoord[1] - 512])
            if distance_from_center > 400:
                return True
            else:
                return False

        def velocity_filter(xyrec):
            dx = np.diff(xyrec[0][self.timerange[0]+self.pre_c:self.timerange[1]])
            dy = np.diff(xyrec[1][self.timerange[0]+self.pre_c:self.timerange[1]])
            velocities = [magvector([xd, yd]) for xd, yd in zip(dx, dy)]
            self.all_velocities.append(velocities)
            if np.max(velocities) < self.velocity_threshold:
                return False
            else:
                return True

        if self.filter_by_led:
            if sum(escape_obj.stim_times_accurate) == 0:
                self.rejected_fish.append(escape_obj.directory)
                return 0
            else:
                self.included_fish.append(escape_obj.directory)
        else:
            self.included_fish.append(escape_obj.directory)

        if escape_obj.condition != self.condition:
            raise Exception('input condition mistmatch with class')
        else:
            # 1 for distance, 0 for angle.
            if self.filter_index < 2:
                initial_filterval_to_barrier = [initial[self.filter_index] if initial != [] else np.nan for initial
                                                in escape_obj.initial_conditions]
                if self.filter_index == 0:
                    initial_filterval_to_barrier = list(map(np.rad2deg, initial_filterval_to_barrier))
                    
                if not math.isnan(self.filter_cstart):
                    trialfilter = [i for i, c in enumerate(
                        escape_obj.cstart_angles) if np.sign(c) == self.filter_cstart]
                    print("trialfilter")
                    print(trialfilter)

                trialfilter = [i for i, d in enumerate(
                    initial_filterval_to_barrier) if self.filter_function(d)]
                print(trialfilter)

            self.timerange = escape_obj.timerange
            tap_times = []
            last_x = []
            last_y = []
            last_xy = []
            barrier_xy_by_trial = []
            non_nan_cstarts = []
            non_nan_cstarts_rel_to_prevbout = []
            cstart_angles = []
            cstarts_executed = 0
            cstart_opportunities = 0
            non_nan_collisions = []
            # filter for specific trials that satisify filter conditions
            # get rid of wall trials if you input data that contains wall
            for trial in range(len(escape_obj.xy_coords_by_trial)):
                if escape_obj.xy_coords_by_trial[trial][0] == []:
                    if trial in trialfilter:
                        print("EMPTY XY COORDS")
                        trialfilter.remove(trial)
                if trial in trialfilter:
                    self.escape_data['Total Valid Trials'] += 1
                    if not velocity_filter(
                            escape_obj.xy_coords_by_trial[
                                trial]):
                        self.escape_data['No Escape'] += 1
                        trialfilter.remove(trial)
                        print("VELOCITY FILTER")
                if (trial not in trialfilter):
                    print("TRIAL NOT IN FILTER")
                    continue
                try:
                    if wallfilter(escape_obj.initial_conditions[trial][4]):
                        print("INSIDE WALLFILTER")
                        continue
                except IndexError:
                    pass
                gf = escape_obj.pre_escape[trial]
                # gf is grayframes (e.g. how many gray frames happened before the escape tap)
                try:
                    num_gfs = len(gf)
                except TypeError:
                    num_gfs = 1
                tap_times.append(num_gfs / 200.0)
                last_x.append(escape_obj.xy_coords_by_trial[trial][1][
                    escape_obj.timerange[1]])
                last_y.append(escape_obj.xy_coords_by_trial[trial][0][
                    escape_obj.timerange[1]])
                barrier_xy_by_trial.append(
                    escape_obj.barrier_xy_by_trial[trial])
                if not math.isnan(escape_obj.cstart_rel_to_barrier[trial]):
                    non_nan_cstarts.append(
                        escape_obj.cstart_rel_to_barrier[trial])
                if not math.isnan(escape_obj.cstart_rel_to_last_bout[trial]):
                    non_nan_cstarts_rel_to_prevbout.append(
                        escape_obj.cstart_rel_to_last_bout[trial])
                if not math.isnan(escape_obj.collisions[trial]):
                    non_nan_collisions.append(escape_obj.collisions[trial])
                cstart_angles.append(escape_obj.cstart_angles[trial])
                cstarts_executed += escape_obj.cstarts_per_trial[trial]
                cstart_opportunities += 1
                

            # if len(escape_obj.numgrayframes) != 0:
            #     self.escape_data['Taps Per Entry Into Arena'].append(
            #         len(escape_obj.xy_coords_by_trial) / float(
            #             len(escape_obj.numgrayframes)))
            #     self.escape_data[
            #         'Total Time In Center'] += escape_obj.numgrayframes.tolist()
            self.escape_data['CStarts Per Trial'][0] += cstarts_executed
            self.escape_data['CStarts Per Trial'][1] += cstart_opportunities
            cstart_direction = [np.sign(cs) for cs in cstart_angles if not math.isnan(cs)]
            if len(cstart_direction) > len(cstart_angles) / 2:
                self.escape_data['Left vs Right CStarts'].append(
                    cstart_direction.count(-1) / len(cstart_direction))
#            try:
 #               self.escape_data['Left vs Right CStarts'].append(
  #                  cstart_direction.count(-1) / len(cstart_direction))
   #         except ZeroDivisionError:
    #            self.escape_data['Left vs Right CStarts'].append(np.nan)
            self.escape_data['CStart Angle'] += np.abs(cstart_angles).tolist()
            bl, br, bl_trials, br_trials = escape_obj.escapes_vs_barrierloc(
                0, trialfilter)
            
            self.escape_data['Heading vs Barrier'] += escape_obj.h_vs_b_plot(
                0, trialfilter)

# UNCOMMENT IF YOU WANT ALL TRAJECTORIES NOT JUST CSTART CONFIRMED.
# It's best to keep all b/c trajectory analysis is less susceptible than
# cstart detection to errors. fish is basically always found. 
            self.escape_data['Barrier On Left Trajectories'] += bl
            self.escape_data['Barrier On Right Trajectories'] += br
            br_correct_moves = 0
            bl_correct_moves = 0
            total_right = 0
            total_left = 0
            btouch_after_correct_escape_br = []
            btouch_after_correct_escape_bl = []

            # introduce here an or that says the collision happened late.
            # this excludes trajectories where the fish swims correctly but
            # ends up back next to the barrier after a counterswim, which is not a collision.
            # (i.e. the escape direction is right, a barrier touch occurs at the end of the trajectory). 
            # these metrics now account for collisions that DEFLECT the fish in the right direction
            # and non-collisions that occur after a soft counterturn.

            btouch_win = 35

            # collision times are in frames from 100. 
            
            for bri, r_coords in enumerate(br):
                btouch_after_correct = False
                if np.sum(r_coords[0][self.trajectory_stat_start:]) < 0:
                    # if collision happens at frame 135 or later, likely a non-collision btouch if in the right dir.
                    # you can play with this val a bit to see if it captures your read of the white trials.
                    try:
                        if escape_obj.collision_times[br_trials[bri]][0] > btouch_win:
                            btouch_after_correct_escape_br.append(br_trials[bri])
                            btouch_after_correct = True
                    except IndexError:
                        pass
                    if escape_obj.collision_times[br_trials[bri]] == [] or btouch_after_correct:
                        br_correct_moves += 1
                        total_left += 1
                    else:
                        total_right += 1
                else:
                    total_right += 1


            for bli, l_coords in enumerate(bl):
                btouch_after_correct = False
                if np.sum(l_coords[0][self.trajectory_stat_start:]) > 0:
                    try:
                        if escape_obj.collision_times[bl_trials[bli]][0] > btouch_win:
                            btouch_after_correct_escape_bl.append(bl_trials[bli])
                            btouch_after_correct = True
                    except IndexError:
                        pass
                    if escape_obj.collision_times[bl_trials[bli]] == [] or btouch_after_correct:
                        bl_correct_moves += 1
                        total_right += 1
                    else:
                        total_left += 1
                else:
                    total_left += 1

            print("BARRIER TOUCH")
            print(btouch_after_correct_escape_br)
            print(btouch_after_correct_escape_bl)

            bleft_collision_times = [escape_obj.collision_times[tr] if tr not
                                     in btouch_after_correct_escape_bl else [] for tr in bl_trials]

            bright_collision_times = [escape_obj.collision_times[tr] if tr not
                                      in btouch_after_correct_escape_br else [] for tr in br_trials]
            
            self.escape_data['Collision Trials BLeft'] += bleft_collision_times
            self.escape_data['Collision Trials BRight'] += bright_collision_times
            total_collisions = sum(map(lambda x: x != [], bleft_collision_times + bright_collision_times))
            self.escape_data['Total Collisions'] += total_collisions
            self.escape_data['Collision Percentage'].append(np.float64(total_collisions) / len(non_nan_collisions))
                    
            print(trialfilter)
            print(self.filter_by_led)
            self.escape_data['Correct Trajectory Percentage'].append(
                np.float64(br_correct_moves + bl_correct_moves) / (len(br) + len(bl)))
            self.escape_data['Correct Trajectory Percentage BRight'].append(np.float64(br_correct_moves) / len(br))
            self.escape_data['Correct Trajectory Percentage BLeft'].append(np.float64(bl_correct_moves) / len(bl))
            self.escape_data['Total Correct Trajectories'][0] += (bl_correct_moves + br_correct_moves)
            self.escape_data['Total Correct Trajectories'][1] += (len(bl) + len(br)) - (bl_correct_moves + br_correct_moves)
            print(escape_obj.directory)
            self.escape_data['Left Vs Right Trajectories'][0] += total_left
            self.escape_data['Left Vs Right Trajectories'][1] += total_right
            self.escape_data['Left Traj Percentage'].append(np.float64(total_left) / (total_left + total_right))
            
            last_xy = zip(last_x, last_y)

            # This has to remove the radius of the barrier. 
            self.escape_data['Distance From Barrier After Escape'].append(
                [magvector([xyl[0] - bxy[0], xyl[1] - bxy[1]]) for
                 xyl, bxy in zip(last_xy, barrier_xy_by_trial)])
            
            self.escape_data['Phototaxis to Tap Time'] += tap_times
            self.escape_data['Total Correct CStarts'] += np.sum(non_nan_cstarts)
            self.escape_data['Total CStarts'] += len(non_nan_cstarts)
            self.escape_data['Correct CStart Percentage'].append(np.sum(
                non_nan_cstarts) / float(
                    len(non_nan_cstarts)))
            self.escape_data['CStart Rel to Prevbout'].append(np.sum(
                non_nan_cstarts_rel_to_prevbout) / float(
                    len(non_nan_cstarts_rel_to_prevbout)))


# THIS IS WRONG. THIS IS THE ONSET OF THE STIMULUS!            
            # self.escape_data[
            #     'CStart Latency'] += np.array(escape_obj.stim_init_times)[trialfilter][
            #         np.array(escape_obj.stim_times_accurate).astype(np.bool)[trialfilter]].tolist()
            self.escape_data[
                'CStart Latency'] += np.array(escape_obj.escape_latencies)[trialfilter][
                    np.array(escape_obj.stim_times_accurate).astype(np.bool)[trialfilter]].tolist()

            
    def convert_to_nparrays(self):
        np_array_dict = {}
        for ky, it in self.escape_data.items():
            np_array_dict[ky] = np.array(it)
        return np_array_dict


# first step would be to check if latency stats are even taking everything into account     

class Escapes:

    def __init__(self, exp_type, directory, area_thresh, *sub_class):
        self.pre_c = 10
        self.area_thresh = area_thresh
        self.directory = directory
        self.timerange = [100, 150]
        self.condition = exp_type
        self.xy_coords_by_trial = []
        self.missed_inds_by_trial = []
        self.contours_by_trial = []
    # this asks whether there is a bias in direction based on the HBO. 
        self.stim_init_times = []
        self.stim_times_accurate = []
        self.escape_latencies = []
        self.collisions = []
        self.collision_times = []
        self.collision_bound = 50
        self.cstart_filter_std = .5
        self.cstart_angle_thresh = 50
        self.cstart_time_thresh = 150
        if exp_type in ['l', 'd']:
            bstruct_and_br_label = 'b'
        elif exp_type in ['v', 'i']:
            self.numgrayframes = []
            bstruct_and_br_label = 'v'
        if exp_type == 'n':
            try:
                self.numgrayframes = np.loadtxt(
                    directory + '/numframesgray_n.txt',
                    dtype='str').astype(np.int)
                self.barrier_file = np.loadtxt(
                    directory + '/barrierstruct_n.txt',
                    dtype='str')
            # Will throw IO error for n trials in virtual settings
            except IOError:
                self.numgrayframes = []
                self.barrier_file = np.loadtxt(
                    directory + '/barrierstruct_v.txt',
                    dtype='str')

        else:
            self.barrier_file = np.loadtxt(
                directory + '/barrierstruct_' + bstruct_and_br_label + '.txt',
                dtype='str')

        if exp_type == 'l':
            self.numgrayframes = np.loadtxt(
                directory + '/numframesgray_' + bstruct_and_br_label + '.txt',
                dtype='str').astype(np.int)
        elif exp_type == 'd':
            self.numgrayframes = np.loadtxt(
                directory + '/numframesgray_dark.txt',
                dtype='str').astype(np.int)
            
        self.barrier_coordinates = []
        self.barrier_diam = 0
        self.barrier_xy_by_trial = []
        print(self.directory)
        background_files = sorted(
            [(directory + '/' + f_id, int(f_id[11:13]), f_id[-5:-4])
             for f_id in os.listdir(directory)
             if (f_id[0:10] == 'background' and f_id[12] != '.')])
        back_color_images = [cv2.imread(fil[0]) for fil in background_files]
        backgrounds_gray = [cv2.cvtColor(back_color,
                                         cv2.COLOR_BGR2GRAY)
                            for back_color in back_color_images]

        self.backgrounds = [(bg, bf[1], bf[2]) for bg, bf in zip(
            backgrounds_gray, background_files)]
        pre_escape_files = sorted([directory + '/' + f_id
                                   for f_id in os.listdir(directory)
                                   if f_id[0:15] == 'fishcoords_gray'
                                   and f_id[-5:-4] == exp_type])
        self.pre_escape = [
            np.loadtxt(pe, dtype='str') for pe in pre_escape_files]
        self.pre_escape_bouts = []
        xy_files = sorted([directory + '/' + f_id
                           for f_id in os.listdir(directory)
                           if f_id[0:4] == 'tapr' and f_id[-5:-4] == exp_type])
        self.xy_escapes = [np.loadtxt(xyf, dtype='str') for xyf in xy_files]
        stim_files = sorted([directory + '/' + f_id
                             for f_id in os.listdir(directory)
                             if f_id[0:4] == 'stim'
                             and f_id[-5:-4] == exp_type])
        self.stim_data = [np.loadtxt(st, dtype='str') for st in stim_files]
        self.movie_id = sorted([directory + '/' + f_id
                                for f_id in os.listdir(directory)
                                if (f_id[-5:] == exp_type+'.AVI'
                                    and f_id[0] == 't')])

# Make these dictionaries so you can put in arbitrary trial / value bindings
        self.cstart_rel_to_last_bout = []
        self.cstart_angles = []
        self.cstart_rel_to_barrier = []
        self.cstarts_per_trial = []
        self.all_tailangles = []
        self.tailangle_sums = []
        self.ha_in_timeframe = []
        self.ba_in_timeframe = []
        self.h_vs_b_by_trial = []
        self.initial_conditions = []
        if sub_class == ():
            self.load_experiment()

    # this function simply plots the heading angle vs. the barrier angle
    # over time. the barrier angle runs from the fish's center of mass
    # to the center of the barrier it is escaping from.
    # 0 h_vs_b would indicate the fish pointing at the barrier while
    # pi is perfectly away. 
        
    def h_vs_b_plot(self, plot_fish, *trialfilter):
        # h_vs_b_mod = [np.mod
        #     hb_trial, 2*np.pi) for hb_trial in self.h_vs_b_by_trial]


# if hb_trial[0] is negative, meaning barrier is on left, multiply the trial by -1?
# currently the h_vs_b is on a -180 to 180 scale with 0 as barrier directly in front, negatives on the left.
# abs does do the trick of seeing if the absolute angle grows, but misses crossings (i.e. -60 to 80 would be the same as -60 to -80, which is a
# much lower amplitude turn. TRY:

        if trialfilter != ():
            trialfilter = trialfilter[0]
            h_vs_b_by_trial = [self.h_vs_b_by_trial[i] for i in trialfilter]
        else:
            h_vs_b_by_trial = self.h_vs_b_by_trial
            
        h_vs_b_mod = [np.array(hb_trial) if hb_trial[0] > 0 else -1*np.array(
            hb_trial) for hb_trial in h_vs_b_by_trial]
        
#        h_vs_b_mod = [np.abs(hb_trial) for hb_trial in self.h_vs_b_by_trial]
        if plot_fish:
 #           fig, ax = pl.subplots(1, 1)
#            tsplot(h_vs_b_mod, ax)
  #          pl.show()
            pass
        else:
            return h_vs_b_mod

    def load_experiment(self):
        self.get_xy_coords()
        self.load_barrier_info()
        self.get_correct_barrier()
        self.get_stim_times(False)

    def exporter(self):
        with open(self.directory + '/escapes_' +
                  self.condition + '.pkl', 'wb') as file:
            pickle.dump(self, file)

    def load_barrier_info(self):
        barrier_file = self.barrier_file
        xylist = []
        diam_list = []
        for line, j in enumerate(barrier_file[2:]):
            if line % 2 == 0:
                xylist.append(x_and_y_coord(j))
            else:
                diam_list.append(float(j))
        self.barrier_coordinates = xylist
        self.barrier_diam = np.round(np.mean(diam_list))

# this function takes in the xy coordinates for each tap
# trial, throwing out outliers and replacing them with
# coords from the frame previous. outliers are kept track of
# and used to ignore video frames from missed indices.
        
    def get_xy_coords(self):
        for filenum, xy_file in enumerate(self.xy_escapes):
            xcoords = []
            ycoords = []
            for coordstring in xy_file:
                x, y = x_and_y_coord(coordstring)
                xcoords.append(x)
                ycoords.append(y)
            xc_trial, yc_trial, missed_inds = outlier_filter(xcoords,
                                                             ycoords, [])
            self.xy_coords_by_trial.append([xc_trial,
                                            yc_trial])
            self.missed_inds_by_trial.append(missed_inds)



# There is a 100 pixel wide window surrounding the LED.
# At first, the LED is in the leftmost 50 pixels. After stim, in rightmost.
# Have 2000 frames of 100 pixel windows. Get exact timing
# using when the LED reaches steady state.
# The stimulus itself is 200ms long, and lasts for ~100 frames.
# The steady state should be taken around 50 frames into the stimulus.

    def get_stim_times(self, plot_stim):
        for stim_file in self.stim_data:
            stimdata = np.genfromtxt(stim_file)
            light_profile = [np.sum(a[50:100]) for a in partition(100, stimdata)]
            stim_init = np.argmax(light_profile[0:140])
            if not (self.timerange[0] < stim_init < self.timerange[1]):
                stim_init = np.nan
            elif light_profile[stim_init] < 200:
                stim_init = np.nan
            if plot_stim:
                pl.plot(light_profile)
                if not math.isnan(stim_init):
                    pl.plot([stim_init], [0], marker='.', color='r')
                pl.show()
            if math.isnan(stim_init):
                self.stim_init_times.append(self.pre_c)
                self.stim_times_accurate.append(0)
            else:
                self.stim_init_times.append(stim_init-self.timerange[0])
                self.stim_times_accurate.append(1)

# finds which barrier in the barrier file the fish is escaping from
# using vector distance.

    def get_correct_barrier(self):
        for coords in self.xy_coords_by_trial:
            if coords[0] == []:
                self.barrier_xy_by_trial.append([])
                continue
            mag_distance = []
            init_x = coords[0][self.timerange[0]]
            init_y = coords[1][self.timerange[0]]
            for barr_xy in self.barrier_coordinates:
                temp_distance = np.sqrt(
                    (barr_xy[0]-init_x)**2 + (barr_xy[1] - init_y)**2)
                mag_distance.append(temp_distance)
            correct_barrier_index = np.argmin(mag_distance)
            self.barrier_xy_by_trial.append(self.barrier_coordinates[
                    correct_barrier_index])

# this function asks if there are any coordinates that are within 2 pixels of
# the barrier between 10ms after the escape and the end of the escape. if there
# are 2 (allowing 1 for noise) its a collision

    def collision_course(self, *plot):
        
        self.collisions = []
        self.collision_times = []
            
        for trialnum, xyc in enumerate(self.xy_coords_by_trial):
            if xyc[0] == []:
                self.collisions.append(np.nan)
                self.collision_times.append([])
                continue
            barrier_x = self.barrier_xy_by_trial[trialnum][0]
            barrier_y = self.barrier_xy_by_trial[trialnum][1]
            barrier_diameter = self.barrier_diam
            xcoords = self.xy_coords_by_trial[trialnum][0]
            ycoords = self.xy_coords_by_trial[trialnum][1]
            collision_coords = 0
            # use the frame after the completion of the cstart as the start of the trajectory.
            # if no cstart detected, use frame after avg. cstart time (8ms = 4 frames)
            if not math.isnan(self.escape_latencies[trialnum]):
                t_offset = self.stim_init_times[trialnum] + self.escape_latencies[trialnum] + 1
            else:
                t_offset = self.stim_init_times[trialnum] + 5
            t_init = int(self.timerange[0] + t_offset)
            collision_times = []
            t_end = 0
            if self.collision_bound + t_init > self.timerange[1]:
                t_end = self.timerange[1]
            else:
                t_end = t_init + self.collision_bound
            for c_time, (fish_x, fish_y) in enumerate(zip(
                    xcoords[t_init:t_end],
                    ycoords[t_init:t_end])):
                if collision(barrier_x, barrier_y, self.barrier_diam,
                             fish_x, fish_y):
                    collision_coords += 1
                    collision_times.append(c_time+t_offset)
            # prevents a single noise frame from inducing a collision
            if collision_coords > 1:
                self.collisions.append(1)
                self.collision_times.append(collision_times)
            else:
                self.collisions.append(0)
                self.collision_times.append([])
            if plot != ():
                fig = pl.figure()
                axes = fig.add_subplot(111)
                barrier_plot = pl.Circle((barrier_x, barrier_y),
                                         barrier_diameter / 2, fc='r')
                axes.add_artist(barrier_plot)
                axes.grid(False)
                colorline(
                    np.array(xcoords[self.timerange[0]:self.timerange[1]]),
                    np.array(ycoords[self.timerange[0]:self.timerange[1]]))
                axes.set_xlim(barrier_x - 200, barrier_x + 200)
                axes.set_ylim(barrier_y - 200, barrier_y + 200)
                axes.set_aspect('equal')
                colo = ScalarMappable(cmap='afmhot')
                colo.set_array(
                    np.arange(
                        float(self.timerange[0]) / 500, float(self.timerange[1]) / 500,
                        .1))
                pl.colorbar(colo)
                pl.title('Trial' + str(trialnum))
                pl.show()

# for each tap trial, a raw video of the escape is recorded and a background
# saved to use for contour finding. this function calls the contourfinding
# function and finds the orientation of the found contour by comparing its
# center of mass to its contours average position. this yields an
# extremely accurate vector for orientation which is filtered and
# converted to an angle.

    def find_correct_background(self, curr_trial):
        trial_list = np.array([b[1] for b in self.backgrounds])
        min_value = np.min(np.abs(trial_list-curr_trial))
        closest_inds = np.where(np.abs(trial_list-curr_trial) == min_value)[0]
        candidate_backgrounds = self.backgrounds[closest_inds[0]:closest_inds[-1]+1]
        br_conditions = [cb[2] for cb in candidate_backgrounds]
        if self.condition in br_conditions:
            return candidate_backgrounds[br_conditions.index(self.condition)][0]
        else:
            return candidate_backgrounds[0][0]
                                
    def get_orientation(self, makevid):
        for trial, (vid_file, xy) in enumerate(
                zip(self.movie_id, self.xy_coords_by_trial)):
            if xy[0] == []:
                self.ha_in_timeframe.append([])
                self.contours_by_trial.append([])
                continue
            heading_vec_list = []
            contour_list = []
            fps = 500
            vid = imageio.get_reader(vid_file, 'ffmpeg')
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            dr = self.directory
            if makevid:
                ha_vid = cv2.VideoWriter(
                    dr + '/ha' + str(trial) + self.condition + '.AVI',
                    fourcc, fps, (80, 80), True)
                thresh_vid = cv2.VideoWriter(
                    dr + '/thresh' + str(trial) + self.condition + '.AVI',
                    fourcc, fps, (80, 80), True)
            xcoords = xy[0][self.timerange[0]:self.timerange[1]]
            ycoords = xy[1][self.timerange[0]:self.timerange[1]]
            missed_inds_trial = self.missed_inds_by_trial[trial]
    # the arrays are indexed in reverse from how you'd like to plot.
            for frame, (x, y) in enumerate(zip(xcoords, ycoords)):
                y = 1024 - y
                im_color = vid.get_data(self.timerange[0] + frame)
                im = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY)
                matched_background = self.find_correct_background(trial)
                background_roi = slice_background(
                    matched_background, x, y)
                brsub = cv2.absdiff(im, background_roi).astype(np.uint8)
                fishcont, mid_x, mid_y, th = self.contourfinder(brsub, 30)
                contour_list.append(fishcont)
                th = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)
                if frame in missed_inds_trial:
                    heading_vec_list.append([np.nan, np.nan])
                    if makevid:
                        ha_vid.write(np.zeros(
                            [im_color.shape[0],
                             im_color.shape[1],
                             im_color.shape[2]]).astype(np.uint8))
                        thresh_vid.write(np.zeros(
                            [th.shape[0],
                             th.shape[1]]).astype(np.uint8))
                    continue
                if math.isnan(mid_x):
                    heading_vec_list.append([mid_x, mid_y])
                    if makevid:
                        ha_vid.write(im_color)
                        thresh_vid.write(th)
                    continue
                fish_xy_moments = cv2.moments(fishcont)
                fish_com_x = int(fish_xy_moments['m10']/fish_xy_moments['m00'])
                fish_com_y = int(fish_xy_moments['m01']/fish_xy_moments['m00'])
                fishcont_middle = np.mean(fishcont, axis=0)[0].astype(np.int)
                fish_middle_x = fishcont_middle[0]
                fish_middle_y = fishcont_middle[1]
                cv2.circle(im_color, (fish_com_x, fish_com_y), 1, (255, 0, 0), 1)
    # com actually comes out red, mid blue
                cv2.circle(im_color,
                           (fish_middle_x, fish_middle_y), 1, (0, 0, 255), 1)
                cv2.drawContours(im_color, [fishcont], -1, (0, 255, 0), 1)
                vec_heading = np.array(
                    [fish_com_x, fish_com_y]) - np.array([fish_middle_x,
                                                          fish_middle_y])
                heading_vec_list.append(vec_heading)
                if makevid:
                    ha_vid.write(im_color)
                    thresh_vid.write(th)
   
            vid.close()
            if makevid:
                ha_vid.release()
                thresh_vid.release()

            filt_heading_vec_list = filter_uvec(heading_vec_list, 1)
            heading_angles = [np.arctan2(vec[1], vec[0])
                              if not math.isnan(vec[0])
                              else float('nan') for vec
                              in filt_heading_vec_list]

# yields coords with some negatives in a reverse unit circle (i.e clockwise)
# have to normalize to put in unit circle coords.

            norm_orientation = [-ang if ang < 0 else 2 * np.pi - ang
                                for ang in heading_angles]
            self.ha_in_timeframe.append(norm_orientation)
            self.contours_by_trial.append(contour_list)

# this function calculates a vector from the fish's position to the
# barrier it is escaping

    def vec_to_barrier(self):
        for trial, xy in enumerate(self.xy_coords_by_trial):
            if xy[0] == []:
                self.ba_in_timeframe.append([])
                continue
            vecs_to_barrier = []
            x = xy[0][self.timerange[0]:self.timerange[1]]
            y = xy[1][self.timerange[0]:self.timerange[1]]
            barr_xy = np.array(self.barrier_xy_by_trial[trial])
            for fish_x, fish_y in zip(x, y):
                fish_xy = np.array([fish_x, fish_y])
                b_vec = barr_xy - fish_xy
                vecs_to_barrier.append(b_vec)
            angles = [np.arctan2(vec[1], vec[0])
                      if not math.isnan(vec[0]) else float('nan')
                      for vec in vecs_to_barrier]
            transformed_angles = [2*np.pi + ang if ang < 0 else ang
                                  for ang in angles]
            self.ba_in_timeframe.append(transformed_angles)

# this function compares the heading angle of the fish to the
# vector to the barrier over time.

    def heading_v_barrier(self):
        for h_angles, b_angles in zip(self.ha_in_timeframe,
                                      self.ba_in_timeframe):
            if h_angles == []:
                self.h_vs_b_by_trial.append([])
                continue
            diffs = []
            right_or_left = []
            for ha, ba in zip(h_angles, b_angles):
                if ha > ba:
                    diff = ha - ba
                    if diff > np.pi:
                        diff = 2 * np.pi - ha + ba
                        right_or_left.append('l')
                    else:
                        right_or_left.append('r')
                elif ba > ha:
                    diff = ba - ha
                    if diff > np.pi:
                        diff = 2 * np.pi - ba + ha
                        right_or_left.append('r')
                    else:
                        right_or_left.append('l')
                elif math.isnan(ha) or math.isnan(ba):
                    diff = float('nan')
                    right_or_left.append('nan')
                else:
                    diff = 0
                    right_or_left.append('n')
                diffs.append(diff)
            diffs = [-df if dirc == 'l' else df
                     for df, dirc in zip(diffs, right_or_left)]
            self.h_vs_b_by_trial.append(diffs)

# Think of barrier axis as the X axis of the unit circle.
# ha to the right of the barrier are negative
# (barrier is left of the fish), while ha to the left of
# the barrier are positive (barrier is on the right of the fish).

# this function finds the state of the fish in the moment
# before the tap stimulus arrives. these conditions are used
# for finding heading to the barrier and the cstart

    def find_initial_conditions(self):
        for trial, xyc in enumerate(self.xy_coords_by_trial):
            if xyc[0] == []:
                self.initial_conditions.append([])
                continue
            ha_avg = np.nanmean(self.ha_in_timeframe[trial][0:self.pre_c])
            ba_avg = np.nanmean(self.ba_in_timeframe[trial][0:self.pre_c])
            h_to_b = np.nanmean(self.h_vs_b_by_trial[trial][0:self.pre_c])
            barr_xy = np.array(self.barrier_xy_by_trial[trial])
            xy = self.xy_coords_by_trial[trial]
            fish_x = np.nanmean(
                xy[0][self.timerange[0]:self.timerange[0]+self.pre_c])
            fish_y = np.nanmean(
                xy[1][self.timerange[0]:self.timerange[0]+self.pre_c])
            fish_xy = np.array([int(fish_x), int(fish_y)])
            vec = barr_xy - fish_xy
            mag = magvector(vec) - (self.barrier_diam / 2)
            self.initial_conditions.append([h_to_b, mag, ha_avg,
                                            ba_avg, fish_xy])

    def trial_analyzer(self, plotc):
        self.get_orientation(True)
        self.vec_to_barrier()
        self.heading_v_barrier()
        self.find_initial_conditions()
        self.body_curl()
        self.find_cstart(False)
        self.collision_course()

# this function splits up escapes by whether the barrier was to the
# right or left of the fish at the onset of the tap.
# you can chose to plot these trajectories for a single fish or
# wrap this function and plot many fish.

    def escapes_vs_barrierloc(self, plotit, *trialfilter):
        if plotit:
            turn_fig = pl.figure()
            turn_ax = turn_fig.add_subplot(111)
            turn_ax.set_xlim([-100, 100])
            turn_ax.set_ylim([-100, 100])
        timerange = self.timerange
        barrier_on_left = []
        barrier_on_right = []
        barrier_on_left_trials = []
        barrier_on_right_trials = []
        if trialfilter != ():
            trialfilter = trialfilter[0]
        else:
            trialfilter = range(len(self.xy_coords_by_trial))
        for trial, xy_coords in enumerate(self.xy_coords_by_trial):
            if trial not in trialfilter:
                continue
            if xy_coords == []:
                continue
            to_barrier_init = self.initial_conditions[trial][0]
            ha_init = self.initial_conditions[trial][2]
            if not math.isnan(ha_init):
                zipped_coords = zip(xy_coords[0], xy_coords[1])
                escape_coords = rotate_coords(zipped_coords, -ha_init)
                x_escape = np.array(
                    [x for [x, y] in escape_coords[timerange[0]:timerange[1]]])
                x_escape = x_escape - x_escape[0]
                y_escape = np.array(
                    [y for [x, y] in escape_coords[timerange[0]:timerange[1]]])
                y_escape = y_escape - y_escape[0]
                if to_barrier_init < 0:
                    barrier_on_left.append([x_escape, y_escape])
                    barrier_on_left_trials.append(trial)
                    if plotit:
                        turn_ax.plot(
                            x_escape,
                            y_escape, 'b')
                        turn_ax.text(
                            x_escape[-1],
                            y_escape[-1],
                            str(trial),
                            size=10,
                            backgroundcolor='w')
                elif to_barrier_init > 0:
                    barrier_on_right.append([x_escape, y_escape])
                    barrier_on_right_trials.append(trial)
                    if plotit:
                        turn_ax.plot(
                            x_escape,
                            y_escape, 'm')
                        turn_ax.text(
                            x_escape[-1],
                            y_escape[-1],
                            str(trial),
                            size=10,
                            backgroundcolor='w')
        if plotit:
            pl.show()
        else:
            return barrier_on_left, barrier_on_right, barrier_on_left_trials, barrier_on_right_trials

# this function uses the thresholded image where contours were discovered
# and uses the contour found in get_orientation. the contour is rotated
# according to the fish's heading angle.
# points along the contour are used to find the cumulative angle of the
# tail and the index where the c-start begins.
        
    def find_cstart(self, plotornot):
        for trial, xyc in enumerate(self.xy_coords_by_trial):
            if xyc[0] == []: # or not self.stim_times_accurate[trial]:
                self.cstart_rel_to_barrier.append(np.nan)
                self.cstart_rel_to_last_bout.append(np.nan)
                self.cstart_angles.append(np.nan)
                self.escape_latencies.append(np.nan)
                self.pre_escape_bouts.append(np.nan)
                continue
            tail_kern = Gaussian1DKernel(self.cstart_filter_std)
            stim_init = self.stim_init_times[trial]
            print("Stim Init")
            pre_xy_file = self.pre_escape[trial]
            try:
                pre_xy_file[0]
            except IndexError:
                pre_xy_file = [pre_xy_file.tolist()]
            pre_xcoords = []
            pre_ycoords = []
            for coordstring in pre_xy_file[-1000:]:
                x, y = x_and_y_coord(coordstring)
                pre_xcoords.append(x)
                pre_ycoords.append(y)
            pre_xcoords += self.xy_coords_by_trial[
                trial][0][0:stim_init+self.timerange[0]:2]
            pre_ycoords += self.xy_coords_by_trial[
                trial][1][0:stim_init+self.timerange[0]:2]
            vel_vector = [np.sqrt(
               np.dot(
                   [v2[0]-v1[0], v2[1]-v1[1]],
                   [v2[0]-v1[0], v2[1]-v1[1]])) for v1, v2 in sliding_window(
                       2, zip(pre_xcoords, pre_ycoords))]
            vv_filtered = gaussian_filter(vel_vector, 10)
            bout_inds = argrelextrema(
                np.array(vv_filtered), np.greater_equal,  order=5)[0]
            if len(bout_inds) != 0:
                bi_thresh = [arg for arg in bout_inds if vv_filtered[arg] > .5]
                if len(bi_thresh) != 0:
                    fi = [bi_thresh[0]]
                else:
                    fi = []
                bi_norepeats = fi + [b for a, b in sliding_window(
                    2, bi_thresh) if b-a > 20]
                # pl.plot(vv_filtered)
                # pl.plot(bi_norepeats,
                #         np.zeros(len(bi_norepeats)), marker='.')
                # pl.show()

# THESE SEEM KIND OF ARBITRARY RIGHT NOW. BUT TRY IT.
            bout_init_position = [[np.nanmean(pre_xcoords[bi-40:bi-30]),
                                   np.nanmean(pre_ycoords[bi-40:bi-30])]
                                  for bi in bi_norepeats]
            bout_post_position = [[np.nanmean(pre_xcoords[bi+10:bi+40]),
                                   np.nanmean(pre_ycoords[bi+10:bi+40])]
                                  for bi in bi_norepeats]
            disp_vecs = [[b[0]-a[0], b[1]-a[1]] for a,
                         b in zip(bout_init_position, bout_post_position)]
            dots = [np.dot(i, j) / (magvector(i) * magvector(j))
                    for i, j in sliding_window(2, disp_vecs)]
            ang = [np.arccos(a) for a in dots]
            crosses = [np.cross(i, j) for i, j in sliding_window(2, disp_vecs)]
            bout_angles = []
            for a, c in zip(ang, crosses):
                if c < 0:
                    bout_angles.append(a)
                else:
                    bout_angles.append(-1*a)
            if bout_angles:
                print(bout_angles[-1])
            self.pre_escape_bouts.append(bout_angles)
            c_thresh = self.cstart_angle_thresh
  #          cstart_window = [0, stim_init + 20]
 #           ta = convolve(self.tailangle_sums[trial][0:cstart_window[1]], tail_kern)
            ta = convolve(self.tailangle_sums[trial], tail_kern)
            # avg_curl_init = np.nanmean(ta[0:self.pre_c])
            # ta = ta - avg_curl_init

            # currently the "order" arg to argrelmin and max is only 1, meaning
            # the relmin and max happen if the left and right of the index are both smaller or both bigger. also, the gaussian
            # filter may be a bit too strong -- may be cutting off the max tail angle so people will be less convinced its a cstart.
            # could switch to .2 or .5
            c_start_angle = float('nan')
            ta_min = argrelmin(ta)[0].tolist()
            ta_max = argrelmax(ta)[0].tolist()
            ta_maxandmin = [x for x in sorted(ta_min + ta_max) if (
                (stim_init < x < self.cstart_time_thresh+stim_init) and abs(
                    ta[x]) > c_thresh)]
            if plotornot:
                pl.plot(ta)
                pl.plot(self.tailangle_sums[trial])
                if len(ta_maxandmin) != 0:
                    pl.plot([ta_maxandmin[0]], [0], marker='.', color='r')
                pl.title('trial' + str(trial))
                pl.show()
            if not ta_maxandmin:
                print('nans to cstart varbs')
                self.cstart_rel_to_last_bout.append(np.nan)
                self.cstart_rel_to_barrier.append(np.nan)
                self.cstart_angles.append(np.nan)
                self.escape_latencies.append(np.nan)
                self.cstarts_per_trial.append(0)
                continue
            c_start_angle = ta[ta_maxandmin[0]]
            c_start_ind = ta_maxandmin[0]
            self.cstart_angles.append(c_start_angle)
            self.cstarts_per_trial.append(1)
            if self.stim_times_accurate[trial]:
                self.escape_latencies.append(c_start_ind - stim_init)
            else:
                self.escape_latencies.append(np.nan)
    # Get latency here based on stim index and c_start_index


    # switch this threshold to ask questions about specific intial conditions before cstart
            if not math.isnan(c_start_angle):
#                    np.abs(self.initial_conditions[trial][0]) > .2):
                if np.sum(
                        np.sign(
                            [c_start_angle,
                             self.initial_conditions[trial][0]])) == 0:
                    print('away')
                    self.cstart_rel_to_barrier.append(1)
                else:
                    print('towards')
                    self.cstart_rel_to_barrier.append(0)

                if bout_angles:
                    if np.sum(np.sign(
                            [c_start_angle,
                             bout_angles[-1]])) == 0:
                        self.cstart_rel_to_last_bout.append(1)
                    else:
                        self.cstart_rel_to_last_bout.append(0)
                else:
                    self.cstart_rel_to_last_bout.append(np.nan)

            else:
                self.cstart_rel_to_barrier.append(np.nan)
                self.cstart_rel_to_last_bout.append(np.nan)

# contourfinder dynamically thresholds a background subtracted image
# until a criteria of contour size and proximity to the fish is reached for a
# contour. the function is recursive in that it continues to lower the
# threshold until the correct contour is found.

    def contourfinder(self, im, threshval):
        
        def cont_distance(cont_list, xy_center):
            for cont in cont_list:
                rect = cv2.minAreaRect(cont)
                box = cv2.boxPoints(rect)
                xcont, ycont = np.mean(box, axis=0).astype(np.int)
                vec_dist = [xy_center[0] - xcont, xy_center[1] - ycont]
                if magvector(vec_dist) < 10:
                    return cont, xcont, ycont
            return [], np.nan, np.nan
        # good params at 120 low area and dilate at 3x3
        # try dilate 5x5. works ok with 250.
        xy_cent = [40, 40]
        r, th = cv2.threshold(im, threshval, 255, cv2.THRESH_BINARY)
        rim, contours, hierarchy = cv2.findContours(
            th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        areamin = self.area_thresh
        areamax = 100
        contcomp = [cnt for cnt in
                    contours if areamin < cv2.contourArea(
                        cnt) < areamax]
        if contcomp:
            fishcont, x, y = cont_distance(contcomp, xy_cent)
            if math.isnan(x):
                return self.contourfinder(im, threshval-1)
            else:
                return fishcont, x, y, th
    # this is a catch for missing the fish
    # original was 47 for areamin, 15 for threshval. 
    #    if threshval < 15:
    
        if threshval <= 10:
            if not contours:
                return np.array([]), float('NaN'), float('NaN'), np.zeros(
                    [im.shape[0], im.shape[1]]).astype(np.uint8)
            if areamin * .75 < cv2.contourArea(contours[0]) < areamax:
                fishcont, x, y = cont_distance(contours[0], xy_cent)
                if math.isnan(x):
                    return np.array([]), float('NaN'), float('NaN'), np.zeros(
                        [im.shape[0], im.shape[1]]).astype(np.uint8)
                else:
                    return contours[0], x, y, th
            else:
#                print('thresh too low')
                return np.array([]), float('NaN'), float('NaN'), np.zeros(
                    [im.shape[0], im.shape[1]]).astype(np.uint8)
        else:
            return self.contourfinder(im, threshval-1)

    # body curl finds the c-start angle using the previously described
    # fish contour 

    def body_curl(self):

        def body_points(seg1, seg2, endpoint):
            right = [unpack[0] for unpack in seg1]
            left = [unpack[0] for unpack in seg2]
            bp = [[int(np.mean([a[0], b[0]])), int(np.mean([a[1], b[1]]))]
                  for a, b in zip(right, left[::-1])]
            bp.append(endpoint[0])
            return bp

        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        fps = 500
        for trial, xyc in enumerate(self.xy_coords_by_trial):
            if xyc[0] == []:
                self.tailangle_sums.append([])
                self.all_tailangles.append([])
                continue
            cstart_vid = cv2.VideoWriter(
                self.directory + '/cstart' + str(
                    trial) + self.condition + '.AVI',
                fourcc, fps, (80, 80), True)
            threshvid = imageio.get_reader(self.directory + '/thresh' + str(
                trial)+self.condition+'.AVI', 'ffmpeg')
            all_angles = []
            ha_adjusted = deque([np.mod(90-(np.degrees(angle)), 360)
                                 for angle in self.ha_in_timeframe[trial]])
            cnt = 0
            # make this a variable -- 25 should be added to timeframe multiple times. 
            for frame in range(self.timerange[1] - self.timerange[0]):
                ha_adj = ha_adjusted.popleft()
                try:
                    im_color = threshvid.get_data(frame)
                except:
                    print("Got no Vid")
                im = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY)
                im_rot, m, c = rotate_image(im, ha_adj)
                im_rot_color = cv2.cvtColor(im_rot, cv2.COLOR_GRAY2RGB)
                body_unrotated = self.contours_by_trial[trial][frame]
                body = rotate_contour(im, ha_adj, body_unrotated)
                if body.shape[0] == 0:
                    all_angles.append([])
                    cstart_vid.write(im_rot_color)
                    continue
                body_perimeter = cv2.arcLength(body, True)
                highest_pt = np.argmin([bp[0][1] for bp in body])
                # rearrange body points so top y value is on top.
                body = np.concatenate([body[highest_pt:], body[0:highest_pt]])
                body_segment1 = []
                body_segment2 = []
                segment = 1.0
                numsegs = 14.0
                # Bool to arclength is whether it is a closed contour.
                # this must not act to restrict, but to loop.

                # there are more points in body than numsegs. you are splitting the contour up
                # into numsegs equal parts. 
                for i in range(len(body)):
                    if cv2.arcLength(body[0:i+1],
                                     False) > body_perimeter*(segment/numsegs):
                        if segment < (numsegs/2):
                            body_segment1.append(body[i])
                        elif segment > (numsegs/2):
                            body_segment2.append(body[i])
                        elif segment == (numsegs/2):
                            endpoint = body[i]
                        segment += 1

                avg_body_points = body_points(body_segment1,
                                              body_segment2, endpoint)[1:]

# First point inside head is unreliable. take 1:. 
                for bp in avg_body_points:
                    cv2.ellipse(im_rot_color,
                                (bp[0], bp[1]),
                                (1, 1), 0, 0, 360, (255, 0, 255), -1)
                cv2.drawContours(im_rot_color, [body], -1, (0, 255, 0), 1)
                cstart_vid.write(im_rot_color)
                body_gen = toolz.itertoolz.sliding_window(2, avg_body_points)
                body_point_diffs = [(b[0]-a[0],
                                     b[1]-a[1]) for a, b in body_gen]
    #            print body_point_diffs
                angles = []
                # Angles are correct given body_points. 
                for vec1, vec2 in toolz.itertoolz.sliding_window(
                        2, body_point_diffs):
                    dp = np.dot(vec1, vec2)
                    mag1 = np.sqrt(np.dot(vec1, vec1))
                    mag2 = np.sqrt(np.dot(vec2, vec2))
                    try:
                        ang = np.arccos(dp / (mag1*mag2))
                    except FloatingPointError:
                        angles.append(np.nan)
                        continue
                    if np.cross(vec1, vec2) > 0:
                        ang *= -1
                    angles.append(np.degrees(ang))
                cnt += 1
                all_angles.append(angles)
            cstart_vid.release()
            threshvid.close()
            self.tailangle_sums.append([np.nansum(i) for i in all_angles])
            self.all_tailangles.append(all_angles)


def find_darkest_pixel(im):
    boxfiltered_im = cv2.boxFilter(im, 0, (3, 3))
    boxfiltered_im = cv2.boxFilter(boxfiltered_im, 0, (3, 3))
    max_y, max_x = np.unravel_index(boxfiltered_im.argmax(),
                                    boxfiltered_im.shape)
    return max_x, max_y


def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

# Interface to LineCollection:


def colorline(x,
              y,
              z=None,
              cmap=pl.get_cmap('afmhot'),
              norm=pl.Normalize(0.0, 1.0),
              linewidth=3,
              alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
    # Special case if a single number:
    if not hasattr(
            z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = LineCollection(
        segments,
        array=z,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        alpha=alpha)
    ax = pl.gca()
    ax.add_collection(lc)
    return lc


def clear_frame(ax=None):
    # Taken from a post by Tony S Yu
    if ax is None:
        ax = pl.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.itervalues():
        spine.set_visible(False)

#this forces the coordinate system to be facing upward at 0 rad.


def rotate_coords(coords, angle):
    angle = angle + np.pi / 2
    center = [640, 512]
    rotated_coords = []
    for x, y in coords:
        xcoord = x - center[0]
        ycoord = y - center[1]
        xcoord_rotated = (xcoord * np.cos(angle) - ycoord * np.sin(angle)
                          ) + center[0]
        ycoord_rotated = (xcoord * np.sin(angle) + ycoord * np.cos(angle)
                          ) + center[1]
        if not math.isnan(xcoord_rotated) and not math.isnan(ycoord_rotated):
            rotated_coords.append([int(xcoord_rotated), int(ycoord_rotated)])
#    else:
#  rotated_coords.append([float('NaN'), float('NaN')])
    return rotated_coords


def outlier_filter(xcoords, ycoords, missed_inds):
    new_x = [xcoords[0]]
    new_y = [ycoords[0]]
    for i, crds in enumerate(zip(xcoords[1:], ycoords[1:])):
        diff_vec = [crds[0] - new_x[-1], crds[1] - new_y[-1]]
        vmag = magvector(diff_vec)
        if i == len(xcoords) - 1:
            return new_x, new_y, missed_inds
        elif vmag < 100:
            new_x.append(crds[0])
            new_y.append(crds[1])
        else:
            missed_inds.append(i+1)
            new_x.append(new_x[-1])
            new_y.append(new_y[-1])
            xcoords = new_x + xcoords[i+2:]
            ycoords = new_y + ycoords[i+2:]
            try:
                return outlier_filter(xcoords, ycoords, missed_inds)
            except RuntimeError:
                return [], [], []
    return new_x, new_y, missed_inds


def filter_list(templist):
    filtlist = scipy.ndimage.filters.gaussian_filter(templist, 2)
    return filtlist


def filter_uvec(vecs, sd):
    gkern = Gaussian1DKernel(sd)
    npvecs = np.array(vecs)
    filt_vecs = np.copy(npvecs)
    for i in range(npvecs[0].shape[0]):
        filt_vecs[:, i] = convolve(npvecs[:, i], gkern)
    return filt_vecs


def slice_background(br, xcrd, ycrd):
    br_roi = np.array(
        br[int(ycrd)-40:int(ycrd)+40,
           int(xcrd)-40:int(xcrd)+40]).astype(np.uint8)
    return br_roi


def x_and_y_coord(coord):
    xcoord = ''
    ycoord = ''
    x_incomplete = True
    y_incomplete = True
    for char in coord:
        if char == ',':
            x_incomplete = False
            continue
        if char == '}':
            y_incomplete = False
        if x_incomplete and char != '{' and char != 'X' and char != '=':
            xcoord += char
        if not x_incomplete and y_incomplete and char != 'Y' and char != '=':
            ycoord += char
    return float(xcoord), 1024 - float(ycoord)


def rotate_image(image, angle, *contour):
    image_center = tuple(np.array(image.shape) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale=1.0)
    if contour != ():
        result = cv2.warpAffine(
                contour[0], rot_mat, contour[0].shape, flags=cv2.INTER_LINEAR)
    else:
        result = cv2.warpAffine(
                image, rot_mat, image.shape, flags=cv2.INTER_LINEAR)
    return result, rot_mat, image_center


def rotate_contour(image, angle, contour):
    rotated_contour = []
    image_center = tuple(np.array(image.shape) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale=1.0)
    A = rot_mat[0:2, 0:2]
    B = rot_mat[:, 2].reshape(-1, 1)
    for contpoint in contour:
        new_point = np.matmul(A, contpoint.reshape(-1, 1)) + B
        rotated_contour.append([[new_point[0][0], new_point[1][0]]])
    return np.array(rotated_contour).astype(np.int32)


def magvector(vec):
    mag = np.sqrt(np.dot(vec, vec))
    return mag


def collect_collision_stat(fishlist, cond, height_multiplier, filt):
    ec_list = [experiment_collector(fish, cond, filt) for fish in fishlist]
    condition_arrays = [ec[0].convert_to_nparrays() for ec in ec_list]
    collision_stat = [[hm*float(c_array['Total Collisions']) / c_array['Total Correct Trajectories'][1]] for hm, c_array in zip(list(height_multiplier), condition_arrays)]
    plot_varb_over_ecs([collision_stat, lambda x: x])
    return collision_stat
    

def get_n_trajectories(fish_experiment_list):
    all_n_x = []
    all_n_y = []
    for drct_list in fish_experiment_list:
        ec = experiment_collector(drct_list, ['n'], [0, [], 1])
        for fish in ec:
            all_traj = fish.escape_data[
                "Barrier On Left Trajectories"] + fish.escape_data[
                    "Barrier On Right Trajectories"]
            all_n_x += [tx[0] for tx in all_traj]
            all_n_y += [ty[1] for ty in all_traj]
    return [all_n_x, all_n_y]
        

def plot_collision_stat(all_collisions, num_n_trials):
    fig, ax = pl.subplots(1, 1, figsize=(6, 3))
    collision_stats_to_percentage = [list(map(
        lambda x: np.sum(x) / (num_n_trials * len(x)), cp)) for cp in all_collisions]
    xvals = []
    for i, d in enumerate(collision_stats_to_percentage):
        xvals.append(i*np.ones(len(d)))
        xv_concat = np.concatenate(xvals)
        yvals = np.concatenate(collision_stats_to_percentage)
    sb.pointplot(x=xv_concat, y=yvals, color='k')
    pl.show()
    return collision_stats_to_percentage
    

def infer_collisions(fish_experiment_list, plotornot):
    all_n_trajectories = get_n_trajectories(fish_experiment_list)
    x_escape_n, y_escape_n = all_n_trajectories
    print(len(x_escape_n))
    collisions_across_all_conditions = []
    for drct_list in fish_experiment_list:
        collisions_given_condition = []
        for drcty in drct_list:
            drct = '/Volumes/Esc_and_2P/Escape_Results/' + drcty
            collisions_per_fish = []
            try:
                esc_obj = pickle.load(open(
                    drct + '/escapes_l.pkl', 'rb'))
            except IOError:
                print(drct)
                print("PICKLING ERROR")
                continue

            for trial, xy in enumerate(esc_obj.xy_coords_by_trial):
                angle, mag = esc_obj.initial_conditions[trial][0:2]
                mag += (esc_obj.barrier_diam / 2)
                if 0 <= angle < np.pi / 2:
                    barrier_x = np.sin(angle) * mag
                    barrier_y = np.cos(angle) * mag
                elif angle >= np.pi / 2:
                    barrier_x = np.sin(np.pi - angle) * mag
                    barrier_y = -np.cos(np.pi - angle) * mag
                elif 0 >= angle > -np.pi / 2:
                    barrier_x = -np.sin(-angle) * mag
                    barrier_y = np.cos(-angle) * mag
                elif angle <= -np.pi / 2:
                    barrier_x = -np.sin(np.pi + angle) * mag
                    barrier_y = -np.cos(np.pi + angle) * mag
                if plotornot:
                    turn_fig = pl.figure()
                    turn_ax = turn_fig.add_subplot(111)
                    turn_ax.set_xlim([-200, 200])
                    turn_ax.set_ylim([-200, 200])
                    turn_ax.set_aspect('equal')
                    barr = pl.Circle((barrier_x, barrier_y), esc_obj.barrier_diam / 2,
                                          fc='r')
                    turn_ax.add_artist(barr)
                collisions = 0
                # conditions here are identical to collision detector
                t_init = esc_obj.pre_c + 5
                t_end = t_init + esc_obj.collision_bound
                for x_esc, y_esc in zip(x_escape_n,
                                        y_escape_n):
                    collide = [collision(barrier_x,
                                         barrier_y,
                                         esc_obj.barrier_diam, x, y)
                               for x, y in zip(x_esc[t_init:t_end],
                                               y_esc[t_init:t_end])]
                    if plotornot:
                        turn_ax.plot(x_esc, y_esc)
                    if sum(collide) > 1:
                        collisions += 1
                collisions_per_fish.append(collisions)
            collisions_given_condition.append(collisions_per_fish)
        collisions_across_all_conditions.append(collisions_given_condition)
    return collisions_across_all_conditions, len(x_escape_n)
                        


def collect_varb_across_ec(fishlist, cond, varb, filt):
    if type(cond) == list:
        ec_list = [experiment_collector(fish, c, filt) for fish, c in zip(fishlist, cond)]
    else:
        ec_list = [experiment_collector(fish, cond, filt) for fish in fishlist]
    condition_arrays = [ec[0].convert_to_nparrays() for ec in ec_list]
    data_for_varb = [[v for v in c_array[varb] if math.isfinite(v)] for c_array in condition_arrays]
#   filtered_data = list(filter(lambda x: x.size > 0, data_for_varb))

    # come up with metric for collision normalization. use cstart bias plus collision rate. 
    return data_for_varb
#    sb.boxplot(x=range(len(data_for_varb)), y=data_for_varb)
#    pl.show()

# lambda x: -1*(2*x - 1) will be the mapfunction for barrier on right
# lambda x: (2*x - 1) will be the mapfunction for barrier on left


# dv1 is barrier on left, going to the right. (pink)
# dv2 is barrier on right, going to the left. (blue)

def plot_varb_over_ecs(dv1, *dv2):
    fig, ax = pl.subplots(1, 1, figsize=(4, 1))
    sb.set(style="ticks", rc={"lines.linewidth": .75})
    if dv2 != ():
        dv2, mapfunc2 = dv2[0]
        xvals2 = []
        for i, d in enumerate(dv2):
            xvals2.append(i*np.ones(len(d)))
        xv_concat2 = np.concatenate(xvals2)
        yvals2 = list(map(mapfunc2, np.concatenate(dv2)))
        sb.pointplot(x=xv_concat2, y=yvals2, color='dodgerblue', markers='s')
 #       sb.stripplot(x=xv_concat2, y=yvals2, dodge=False, alpha=.2, zorder=0, jitter=.005, color='deeppink')
    dv, mapfunc = dv1
    xvals = []
    print(dv)
    for i, d in enumerate(dv):
        xvals.append(i*np.ones(len(d)))
    xv_concat = np.concatenate(xvals)
    yvals = list(map(mapfunc, np.concatenate(dv)))
    if dv2 == ():
        sb.pointplot(x=xv_concat, y=yvals, color='gray', markers='s')
    else:
        sb.pointplot(x=xv_concat, y=yvals, color='deeppink', markers='s')
#    sb.stripplot(x=xv_concat, y=yvals, dodge=False, alpha=.2, zorder=0, jitter=.005, color='dodgerblue')
    sb.despine()
    pl.show()


# enter complete experiment collectors (not filtered) here.

# this plot shows that the escape away from barriers on the side of the ablated mauthner is enhanced relative to
# escapes toward the ablated mauthner in N trials. escapes away from the ablated mauthner are unchanged -- should go down
# in cases where barrier is on the side of the intact mauthner. 

# left plot shows the propensity to escape in the direction dictated by the intact mauthner.
# it shows that when the fish encounters a barrier on the side of the intact mauthner, it escapes away more
# frequently than expected from the control case -- this can't be inhibition of the opposite mauthner, because it isn't there. 

# right plot shows the propensity to escape in the direction dictated by the
# ablated mauthner. if the fish encounters a barrier on the side of the ablated mauthner, it escapes in the direction dictated by the ablated mauthner at the same frequency as in control conditions, indicating that an intact mauthner is required for the bias, and that the bias is excitatory (if it were inhibitory to the opposite mauthner, would see a shift). 

# add ECs with l and n trials

    
def pairwise_l_to_n_PI(ec_left, ec_right):
    sb.set(style="ticks", rc={"lines.linewidth": 1})
    cp = sb.color_palette('husl', 8)
    # quantify this using turns towards the ablated mauthner.
    # barrier on side of ablated mauthner vs opposite.
    fig, axes = pl.subplots(2, 2, sharey=True)
    l_right_mauth_bleft = []
    l_right_mauth_bright = []
    l_left_mauth_bleft = []
    l_left_mauth_bright = []
    n_right_mauth_bright = []
    n_right_mauth_bleft = []
    n_left_mauth_bright = []
    n_left_mauth_bleft = []
    for fish_ind, fish in enumerate(ec_left[0].included_fish):
        try:
            n_trial_index = ec_left[1].included_fish.index(fish)
        except ValueError:
            print("VAL ERROR")
            continue
        pi_bright = -1 * (2 * ec_left[0].escape_data['Correct Trajectory Percentage BRight'][fish_ind] - 1)
        pi_bleft = 2 * ec_left[0].escape_data['Correct Trajectory Percentage BLeft'][fish_ind] - 1
        pi_control = -1 * (2 * ec_left[1].escape_data['Left Traj Percentage'][n_trial_index] - 1)

        if math.isfinite(pi_bright):
            n_left_mauth_bright.append(pi_control)
            l_left_mauth_bright.append(pi_bright)
            sb.lineplot(x=np.array([0, 1]), y=np.array([pi_control, pi_bright]),
                        ax=axes[0, 0], markers=True, marker='.', color=cp[3], alpha=0.3)
      
        if math.isfinite(pi_bleft):
            n_left_mauth_bleft.append(pi_control)
            l_left_mauth_bleft.append(pi_bleft)
            sb.lineplot(x=np.array([0, 1]), y=np.array([pi_control, pi_bleft]),
                        ax=axes[0, 1], markers=True, marker='.', color=cp[3], alpha=.3)

    for fish_ind, fish in enumerate(ec_right[0].included_fish):
        try:
            n_trial_index = ec_right[1].included_fish.index(fish)
        except ValueError:
            print("VAL ERROR")
            continue
        pi_bright = -1 * (2 * ec_right[0].escape_data['Correct Trajectory Percentage BRight'][fish_ind] - 1)
        pi_bleft = 2 * ec_right[0].escape_data['Correct Trajectory Percentage BLeft'][fish_ind] - 1
        pi_control = -1 * (2 * ec_right[1].escape_data['Left Traj Percentage'][n_trial_index] - 1)

        if math.isfinite(pi_bright):
            n_right_mauth_bright.append(pi_control)
            l_right_mauth_bright.append(pi_bright)
            sb.lineplot(x=np.array([0, 1]), y=np.array([pi_control, pi_bright]),
                        ax=axes[1, 0], markers=True, marker='.', color=cp[1], alpha=.3)
      
        if math.isfinite(pi_bleft):
            n_right_mauth_bleft.append(pi_control)
            l_right_mauth_bleft.append(pi_bleft)
            sb.lineplot(x=np.array([0, 1]), y=np.array([pi_control, pi_bleft]),
                        ax=axes[1, 1], markers=True, marker='.', color=cp[1], alpha=.3)



#    fig2, axes2 = pl.subplots(1, 2, sharey=True)
    sb.pointplot(x=np.concatenate([np.zeros(len(n_left_mauth_bright)),
                                   np.ones(len(l_left_mauth_bright))]),
                 y=np.concatenate([n_left_mauth_bright,
                                   l_left_mauth_bright]), color='k', ax=axes[0, 0], zorder=20)
    sb.pointplot(x=np.concatenate([np.zeros(len(n_left_mauth_bleft)),
                                   np.ones(len(l_left_mauth_bleft))]),
                 y=np.concatenate([n_left_mauth_bleft,
                                   l_left_mauth_bleft]), color='k', ax=axes[0, 1], zorder=20)
    sb.pointplot(x=np.concatenate([np.zeros(len(n_right_mauth_bright)),
                                   np.ones(len(l_right_mauth_bright))]),
                 y=np.concatenate([n_right_mauth_bright,
                                   l_right_mauth_bright]), color='k', ax=axes[1, 0], zorder=20)
    sb.pointplot(x=np.concatenate([np.zeros(len(n_right_mauth_bleft)),
                                   np.ones(len(l_right_mauth_bleft))]),
                 y=np.concatenate([n_right_mauth_bleft,
                                   l_right_mauth_bleft]), color='k', ax=axes[1, 1], zorder=20)

    sb.despine()


    fig2, axes2 = pl.subplots(2, 1, sharey=True)

           
    # you have L fish and R fish. combine them.
    # right_mauth_bleft == left_mauth_bright
    # right_mauth_bright == left_mauth_bleft

    # swims in direction of ablated mauthner. currently in right PI index.
    # this is correct for right ablated. 1 minus for left ablated.
    
    # add these, doing a 1- on 

    l_barrier_on_ablated_side = l_right_mauth_bright + [-pi for pi in l_left_mauth_bleft]
    l_barrier_on_nonablated_side = l_right_mauth_bleft + [-pi for pi in l_left_mauth_bright]

    n_barrier_on_ablated_side = n_right_mauth_bright + [-pi for pi in n_left_mauth_bleft]
    n_barrier_on_nonablated_side = n_right_mauth_bleft + [-pi for pi in n_left_mauth_bright]

    for i, (pi_barr, pi_control) in enumerate(zip(l_barrier_on_ablated_side, n_barrier_on_ablated_side)):
        if i < len(l_right_mauth_bright):
            curr_color = cp[3]
        else:
            curr_color = cp[1]
        sb.lineplot(x=np.array([0, 1]), y=np.array([pi_control, pi_barr]),
                    ax=axes2[0], markers=True, marker='.', color=curr_color, alpha=0.3)

    for i, (pi_barr, pi_control) in enumerate(zip(l_barrier_on_nonablated_side, n_barrier_on_nonablated_side)):
        if i < len(l_right_mauth_bleft):
            curr_color = cp[3]
        else:
            curr_color = cp[1]
        sb.lineplot(x=np.array([0, 1]), y=np.array([pi_control, pi_barr]),
                    ax=axes2[1], markers=True, marker='.', color=curr_color, alpha=0.3)
        
    sb.pointplot(x=np.concatenate([np.zeros(len(n_barrier_on_ablated_side)),
                                   np.ones(len(l_barrier_on_ablated_side))]),
                 y=np.concatenate([n_barrier_on_ablated_side,
                                   l_barrier_on_ablated_side]), color='k', ax=axes2[0], zorder=20)
    sb.pointplot(x=np.concatenate([np.zeros(len(n_barrier_on_nonablated_side)),
                                   np.ones(len(l_barrier_on_nonablated_side))]),
                 y=np.concatenate([n_barrier_on_nonablated_side,
                                   l_barrier_on_nonablated_side]), color='k', ax=axes2[1], zorder=20)
        
#    axes[0, 0].set_ylim([0, 1.2])
    print(scipy.stats.ttest_rel(n_left_mauth_bright, l_left_mauth_bright))
    print(scipy.stats.ttest_rel(n_left_mauth_bleft, l_left_mauth_bleft))
    print(scipy.stats.ttest_rel(n_right_mauth_bright, l_right_mauth_bright))
    print(scipy.stats.ttest_rel(n_right_mauth_bleft, l_right_mauth_bleft))

    print(scipy.stats.ttest_rel(n_barrier_on_ablated_side, l_barrier_on_ablated_side))
    print(scipy.stats.ttest_rel(n_barrier_on_nonablated_side, l_barrier_on_nonablated_side))


    
    pl.show()


    
# need a hairplot for N trials combined across all experiments.
# need a hairplot for L and D trials with collisions.

    

def hairplot_heatmap(ec, ec_index):
    combined_data = ec[ec_index].convert_to_nparrays()
    fig, ax = pl.subplots(1, 2)
    bounds = 100
    coord_matrix_bleft = np.zeros([bounds*2, bounds*2])
    coord_matrix_bright = np.zeros([bounds*2, bounds*2])
    collision_coords_bleft = []
    collision_coords_bright = []
    timewin = [10, -1]
 #   ax.set_aspect('equal')
#        escape_win = [10, 25]
    for bl_coords, bl_collision in zip(combined_data['Barrier On Left Trajectories'],
                                       combined_data['Collision Trials BLeft']):
        bl_coords += bounds
        bl_coords_unique = list(set(zip(bl_coords[0][timewin[0]:timewin[1]],
                                        bl_coords[1][timewin[0]:timewin[1]])))
        for x, y in bl_coords_unique:
            try:
                coord_matrix_bleft[-y, x] += 1
            except IndexError:
                print(x)
                print(y)
                continue
        if bl_collision != []:
            collision_coords_bleft.append([bl_coords[0][bl_collision[0]],
                                           bl_coords[1][bl_collision[0]]])
            ax[0].plot(bl_coords[0],
                       bl_coords[1])

                                          
    for br_coords, br_collision in zip(combined_data['Barrier On Right Trajectories'],
                                       combined_data['Collision Trials BRight']):
        br_coords += bounds
        br_coords_unique = list(set(zip(br_coords[0][timewin[0]:timewin[1]],
                                        br_coords[1][timewin[0]:timewin[1]])))
        for x, y in br_coords_unique:
            try:
                coord_matrix_bright[-y, x] += 1
            except IndexError:
                continue
        if br_collision != []:
            collision_coords_bright.append([br_coords[0][br_collision[0]],
                                            br_coords[1][br_collision[0]]])
            ax[1].plot(br_coords[0],
                       br_coords[1])
            
                                           

    clipmin_left = np.ceil(sum(sum(coord_matrix_bleft)) / 1000)
    clipmax_left = np.ceil(sum(sum(coord_matrix_bleft)) / 100)
    clipmin_right = np.ceil(sum(sum(coord_matrix_bright)) / 1000)
    clipmax_right = np.ceil(sum(sum(coord_matrix_bright)) / 100)
    cpal = [(1.0, 1.0, 1.0)] + sb.color_palette('hot')
    cmap = sb.color_palette(cpal, as_cmap=True)
 #   sb.heatmap(np.clip(cv2.boxFilter(coord_matrix_bleft, ddepth=-1, ksize=(5, 5),
 #                                    normalize=False), clipmin_left, clipmax_left),
   #            ax=ax[0], cmap=cmap)
    ax[0].scatter([x[0] for x in collision_coords_bleft],
                  [y[1] for y in collision_coords_bleft])
 #   sb.heatmap(np.clip(cv2.boxFilter(coord_matrix_bright, ddepth=-1, ksize=(5, 5),
 #                                    normalize=False), clipmin_right, clipmax_right),
    #           ax=ax[1], cmap=cmap)
    ax[1].scatter([x[0] for x in collision_coords_bright],
                  [y[1] for y in collision_coords_bright])
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[0].vlines(bounds, 0, bounds*2, colors='gray', linestyles='dashed', lw=.5)
    ax[1].vlines(bounds, 0, bounds*2, colors='gray', linestyles='dashed', lw=.5)
    pl.show()
    return coord_matrix_bleft, coord_matrix_bright

            
    
        
def hairplot_w_preferenceindex(fish, cond, led_filter, frontwin, *nocolor):
    ec = make_ec_collection(fish, cond, led_filter, frontwin)
    combined_data = ec[0][0].convert_to_nparrays()
    barrier_on_right_arrays = ec[1][0].convert_to_nparrays()
    barrier_on_left_arrays = ec[2][0].convert_to_nparrays()
    if nocolor == ():
        plotcolors = ['deeppink', 'dodgerblue']
    else:
        plotcolors = ['k', 'k']
#    fig, axes = pl.subplots(2, 1)
    fig = pl.figure()
    gs = fig.add_gridspec(8, 7)
    ax1 = fig.add_subplot(gs[0, 1:6])
    ax2 = fig.add_subplot(gs[1:, :])
    axes = [ax1, ax2]
    axes[0].set_xlim([-1.1, 1.1])
    axes[0].set_ylim([-2, 2])
    bound = 100
    axes[1].set_xlim([-bound, bound])
    axes[1].set_ylim([-bound, bound])
    axes[1].vlines(0, -50, 50, colors='gray', linestyles='dashed')
    axes[1].set_aspect('equal')
    correct_moves = 0
    incorrect_moves = 0
#        escape_win = [10, 25]
    for bl_coords, br_coords in itz.zip_longest(
          combined_data['Barrier On Left Trajectories'],
          combined_data['Barrier On Right Trajectories']):
        if bl_coords is not None:
            axes[1].plot(bl_coords[0], bl_coords[1],
                         color=plotcolors[0], linewidth=.8, alpha=.4)
        if br_coords is not None:
            axes[1].plot(br_coords[0], br_coords[1],
                         color=plotcolors[1], linewidth=.8, alpha=.4)
  # CHANGE SIGN OF THIS TO SHOW CORRECT MOVES AS SIMPLY LEFT TURNS
    correct_moves, incorrect_moves = combined_data['Total Correct Trajectories']
  
    correct_percentage = int(round(
        100*correct_moves/(correct_moves+incorrect_moves), 0))
    axes[1].text(
        -45, -80,
        str(correct_percentage) + '% Away From Barrier', size=10)


    left_percentage = int(round(100*combined_data['Left Vs Right Trajectories'][0] / sum(
            combined_data['Left Vs Right Trajectories'])))
    axes[1].text(
        -45, -100, str(left_percentage) + '% Left Trajectories', size=10)
    
    # axes[1].text(
    #     -95, -95,
    #     str(incorrect_moves) + ' Incorrect', size=10)

    # condition left prob right, condition right prob left per fish

    cstart_percentage_b_on_right = [(2*csp -1) for
                                    csp in barrier_on_right_arrays['Correct CStart Percentage']]
                                       
    cstart_percentage_b_on_left = [-1*(2*csp-1) for
                                   csp in barrier_on_left_arrays['Correct CStart Percentage']]


    # this val will be negative if doing well, positive if not
    
    correct_traj_percentage_b_on_left = [(2*ctp - 1) for
                                         ctp in combined_data['Correct Trajectory Percentage BLeft'] if math.isfinite(ctp)]

    
    # this val will be negative if doing well, positive if not. 

    correct_traj_percentage_b_on_right = [-1*(2*ctp - 1) for
                                          ctp in combined_data['Correct Trajectory Percentage BRight'] if math.isfinite(ctp)]

    # sb.barplot(data=cstart_percentage_b_on_right,
    #            ax=axes[0], estimator=np.nanmean, color=plotcolors[0], orient='h', errwidth=.4)

    # sb.barplot(data=cstart_percentage_b_on_left,
    #             ax=axes[0], estimator=np.nanmean, color=plotcolors[1], orient='h', errwidth=.4)

    sb.barplot(data=correct_traj_percentage_b_on_left,
                ax=axes[0], estimator=np.nanmean, color=plotcolors[0], orient='h', errwidth=.4)
    
    sb.barplot(data=correct_traj_percentage_b_on_right,
               ax=axes[0], estimator=np.nanmean, color=plotcolors[1], orient='h', errwidth=.4)
    sb.despine()
#     sb.swarmplot(data=cstart_percentage_data_c1, ax=axes[0], color="b", alpha=.35, size=3, orient='h')
 #   sb.swarmplot(data=cstart_percentage_data_c2, ax=axes[0], color="k", alpha=.35, size=3, orient='h')
    axes[1].set_axis_off()
    pl.tight_layout()
    pl.show()
    return ec


def plot_all_results(cond_collector_list):
    cond_list = [c.condition for c in cond_collector_list]
    # if cond_list != cond_list_orig:
    #     raise Exception('cond_collector in wrong order')
#    esc_assess_time = (cond_collector_list[
#        0].timerange[1] - cond_collector_list[0].timerange[0]) / 2
    cpal = sb.color_palette()
    fig, axes = pl.subplots(1, 2)
    axes[0].set_title('Fish Orientation vs. Barrier (rad)')
    axes[1].set_title('Distance from Barrier at Escape Termination')
    cond_data_arrays = []
    for cond_ind, cond_data_as_list in enumerate(cond_collector_list):
        cond_data = cond_data_as_list.convert_to_nparrays()
        cond_data_arrays.append(cond_data)
        try:
#            tsplot(cond_data['Heading vs Barrier'], axes[0])
            ts_plot(cond_data['Heading vs Barrier'], axes[0])
#            sb.lineplot(data=df_conddata,
 #                       ax=axes[0], estimator=np.nanmean, color=cpal[cond_ind])
        except RuntimeError:
            print(cond_data['Heading vs Barrier'])
      
    pl.tight_layout()
    barfig, barax = pl.subplots(3, 3, figsize=(8, 6))
    barax[0, 0].set_title('% CStart Away from Barrier')
    barax[1, 0].set_title('# Collisions')
    barax[0, 1].set_title('CStart Latency (ms)')
    barax[0, 2].set_title('CStart Angle (deg)')
    barax[1, 1].set_title('Correct Trajectory %')
   # barax[1, 2].set_title('Total Time Spent in Barrier Zone')
#    barax[1, 2].set_title('Number of Duds')
    barax[1, 2].set_title('% Left CStarts')
    barax[2, 0].set_title('Phototaxis to Tap Time')
    barax[2, 1].set_title('CStart Rel to Prevbout')
    barax[2, 2].set_title('Pooled Correct CStart Percentage')
    xlocs = np.arange(len(cond_data_arrays))
    correct_bars = [c['Total Correct CStarts'] for c in cond_data_arrays]
    total_bars = [c['Total CStarts'] for c in cond_data_arrays]
    dud_bars = [c['No Escape'] for c in cond_data_arrays]
    collision_bars = [c['Total Collisions'] for c in cond_data_arrays]
    total_valid_bars = [c['Total Valid Trials'] for c in cond_data_arrays]
    left_cstart_bar = [c['Left vs Right CStarts'] for c in cond_data_arrays]
    print(left_cstart_bar)
#    barax[1, 0].bar(xlocs, total_bars)
#    barax[1, 0].bar(xlocs, collision_bars)
    barax[2, 2].bar(xlocs, total_bars)
    barax[2, 2].bar(xlocs, correct_bars)
#    barax[1, 2].bar(xlocs, total_valid_bars)
#    barax[1, 2].bar(xlocs, dud_bars)
    sb.barplot(data=left_cstart_bar, ax=barax[1, 2], estimator=np.nanmedian, palette=cpal)
    sb.swarmplot(data=left_cstart_bar, ax=barax[1, 2], color="0", alpha=.35, size=3)
#    sb.violinplot(data=[np.concatenate(bdist, axis=0) for bdist 
 #                       in [c['Distance From Barrier After Escape']
  #                      for c in cond_data_arrays]],  ax=axes[1])
    cstart_percentage_data = [cdir[~np.isnan(cdir)] for
                              cdir in [c['Correct CStart Percentage']
                              for c in cond_data_arrays]]
    cstart_rel_to_prevbout = [cdir[~np.isnan(cdir)] for
                              cdir in [c['CStart Rel to Prevbout']
                              for c in cond_data_arrays]]
    ctraj_percentage_data = [cdir[~np.isnan(cdir)] for
                             cdir in [c['Correct Trajectory Percentage']
                                      for c in cond_data_arrays]]
    

# add swarmplot on top of barplot and make barplot transparent
#sns.barplot(x="day", y="total_bill", data=tips, capsize=.1, ci="sd")
#sns.swarmplot(x="day", y="total_bill", data=tips, color="0", alpha=.35)

    cpal = sb.color_palette("hls", 8)
    sb.barplot(data=cstart_percentage_data, 
               ax=barax[0, 0], estimator=np.nanmean, palette=cpal)
    sb.swarmplot(data=cstart_percentage_data, ax=barax[0,0], color="0", alpha=.35, size=3)

    sb.barplot(data=ctraj_percentage_data, 
               ax=barax[1, 1], estimator=np.nanmean, palette=cpal)
    sb.swarmplot(data=ctraj_percentage_data, ax=barax[1,1], color="0", alpha=.35, size=3)


    
    sb.barplot(data=cstart_rel_to_prevbout, ax=barax[2, 1], estimator=np.nanmean, palette=cpal)
    collision_percentage_data = [clp[~np.isnan(clp)] for
                                 clp in [c['Collision Percentage']
                                         for c in cond_data_arrays]]
    sb.barplot(data=collision_percentage_data, 
               ax=barax[1, 0], estimator=np.nanmean, palette=cpal)
#    sb.swarmplot(data=collision_percentage_data, ax=barax[1,0], color="0", alpha=.35, size=3)

    sb.boxplot(data=[2*clat[~np.isnan(clat)] for
                     clat in [c['CStart Latency']
                              for c in cond_data_arrays]],
               ax=barax[0, 1], whis=np.inf)
    sb.boxplot(data=[cang[~np.isnan(cang)] for
                     cang in [c['CStart Angle']
                              for c in cond_data_arrays]],
               ax=barax[0, 2], notch=True)
    
    taps_per_entry = [num_entries[~np.isnan(num_entries)] for
                      num_entries in [c['Taps Per Entry Into Arena']
                                      for c in cond_data_arrays]]
    sb.boxplot(data=taps_per_entry,
               ax=barax[1, 1])
    # sb.boxplot(data=[dur[~np.isnan(dur)] for
    #                  dur in [c['Total Time In Center']
    #                          for c in cond_data_arrays]],
    #            ax=barax[1, 2])

    
    sb.boxplot(data=[ptax[~np.isnan(ptax)] for
                     ptax in [c['Phototaxis to Tap Time']
                              for c in cond_data_arrays]],
               ax=barax[2, 0])
    pl.tight_layout()
    pl.show()
    return [cang[~np.isnan(cang)] for
                     cang in [c['CStart Angle']
                              for c in cond_data_arrays]], [2*clat[~np.isnan(clat)] for 
                     clat in [c['CStart Latency']
                              for c in cond_data_arrays]]


def parse_obj_by_trial(drct_list, cond, mods):
    os.chdir('/Volumes/Esc_and_2P/Escape_Results/')
#    os.chdir('/Volumes/Seagate/AndrewTestData/')
    for drct in drct_list:
        fish_id = '/' + drct
        pl.ioff()
        area_thresh = 47
#        area_thresh = 30
        esc_dir = os.getcwd() + fish_id
        print(esc_dir)
        plotcstarts = False
        escape_obj = Escapes(cond, esc_dir, area_thresh, 1)
        escape_obj.exporter()
        for i in range(mods):
            esc = copy.deepcopy(escape_obj)
            esc.backgrounds = esc.backgrounds[i::mods]
            esc.pre_escape = esc.pre_escape[i::mods]
            esc.xy_escapes = esc.xy_escapes[i::mods]
            esc.stim_data = esc.stim_data[i::mods]
            esc.movie_id = esc.movie_id[i::mods]
            esc.numgrayframes = esc.numgrayframes[i:mods]
            esc.condition = escape_obj.condition + str(i)
            esc.load_experiment()
            esc.trial_analyzer(plotcstarts)
            esc.exporter()


def experiment_collector(drct_list, cond_list, filter_settings, *new_exps):
    cond_collector_list = [Condition_Collector(cl) for cl in cond_list]
    for cc in cond_collector_list:
        cc.filter_index = filter_settings[0]
        cc.filter_function = filter_settings[1]
        cc.filter_by_led = filter_settings[2]
    if new_exps != ():
        new_exps = new_exps[0]
    os.chdir('/Volumes/Esc_and_2P/Escape_Results')
#    os.chdir('/Volumes/Seagate/AndrewTestData/')
    for newexp_dirct in new_exps:
        fish_id = '/' + newexp_dirct
        pl.ioff()

        # CHANGE THIS TO ALTER THE MINIMUM AREA THE FISH TAKES UP BEFORE BEING
        # CALLED A FISH. 
        
        area_thresh = 47
#        esc_dir = os.getcwd() + fish_id
        esc_dir = '/Volumes/Esc_and_2P/Escape_Results' + fish_id
#        esc_dir = '/Volumes/Seagate/AndrewTestData' + fish_id
        print(esc_dir)
        plotcstarts = False
        for cond in cond_list:
            try:
                escape_obj = Escapes(cond, esc_dir, area_thresh)
                print("past trial generator")
                escape_obj.trial_analyzer(plotcstarts)
                print("past trial analyzer")
                escape_obj.exporter()
            except IOError as err:
                print(err)
                print("No " + cond + " Trials in fish" + str(esc_dir))
# CATCH FOR SPLITTING ACCORDING TO TRIAL GOES HERE. 
            
    for drct in drct_list:
        drct = '/Volumes/Esc_and_2P/Escape_Results/' + drct
#        drct = '/Volumes/Seagate/AndrewTestData/' + drct
        for cond_ind, cond_collector in enumerate(cond_collector_list):
            try:
                esc_obj = pickle.load(open(
                    drct + '/escapes_' + cond_list[cond_ind] + '.pkl', 'rb'))
                print(esc_obj.stim_init_times)
                print(esc_obj.escape_latencies)
                cond_collector.update_ddict(esc_obj)
            except IOError:
                print(drct)
                print("PICKLING ERROR")
    return cond_collector_list


def make_ec_collection(fish, cond, led_filter, frontwin):
    ec_all = experiment_collector(fish, [cond], [0, lambda x: (frontwin < x < 180) or (-180 < x < -frontwin), led_filter])
    ec_r = experiment_collector(fish, [cond], [0, lambda x: (frontwin < x < 180), led_filter])
    ec_l = experiment_collector(fish, [cond], [0, lambda x: (-180 < x < -frontwin), led_filter])
    return ec_all, ec_r, ec_l
    

# proximityval is the amnt of pixels from the location point to the barrier edge
# (i.e. there is space between the pinpointed location and barrier)

def collision(xb, yb, bd, x, y):  
    proximity_val = 6
    vec = np.sqrt((x - xb)**2 + (y - yb)**2)
    if math.isnan(x):
        return False
    elif vec < (bd / 2) + proximity_val:
        return True
    else:
        return False


def boxplot_across_conditions(data_lists, cmap):
    xvals = []
    yvals = []
    for i, dl in enumerate(data_lists):
        xvals.append(i*np.ones(len(dl)))
        yvals.append(dl)
    sb.boxplot(x=np.concatenate(xvals), y=np.concatenate(yvals), palette=cmap)
    


if __name__ == '__main__':






    
    viswin = 25
    visfunc = lambda x: (-180 + viswin < x < -viswin) or (viswin < x < 180-viswin)

    # TODO 11/10/21:
    # Collisions introduce a bit of a problem with trajectory correctness assignment.
    # Seems like there are some trials with a crash followed by an escape in the opposite direction.
    # Make sure you don't allow a "correct trajectory" call when there is a collision.
    # interestingly though, i think collisions are messing up the white barrier data.
    # the cstart rate is 70% away. trajectories match if you don't throw out collisions.
    # the fish may be much closer to the barrier when the tap is initiated, meaning
    # collisions might be called too easily. have to investigate this further for both
    # dark trials and white barrier trials bc these are the only two trials where
    # collisions play a significant role in the trajectories. seems like correct
    # trajectory data is now underreported for both dark and light -- check the
    # brleft and brright coord parsing. 

    # run out with 50 as the collision_bound to include all collisions. then
    # deal with collisions in the Condition_Collector objects now that you've collected
    # all collision times. the collision times stored in Collition Trials BRight and BLeft
    # are times from the 100th frame onward, so the coords from trajectory containers
    # can be directly indexed with these numbers. 

    

  #  dv = collect_varb_across_ec([four_b, red24mm_4mmdist, red12mm_4mmdist_2h, red12mm_4mmdist, red48mm_8mmdist_2h, red48mm_8mmdist], 'l', 'Correct CStart Percentage')


    
    # VIRTUAL BARRIERS
    virtual = ['091119_4', '091019_1', '091019_2', '091119_1', '091119_2', '091119_3',
               '091119_5', '091319_1', '091319_2', '091319_3',
               '091319_4', '092719_1','092719_2', '092719_3', '092719_4', '100219_1', '100219_2',
               '100219_3', '100219_4', '100219_6', '100219_7', '100219_5']
  #  ec1 = experiment_collector(virtual, ['v', 'i', 'n'], [0, [], 1]) #virtual)
  #  plot_all_results(ec1)

    four_w = ['072319_1', '072319_2', '072319_3', '072319_4',
              '072319_5', '072319_6', '072319_7', '072419_3',
              '072419_4', '072419_5', '072419_6', '072419_7',
              '072419_8', '072419_9', '072519_1', '072619_1',
              '072619_2', '072619_3', '072619_4']
    
    four_b = ['022619_2', '030519_1', '030519_2', '030719_1',
              '030719_2', '030719_3', '032619_1', '032819_1',
              '032919_1', '040319_1', '040419_2', '040519_2',
              '041719_1', '041819_1', '041919_2', '042319_1',
              '042719_1', '102319_1', '102319_2', '102419_1',
              '102519_1', '110219_1', '110219_2']


 #   ec_fourw = experiment_collector(four_w, ['l', 'n'],
 #                                   [0, visfunc, 1])
  #  coordmat_l, coordmat_l = hairplot_heatmap(ec_fourw, 0)
                                        
 #   coordmat_l, coordmat_l = hairplot_heatmap(ec_fourw, 0)
  #  hairplot_w_preferenceindex(four_w, 'l', 1, 30)
  #  hairplot_w_preferenceindex(four_b, 'd', 1, 20)
    
  #  plot_all_results(ec_fourw)

# used to test stim and cstart detection -- perfect! 
#    four_b_1 = ['022619_2', '030519_1']

  #  ec_fourb = experiment_collector(four_b, ['l', 'd', 'n'],
  #                                  [0, visfunc, 1]) 
 #   coordmat_l, coordmat_l = hairplot_heatmap(ec_fourb, 1)
  #  plot_all_results(ec_fourb)
#    cstart_angle, cstart_latency = plot_all_results(ec1)

    wik_mauthner_l = ['052721_1', '060421_1',
                      '060421_3', '060421_4', '060421_5',
                      '021320_1', '021320_2', '021320_3',
                      '022120_1', '022120_2', '022120_3', '061021_1',
                      '061021_3', '061021_6']

    wik_mauthner_r = ['052721_2', '052721_3', '052821_2', '052821_3',
                      '052821_4', '060321_1', '060321_2',
                      '060321_4', '060321_5', '060321_6', '060321_7',
                      '060421_8', '061021_4', '061021_5']

   # mauth_l_ec = experiment_collector(wik_mauthner_l, ['l', 'n'], [0, [], 0], #wik_mauthner_l)
#    plot_all_results(mauth_l_ec)
   # mauth_r_ec = experiment_collector(wik_mauthner_r, ['l', 'n'], [0, [], 0], #wik_mauthner_r)
  #  pairwise_l_to_n_plot(mauth_l_ec, mauth_r_ec)
 #   plot_all_results(mauth_r_ec)

    
    red24mm_4mmdist = ['061121_1', '061121_2', '061121_3', '061121_4',
                       '061121_5', '061121_6', '061121_7', '061421_1',
                       '061421_2', '061421_3', 
                       '061521_2', '061521_3', '061521_4', '061521_5',
                       '061521_6']

  #  red24mm_4mmdist_ec = experiment_collector(red24mm_4mmdist, ['l'], [0, lambda x: (-180 < x < -25) or (25 < x < 180), 1]) #, red24mm_4mmdist)
  #  coordmat_l, coordmat_l = hairplot_heatmap(red24mm_4mmdist_ec, 0)
  #  plot_all_results(red24mm_4mmdist_ec)
    
  #  collision_stats, num_n_trials = infer_collisions([red24mm_4mmdist_ec], False)
  #  plot_collision_stat(collision_stats, num_n_trials)


  #  red24mm_4mmdist_ec = experiment_collector(red24mm_4mmdist, ['l', 'n'],
   #                                           [0, visfunc, 1])

   # plot_all_results(red24mm_4mmdist_ec)

    red12mm_4mmdist = ["061721_1", "061721_2", "061721_3", "061721_4", "061721_5",
                       "061721_6", "061721_7", "061821_1", "061821_2", "061821_3", 
                       "061821_4", "061821_5", "061821_6",  "062221_1", "062221_2",
                       "062221_3", "062221_4", "062221_5", "062221_6"]

   # red12mm_4mmdist_ec = experiment_collector(red12mm_4mmdist,
    #                                          ['l', 'n'], [0, visfunc, 1])
 #   plot_all_results(red12mm_4mmdist_ec)

    red48mm_8mmdist = ["062521_3", "062521_4", "063021_2",
                       "063021_3", "063021_4", "063021_5",
                       "063021_6", "070121_1", "070121_2", 
                       "070221_4", "070221_5", "070221_6", "070621_4",
                       "070721_4", "070721_5", "070721_6", "070921_2",
                       "070921_4"]

   # plot_all_results(red48mm_8mmdist_ec)

    red48mm_8mmdist_2h = [#"070821_1", "070821_2", "070821_8", "070821_9",
        #                  "070921_6",
                          "071221_1", #"071221_2",  "071221_9"

                          #note that 7/8 and 7/9 appear to have no taps! 1221_3 has taps and cstarts,                            # 1221_9 also has no taps. 
                          "071221_3",
                          "071221_5", "071221_6", "071221_7", "071221_8",
                          "071321_3", "071421_1", "071421_2",
                          "071421_3", "071421_4", "071421_5", "071421_7"]

  #   plot_all_results(red48mm_8mmdist_2h_ec)

    red12mm_4mmdist_2h = ["072921_1", "072921_2", "072921_4", "073021_1",
                          "073021_2", "073021_3", "073021_4", "073021_5",
                          "073021_7", "073021_8", "073021_9", "080221_2",
                          "080221_3", "080221_4", "080221_5", "080221_6",
                          "080221_7", "080321_1", "080321_2", "080321_3"]

    # red12mm_4mmdist_ec_2h = experiment_collector(red12mm_4mmdist_2h,
    #                                           ['l', 'n'], [0, visfunc, 1])
    # plot_all_results(red12mm_4mmdist_ec_2h)

    # red48mm_8mmdist_ec_2h = experiment_collector(red48mm_8mmdist_2h,
    #                                           ['l', 'n'], [0, visfunc, 1])
    # plot_all_results(red48mm_8mmdist_ec_2h)

    # red48mm_8mmdist_ec = experiment_collector(red48mm_8mmdist,
    #                                           ['l', 'n'], [0, visfunc, 1])
    # plot_all_results(red48mm_8mmdist_ec)

    mauth_l_ec = experiment_collector(wik_mauthner_l, ['l', 'n'],
                                       [0, visfunc, 0])
    mauth_r_ec = experiment_collector(wik_mauthner_r, ['l', 'n'],
                                       [0, visfunc, 0])
    pairwise_l_to_n_PI(mauth_l_ec, mauth_r_ec)


  #  red48mm_8mmdist_ec = experiment_collector(red48mm_8mmdist, ['l', 'n'],
  #                                            [0, [], 1], red48mm_8mmdist)
                                      
  #  red48mm_8mmdist_2h_ec = experiment_collector(red48mm_8mmdist_2h, ['l', 'n'],
  #                                               [0, [], 1], red48mm_8mmdist_2h)
                                      
  #  red12mm_4mmdist_2h_ec = experiment_collector(red12mm_4mmdist_2h, ['l', 'n'],
  #                                            [0, [], 1], red12mm_4mmdist_2h)
                                      
    fishlist = [four_b, red24mm_4mmdist, red12mm_4mmdist_2h, red12mm_4mmdist,
                red48mm_8mmdist_2h, red48mm_8mmdist, four_w]

#    ecs_bar = [experiment_collector(fish, ['l', 'd', 'n'], [0, lambda x: True, 1], fish) for fish in fishlist]
 #   ecs_virtual = [experiment_collector(fish, ['v', 'i', 'n'], [0, lambda x: True, 1], fish) for fish in [virtual]]
  #  ecs_mauth = [experiment_collector(fish, ['l', 'n'], [0, lambda x: True, 0], fish) for fish in [wik_mauthner_l, wik_mauthner_r]]


    
#    coordmat_l, coordmat_l = hairplot_heatmap(ec1, 0)
 #   plot_all_results(ec)

    
  #  collect_collision_stat(fishlist, 'l', np.ones(len(fishlist)), [0, [], 1])


  #  hairplot_w_preferenceindex(red12mm_4mmdist, 'l', 1)


    # Need a pairwise statistic for the mauthners. How biased before the trial? If all of the mistake escapes come
    # from fish that are completely biased b/c of the mauthner ablation, and none of them come from fish who were
    # 50 / 50, then there's a simpler explanation. Can do this by using included and rejected fish in each
    # condition collector. write a function that takes condcollectors from the l and n condition, then
    # does pairwise statistics: percent left vs percent right for the N trials, then percent correct for each different mauthner on the side that would activate the ablated mauthner. 
    

   # red12mm_4mmdist_2h_ec = experiment_collector(red12mm_4mmdist_2h, ['l', 'n'], [0, [], 1], red12mm_4mmdist_2h)

  #  ec_d = experiment_collector(four_b, ['d'], [0, [], 1])
  #  ec_l = experiment_collector(four_b, ['l'], [0, [], 1])
#    coordmat_l, coordmat_l = hairplot_heatmap(ec_d)
   

   
    """ PLOTS FOR PAPER """

    # white_and_virtual_list = [four_w, virtual, virtual]
    
    # white_and_virtual_bleft = collect_varb_across_ec(white_and_virtual_list, ['l', 'v', 'i'],
    #                                                 "Correct Trajectory Percentage BLeft", [0, visfunc, 1])
    # white_and_virtual_bright = collect_varb_across_ec(white_and_virtual_list, ['l', 'v', 'i'],
    #                                                  "Correct Trajectory Percentage BRight", [0, visfunc, 1])
    # white_and_virtual_correct = collect_varb_across_ec(white_and_virtual_list, ['l', 'v', 'i'],
    #                                                 "Correct Trajectory Percentage", [0, visfunc, 1])

    # plot_varb_over_ecs([white_and_virtual_bleft, lambda x: (2*x -1)],
    #                    [white_and_virtual_bright, lambda x: -1*(2*x - 1)])

    # plot_varb_over_ecs([white_and_virtual_correct, lambda x: 2*x -1])

    # red_bleft = collect_varb_across_ec([four_b, red24mm_4mmdist, red12mm_4mmdist_2h, red12mm_4mmdist, red48mm_8mmdist_2h, red48mm_8mmdist], 'l', 'Correct Trajectory Percentage BLeft', [0, visfunc, 1])

    # red_bright = collect_varb_across_ec([four_b, red24mm_4mmdist, red12mm_4mmdist_2h, red12mm_4mmdist, red48mm_8mmdist_2h, red48mm_8mmdist], 'l', 'Correct Trajectory Percentage BRight', [0, visfunc, 1])

    # red_correct = collect_varb_across_ec([four_b, red24mm_4mmdist, red12mm_4mmdist_2h, red12mm_4mmdist, red48mm_8mmdist_2h, red48mm_8mmdist], 'l', 'Correct Trajectory Percentage', [0, visfunc, 1])


    # plot_varb_over_ecs([red_bleft, lambda x: (2*x -1)],
    #                    [red_bright, lambda x: -1*(2*x - 1)])

    # plot_varb_over_ecs([red_correct, lambda x: 2*x -1])








    
  #  # plot_all_results(red12mm_4mmdist_2h_ec)

  #   correct_bleft = collect_varb_across_ec([four_b, red24mm_4mmdist, red12mm_4mmdist_2h, red12mm_4mmdist, red48mm_8mmdist_2h, red48mm_8mmdist], 'l', 'Correct Trajectory Percentage BLeft', [0, [], 1])

  #   correct_bright = collect_varb_across_ec([four_b, red24mm_4mmdist, red12mm_4mmdist_2h, red12mm_4mmdist, red48mm_8mmdist_2h, red48mm_8mmdist], 'l', 'Correct Trajectory Percentage BRight', [0, [], 1])

  #   correct_overall_comparison = collect_varb_across_ec([four_b, red24mm_4mmdist, red12mm_4mmdist_2h, red12mm_4mmdist, red48mm_8mmdist_2h, red48mm_8mmdist], 'l', 'Correct Trajectory Percentage', [0, [], 1])

  #   correct_overall_PI = [list(map(lambda x: 2*x - 1, x)) for x in correct_overall_comparison]

  #   plot_varb_over_ecs([correct_bleft, lambda x: (2*x -1)],
  #                      [correct_bright, lambda x: -1*(2*x - 1)])

  #   plot_varb_over_ecs([correct_overall_PI, lambda x: x])
    
  #   hairplot_w_preferenceindex(wik_mauthner_l, 'n', 0, "black")
  #   hairplot_w_preferenceindex(wik_mauthner_l, 'l', 0)

  #   hairplot_w_preferenceindex(wik_mauthner_r, 'n', 0, "black")
  #   hairplot_w_preferenceindex(wik_mauthner_r, 'l', 0)
    

  #   dv_ltp = collect_varb_across_ec([four_b, red24mm_4mmdist, red12mm_4mmdist_2h, red12mm_4mmdist, red48mm_8mmdist_2h, red48mm_8mmdist, virtual, four_w], 'n', 'Left Traj Percentage', [0, [], 1])
  #   dv_mauthner_l = collect_varb_across_ec([wik_mauthner_l], 'n', 'Left Traj Percentage', [0, [], 0])
  #   dv_mauthner_r = collect_varb_across_ec([wik_mauthner_r], 'n', 'Left Traj Percentage', [0, [], 0])

  #   ltp_PI = list(map(lambda x: -1*(2*x - 1), np.concatenate(dv_ltp)))
  #   mauthner_l_PI = list(map(lambda x: -1*(2*x - 1), dv_mauthner_l[0]))
  #   mauthner_r_PI = list(map(lambda x: -1*(2*x - 1), dv_mauthner_r[0]))
  #   cpal = sb.color_palette('husl', 8)
  #   cpal2 = sb.color_palette('hls', 8)
  #   fig, ax = pl.subplots(1, 1)
  #   sb.set(style="ticks", rc={"lines.linewidth": 1})
  #   sb.kdeplot(ltp_PI,
  #              color=cpal2[0], clip=[-1, 1], ax=ax)
  #   sb.kdeplot(mauthner_l_PI,
  #              clip=[-1, 1], ax=ax, color=cpal[3])
  #   sb.kdeplot(mauthner_r_PI, 
  #              clip=[-1, 1], ax=ax, color=cpal[1])
  #   ax.set_xlim([-1, 1])
  #   p_n_to_l = ttest_ind(ltp_PI, mauthner_l_PI)
  #   p_n_to_r = ttest_ind(ltp_PI, mauthner_r_PI)
  #   PI_means = list(map(np.mean, [ltp_PI, mauthner_l_PI, mauthner_r_PI]))
  #   PI_stds = list(map(np.std, [ltp_PI, mauthner_l_PI, mauthner_r_PI]))
    
 
 #    collect_collision_stat([four_b], 'l', np.ones(len([four_b])), [0, [], 1])

""" END PLOTS FOR PAPER """ 










  
    # need a ttest for 24mm barrier at 4mm and tall 48mm barrier at 8mm. 

    

    
  #  dv_c = collect_varb_across_ec([four_b, red24mm_4mmdist, red12mm_4mmdist_2h, red12mm_4mmdist, red48mm_8mmdist_2h, red48mm_8mmdist], 'l', 'Correct CStart Percentage', [0, [], 1])


  #  dv_ltp = collect_varb_across_ec([four_b, red24mm_4mmdist, red12mm_4mmdist_2h, red12mm_4mmdist, red48mm_8mmdist_2h, red48mm_8mmdist], 'n', 'Left Traj Percentage', [0, [], 1])


  #  dv_mauthner_l = collect_varb_across_ec([wik_mauthner_l], 'n', 'Left Traj Percentage', [0, [], 0])

  #  dv_mauthner_r = collect_varb_across_ec([wik_mauthner_r], 'n', 'Left Traj Percentage', [0, [], 0])

    
    # dv_c1 = collect_varb_across_ec([four_b, red24mm_4mmdist, red12mm_4mmdist_2h, red12mm_4mmdist, red48mm_8mmdist_2h, red48mm_8mmdist], 'l', 'Correct CStart Percentage', [0, [0, 180], 1])

    # dv_c2 = collect_varb_across_ec([four_b, red24mm_4mmdist, red12mm_4mmdist_2h, red12mm_4mmdist, red48mm_8mmdist_2h, red48mm_8mmdist], 'l', 'Correct CStart Percentage', [0, [-180, 0], 1])

    
 #   dv_t = collect_varb_across_ec([four_b, red24mm_4mmdist, red12mm_4mmdist_2h, red12mm_4mmdist, red48mm_8mmdist_2h, red48mm_8mmdist], 'l', 'Correct Trajectory Percentage', [0, [], 1])


#    plot_varb_over_ecs([dv_c, lambda x: x])
#    plot_varb_over_ecs([dv_t, lambda x: x])


    
# lambda x: -1*(2*x - 1) will be the mapfunction for barrier on right
# lambda x: (2*x - 1) will be the mapfunction for barrier on left

 #   plot_varb_over_ecs([dv, lambda x: (2*x -1)], [dv2, lambda x: -1*(2*x - 1)])


# MAUTHNER PLOTS

# ALL LEFT TRAJECTORY DATA. MAP lambda x: -1*(2*x - 1) to get preference index. (right turns - left turns / turns)

    # dv_ltp = list(map(lambda x: -1*(2*x - 1), np.concatenate(collect_varb_across_ec([four_b, red24mm_4mmdist, red12mm_4mmdist_2h, red12mm_4mmdist, red48mm_8mmdist_2h, red48mm_8mmdist], 'n', 'Left Traj Percentage', [0, [], 1]))))

    # dv_mauthner_l_n = list(map(lambda x: -1*(2*x - 1), collect_varb_across_ec([wik_mauthner_l], 'n', 'Left Traj Percentage', [0, [], 0])[0]))

    # dv_mauthner_r_n = list(map(lambda x: -1*(2*x - 1), collect_varb_across_ec([wik_mauthner_r], 'n', 'Left Traj Percentage', [0, [], 0])[0]))

    # dv_mauthner_l_bright = list(map(lambda x: -1*(2*x - 1), collect_varb_across_ec([wik_mauthner_l], 'l', 'Correct Trajectory Percentage BRight', [0, [], 0])[0]))

    # dv_mauthner_l_bleft = list(map(lambda x: 2*x - 1, collect_varb_across_ec([wik_mauthner_l], 'l', 'Correct Trajectory Percentage BLeft', [0, [], 0])[0]))

    # dv_mauthner_r_bright = list(map(lambda x: -1*(2*x - 1), collect_varb_across_ec([wik_mauthner_r], 'l', 'Correct Trajectory Percentage BRight', [0, [], 0])[0]))

    # dv_mauthner_r_bleft = list(map(lambda x: 2*x - 1, collect_varb_across_ec([wik_mauthner_r], 'l', 'Correct Trajectory Percentage BLeft', [0, [], 0])[0]))   


    # First show the clipped KDE plots for N trials in control and both mauthner ablated cases. 
    # Its key to have a pairwise comparison for barriers. You want a pointplot w ste of the mean
    # for each condition to the next in DEEP black. Then you want the individual datapoints with
    # lines pointing from no barriers to barrier conditions in both on side of mauthner and
    # opposite side of mauthner cases (4 total panels). you want paired ttests for each setup. THese can be in
    # alpha dodger blue to indicate left of barrier, pink for right of barrier. so gray points
    # to corresponding blue and pink points. And a black error barred mean change with sig changes.
    # if these are inconclusive you can speculate as to the reason why, but this was the original
    # data that made me say "wow!" so it should look right.

    # for statistics, take the absolute value of each condition's preference index data to compare across conditions with unpaired tests. for mauthner, use a test on each condition. 

   # cp = sb.color_palette('hls', 8)
   # fig, (ax, ax2) = pl.subplots(1, 2)
   # sb.kdeplot(dv_ltp, color=cp[0], clip=[-1, 1], ax=ax)
   # sb.kdeplot(dv_mauthner_l_n, color=cp[1], clip=[-1, 1], ax=ax)
  #  sb.kdeplot(dv_mauthner_l_bright, color=cp[2], clip=[-1, 1], ax=ax)
   # sb.kdeplot(dv_mauthner_l_bleft, color=cp[3], clip=[-1, 1], ax=ax)
   # sb.kdeplot(dv_mauthner_r_n, color=cp[4], clip=[-1, 1], ax=ax)
#    sb.kdeplot(dv_mauthner_r_bleft, color=cp[5], clip=[-1, 1], ax=ax)
 #   sb.kdeplot(dv_mauthner_r_bright, color=cp[6], clip=[-1, 1], ax=ax)

   # ax.set_xlim([-1, 1])

  #  boxplot_across_conditions([dv_ltp, dv_mauthner_l_n, dv_mauthner_l_bright, dv_mauthner_l_bleft,
   #                            dv_mauthner_r_n, dv_mauthner_r_bleft, dv_mauthner_r_bright], cp)

#    pl.show()
#    p_rn_rbright = ttest_ind(list(map(lambda x: -1*(2*x - 1), dv_mauthner_r_n[0])), list(map(lambda x: -1*(2*x - 1), dv_mauthner_r_bright[0])))

  #  mauth_l_ec = experiment_collector(wik_mauthner_l, ['l', 'n'], [0, [], 0]) #, wik_mauthner_l)
#    plot_all_results(mauth_l_ec)
 #   mauth_r_ec = experiment_collector(wik_mauthner_r, ['l', 'n'], [0, [], 0]) # wik_mauthner_r)
 #    pairwise_l_to_n_PI(mauth_l_ec, mauth_r_ec)
 #   plot_all_results(mauth_r_ec)



#   sb.distplot(list(map(lambda x: -1*(2*x - 1), np.concatenate(dv_ltp))), color=cp[0], kde=False)
#   sb.distplot(list(map(lambda x: -1*(2*x - 1), dv_mauthner_l_n[0])), color=cp[1], kde=False)
#   sb.distplot(list(map(lambda x: -1*(2*x - 1), dv_mauthner_r_n[0])), color=cp[2], kde=False)

# cut it at -1, 1. 


   # each lesion has a significant effect on preference index. 

  #  p_n_to_l = ttest_ind(list(map(lambda x: -1*(2*x - 1), np.concatenate(dv_ltp))),
  #                      list(map(lambda x: -1*(2*x - 1), dv_mauthner_l_n[0])))

  #  p_n_to_r = ttest_ind(list(map(lambda x: -1*(2*x - 1), np.concatenate(dv_ltp))),
  #                      list(map(lambda x: -1*(2*x - 1), dv_mauthner_r_n[0])))


#   boxplot_across_conditions(
   

   # COULD MAKE ARGUMENT THAT SOME HAVE INCOMPLETE LESIONS, SO ONES WITH INCOMPLETE LESIONS ARE INHIBITED, OTHERS
   # STAY THE SAME. 
   
   
  

  

 

# TODO 9/20/2021

# get a collision metric for each condition collector. the collision metric is the
# amount of collisions total divided by the number of incorrect decisions.

# do mauthner trajectory analysis.


# this is a function you can call on no barrier escape objects. (e.g. escape_nb.infer_collisions(escape_obj_w_barrier). generalize it to many compiled escapes -- each obj now has an initial condition already called. 




# TODO

# tell the tale of the mauthner experiment in the parlance of PI.
# yield preditive PIs for each case. all the data you need is in the experiment collectors above.

# align all n-trial escapes from red barrier trials into the initial conditions
# for each experiment in red barrier trials. cool.

# navigation plot. make sure you understand exactly what the stat is.
# current thought is "No. Visits" (in legend say filtered w 2D gaussian).





