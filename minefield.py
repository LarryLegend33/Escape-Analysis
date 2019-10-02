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
from collections import deque
from toolz.itertoolz import sliding_window, partition
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelmin, argrelmax, argrelextrema
from astropy.convolution import convolve, Gaussian1DKernel
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.image import AxesImage
from matplotlib.colors import Colormap
import pdb
from itertools import izip_longest


# This is the main class for this program.
# Escape objects take in barrier locations, XY coords of the fish,
# movies during taps, raw stim files (i.e. the position of the
# tapper light over time), and relevant bacgkrounds for each
# tap trial. The Escape class contains methods for finding the fish's
# orientation relative to a barrier, finding the timing and angle
# of the c-start by calculating the exact moment of the tap,
# and plotting the escape trajectory relative to a barrier position

class Condition_Collector:
    def __init__(self, condition):
        self.condition = condition
        self.escape_data = {'Heading vs Barrier': [],
                            'Distance From Barrier After Escape': [],
                            'Collision Percentage': [],
                            'CStart Latency': [],
                            'CStart Angle': [],
                            'Phototaxis to Tap Time': [],
                            'Correct CStart Percentage': [],
                            'CStart Rel to Prevbout': [], 
                            'Taps Per Entry Into Arena': [],
                            'Total Time In Center': [],
                            'Barrier On Left Trajectories': [],
                            'Barrier On Right Trajectories': []}
        self.timerange = []

    def update_ddict(self, escape_obj):
        if escape_obj.condition != self.condition:
            raise Exception('input condition mistmatch with class')
        else:
            self.timerange = escape_obj.timerange
            tap_times = []
            for gf in escape_obj.pre_escape:
                try:
                    num_gfs = len(gf)
                except TypeError:
                    num_gfs = 1
                tap_times.append(num_gfs / 200.0)
            self.escape_data['Phototaxis to Tap Time'] += tap_times
            last_x = [c[0][escape_obj.timerange[1]]
                      for c in escape_obj.xy_coords_by_trial]
            last_y = [c[1][escape_obj.timerange[1]]
                      for c in escape_obj.xy_coords_by_trial]
            last_xy = zip(last_x, last_y)
            self.escape_data['Distance From Barrier After Escape'].append(
                [magvector([xyl[0] - bxy[0], xyl[1] - bxy[1]]) for
                 xyl, bxy in zip(last_xy, escape_obj.barrier_xy_by_trial)])
            self.escape_data['Heading vs Barrier'] += escape_obj.h_vs_b_plot(0)
            br, bl = escape_obj.escapes_vs_barrierloc(1)
            self.escape_data['Barrier On Left Trajectories'] += bl
            self.escape_data['Barrier On Right Trajectories'] += br
            non_nan_cstarts = [cs for cs in escape_obj.cstart_rel_to_barrier
                               if not math.isnan(cs)]
            non_nan_cstart_rel_to_prevbout = [cs for cs in escape_obj.cstart_rel_to_last_bout
                                              if not math.isnan(cs)]
            self.escape_data['Correct CStart Percentage'].append(np.sum(
                non_nan_cstarts) / float(
                    len(non_nan_cstarts)))
            self.escape_data['CStart Rel to Prevbout'].append(np.sum(
                non_nan_cstart_rel_to_prevbout) / float(
                    len(non_nan_cstart_rel_to_prevbout)))
            self.escape_data[
                'CStart Latency'] += np.array(escape_obj.stim_init_times)[
                    escape_obj.stim_times_accurate].tolist()
            self.escape_data['CStart Angle'] += escape_obj.cstart_angles
            if len(escape_obj.numgrayframes) != 0:
                self.escape_data['Taps Per Entry Into Arena'].append(
                    len(escape_obj.xy_coords_by_trial) / float(
                        len(escape_obj.numgrayframes)))
                self.escape_data[
                    'Total Time In Center'] += escape_obj.numgrayframes
            non_nan_collisions = [col for col in escape_obj.collisions
                                  if not math.isnan(col)]
            self.escape_data['Collision Percentage'].append(
                np.sum(non_nan_collisions) / float(
                    len(non_nan_collisions)))

    def convert_to_nparrays(self):
        np_array_dict = {}
        for ky, it in self.escape_data.iteritems():
            np_array_dict[ky] = np.array(it)
        return np_array_dict


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
        if exp_type in ['l', 'd']:
            bstruct_and_br_label = 'b'
        elif exp_type in ['v', 'i']:
            bstruct_and_br_label = 'v'
        else:
            bstruct_and_br_label = exp_type
        self.barrier_file = np.loadtxt(
            directory + '/barrierstruct_' + bstruct_and_br_label + '.txt',
            dtype='string')
        if exp_type == 'n':
            self.numgrayframes = np.loadtxt(
                directory + '/numframesgray_' + exp_type + '.txt',
                dtype='string').astype(np.int)
        elif exp_type == 'l':
            self.numgrayframes = np.loadtxt(
                directory + '/numframesgray_' + bstruct_and_br_label + '.txt',
                dtype='string').astype(np.int)
        elif exp_type == 'd':
            self.numgrayframes = np.loadtxt(
                directory + '/numframesgray_dark.txt',
                dtype='string').astype(np.int)
        else:
            self.numgrayframes = []
        try:
            self.numgrayframes = self.numgrayframes.tolist()
        except AttributeError:
            pass
        self.barrier_coordinates = []
        self.barrier_diam = 0
        self.barrier_xy_by_trial = []
        print self.directory
        background_files = sorted([directory + '/' + f_id
                                   for f_id in os.listdir(directory)
                                   if f_id[0:10] == 'background'
                                   and f_id[-5:-4] == exp_type])
        back_color_files = [cv2.imread(fil) for fil in background_files]
        self.backgrounds = [cv2.cvtColor(back_color,
                                         cv2.COLOR_BGR2GRAY)
                            for back_color in back_color_files]
        pre_escape_files = sorted([directory + '/' + f_id
                                   for f_id in os.listdir(directory)
                                   if f_id[0:15] == 'fishcoords_gray'
                                   and f_id[-5:-4] == exp_type])
        self.pre_escape = [
            np.loadtxt(pe, dtype='string') for pe in pre_escape_files]
        self.pre_escape_bouts = []
        xy_files = sorted([directory + '/' + f_id
                           for f_id in os.listdir(directory)
                           if f_id[0:4] == 'tapr' and f_id[-5:-4] == exp_type])
        self.xy_escapes = [np.loadtxt(xyf, dtype='string') for xyf in xy_files]
        stim_files = sorted([directory + '/' + f_id
                             for f_id in os.listdir(directory)
                             if f_id[0:4] == 'stim'
                             and f_id[-5:-4] == exp_type])
        self.stim_data = [np.loadtxt(st, dtype='string') for st in stim_files]
        self.movie_id = sorted([directory + '/' + f_id
                                for f_id in os.listdir(directory)
                                if (f_id[-5:] == exp_type+'.AVI'
                                    and f_id[0] == 't')])

# Make these dictionaries so you can put in arbitrary trial / value bindings
        self.cstart_rel_to_last_bout = []
        self.cstart_angles = []
        self.cstart_rel_to_barrier = []
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
        
    def h_vs_b_plot(self, plot_fish):
        # h_vs_b_mod = [np.mod(
        #     hb_trial, 2*np.pi) for hb_trial in self.h_vs_b_by_trial]


# if hb_trial[0] is negative, meaning barrier is on left, multiply the trial by -1?
# currently the h_vs_b is on a -180 to 180 scale with 0 as barrier directly in front, negatives on the left.
# abs does do the trick of seeing if the absolute angle grows, but misses crossings (i.e. -60 to 80 would be the same as -60 to -80, which is a
# much lower amplitude turn. TRY:
        
        h_vs_b_mod = [np.array(hb_trial) if hb_trial[0] > 0 else -1*np.array(
            hb_trial) for hb_trial in self.h_vs_b_by_trial]
        
#        h_vs_b_mod = [np.abs(hb_trial) for hb_trial in self.h_vs_b_by_trial]
        if plot_fish:
            sb.tsplot(h_vs_b_mod)
            pl.show()
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
        print self.condition
        for filenum, xy_file in enumerate(self.xy_escapes):
            xcoords = []
            ycoords = []
            for coordstring in xy_file:
                x, y = x_and_y_coord(coordstring)
                xcoords.append(x)
                ycoords.append(y)
            print filenum
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
            light_profile = [np.sum(np.array(a) * np.arange(100)) for a in partition(100, stimdata)]
            std_light_profile = gaussian_filter([np.std(lp) for lp in sliding_window(5, light_profile)], 1)
            std_max = argrelmax(std_light_profile)[0]
            stim_init = [stdm for stdm, stdlp in zip(
                std_max, std_light_profile[std_max]) if stdlp > 10000]
            stim_init = [st for st in stim_init if (
                self.timerange[0] < st < self.timerange[1])]
            if plot_stim:
                pl.plot(std_light_profile)
                pl.show()
            if not stim_init:
                self.stim_init_times.append(self.pre_c)
                self.stim_times_accurate.append(0)
            else:
                self.stim_init_times.append(stim_init[0]-self.timerange[0])
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
        def collision(xb, yb, bd, x, y):
            proximity_val = 5
            vec = np.sqrt((x - xb)**2 + (y - yb)**2)
            if math.isnan(x):
                return False
            elif vec < (bd / 2) + proximity_val:
                return True
            else:
                return False
            
        for trialnum, xyc in enumerate(self.xy_coords_by_trial):
            if xyc[0] == []:
                self.collisions.append(np.nan)
                continue
            barrier_x = self.barrier_xy_by_trial[trialnum][0]
            barrier_y = self.barrier_xy_by_trial[trialnum][1]
            barrier_diameter = self.barrier_diam
            xcoords = self.xy_coords_by_trial[trialnum][0]
            ycoords = self.xy_coords_by_trial[trialnum][1]
            collision_coords = 0
            if not math.isnan(self.escape_latencies[trialnum]):
                t_init = self.timerange[0] + self.escape_latencies[
                    trialnum] + 5
            else:
                t_init = self.timerange[0] + self.pre_c + 5
            for fish_x, fish_y in zip(
                    xcoords[int(t_init):self.timerange[1]],
                    ycoords[int(t_init):self.timerange[1]]):
                if collision(barrier_x, barrier_y, self.barrier_diam,
                             fish_x, fish_y):
                    collision_coords += 1
            if collision_coords > 1:
                self.collisions.append(1)
            else:
                self.collisions.append(0)
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
                try:
                    background_roi = slice_background(
                        self.backgrounds[trial], x, y)
                except IndexError:
                    background_roi = slice_background(
                        self.backgrounds[-1], x, y)
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
            mag = math.sqrt(np.sum([i*i for i in vec]))
            self.initial_conditions.append([h_to_b, mag, ha_avg, ba_avg])

    def trial_analyzer(self, plotc):
        self.get_orientation(True)
        self.vec_to_barrier()
        self.heading_v_barrier()
        self.find_initial_conditions()
        self.body_curl()
        self.find_cstart(plotc)
        self.collision_course()


# this function splits up escapes by whether the barrier was to the
# right or left of the fish at the onset of the tap.
# you can chose to plot these trajectories for a single fish or
# wrap this function and plot many fish.

    def escapes_vs_barrierloc(self, *dontplot):
        if dontplot == ():
            turn_fig = pl.figure()
            turn_ax = turn_fig.add_subplot(111)
            turn_ax.set_xlim([-100, 100])
            turn_ax.set_ylim([-100, 100])
        timerange = self.timerange
        barrier_on_left = []
        barrier_on_right = []
        for trial, xy_coords in enumerate(self.xy_coords_by_trial):
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
                    if dontplot == ():
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
                    if dontplot == ():
                        turn_ax.plot(
                            x_escape,
                            y_escape, 'm')
                        turn_ax.text(
                            x_escape[-1],
                            y_escape[-1],
                            str(trial),
                            size=10,
                            backgroundcolor='w')
        if dontplot == ():
            pl.show()
        else:
            return barrier_on_left, barrier_on_right

# this function uses the thresholded image where contours were discovered
# and uses the contour found in get_orientation. the contour is rotated
# according to the fish's heading angle.
# points along the contour are used to find the cumulative angle of the
# tail and the index where the c-start begins.
        
    def find_cstart(self, plotornot):
        for trial, xyc in enumerate(self.xy_coords_by_trial):
            if xyc[0] == [] or not self.stim_times_accurate[trial]:
                self.cstart_rel_to_barrier.append(np.nan)
                self.cstart_rel_to_last_bout.append(np.nan)
                self.cstart_angles.append(np.nan)
                self.escape_latencies.append(np.nan)
                self.pre_escape_bouts.append(np.nan)
                continue
            tail_kern = Gaussian1DKernel(1)
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
                print bout_angles[-1]
            self.pre_escape_bouts.append(bout_angles)

            
            c_thresh = 30
            ta = convolve(self.tailangle_sums[trial], tail_kern)
            # avg_curl_init = np.nanmean(ta[0:self.pre_c])
            # ta = ta - avg_curl_init
            c_start_angle = float('nan')
            ta_min = argrelmin(ta)[0].tolist()
            ta_max = argrelmax(ta)[0].tolist()
            ta_maxandmin = [x for x in sorted(ta_min + ta_max) if (
                (x > stim_init) and abs(
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
                continue
            c_start_angle = ta[ta_maxandmin[0]]
            c_start_ind = ta_maxandmin[0]
            self.cstart_angles.append(c_start_angle)
            self.escape_latencies.append(c_start_ind - stim_init)
    # Get latency here based on stim index and c_start_index

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
        if threshval < 15:
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

        def body_points(seg1, seg2):
            right = [unpack[0] for unpack in seg1]
            left = [unpack[0] for unpack in seg2]
            bp = [[int(np.mean([a[0], b[0]])), int(np.mean([a[1], b[1]]))]
                  for a, b in zip(right, left[::-1])]
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
                im_color = threshvid.get_data(frame)
                im = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY)
                im_rot, m, c = rotate_image(im, ha_adj)
                im_rot_color = cv2.cvtColor(im_rot, cv2.COLOR_GRAY2RGB)
                
                # # refinding here is easier than storing the contour then rotating both the image and the contour
                body_unrotated = self.contours_by_trial[trial][frame]
                body = rotate_contour(im, ha_adj, body_unrotated)
                # if self.condition == 'd' and trial == 2 and math.isnan(ha_adj):
                #     print ha_adj
                #     print body
                #     print("ha messed")
                if body.shape[0] == 0:
                    all_angles.append([])
                    cstart_vid.write(im_rot_color)
                    continue
# now find the point on the contour that has the smallest y coord (i.e. is closest to the top). may be largest y coord?

                body_perimeter = cv2.arcLength(body, True)
                highest_pt = np.argmin([bp[0][1] for bp in body])
                body = np.concatenate([body[highest_pt:], body[0:highest_pt]])
                body_segment1 = []
                body_segment2 = []
                segment = 1.0
                numsegs = 18.0
                for i in range(len(body)):
                    if cv2.arcLength(body[0:i+1],
                                     False) > body_perimeter*(segment/numsegs):
                        if segment < (numsegs/2):
                            body_segment1.append(body[i])
                        elif segment > (numsegs/2):
                            body_segment2.append(body[i])
                        elif segment == (numsegs/2):
                            endpoint = body[i].tolist()                       
                        segment += 1
                avg_body_points = body_points(body_segment1, body_segment2)[1:]

# First point inside head is unreliable. take 1:
                for bp in avg_body_points:
                    cv2.ellipse(im_rot_color,
                                (bp[0], bp[1]),
                                (1, 1), 0, 0, 360, (255, 0, 255), -1)
                cv2.drawContours(im_rot_color, [body], -1, (0, 255, 0), 1)
                cstart_vid.write(im_rot_color)
                body_gen = toolz.itertoolz.sliding_window(2, avg_body_points)
                # body_point_diffs = [
                #     (0, 1)] + [
                #         (b[0]-a[0], b[1]-a[1]) for a, b in body_gen]

# got rid of the first angle so it is agnostic to the reference when the
# cstart is
# happening. cstart messes up the reference
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
#            print i
#            print vmag
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
    filt_sd = sd
    npvecs = np.array(vecs)
    filt_vecs = np.copy(npvecs)
    for i in range(npvecs[0].shape[0]):
#        filt_vecs[:, i] = gaussian_filter(npvecs[:, i], filt_sd)
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


def plot_all_results(cond_collector_list):
    cond_list = [c.condition for c in cond_collector_list]
    # if cond_list != cond_list_orig:
    #     raise Exception('cond_collector in wrong order')
    esc_assess_time = (cond_collector_list[
        0].timerange[1] - cond_collector_list[0].timerange[0]) / 2
    cpal = sb.color_palette()
    fig, axes = pl.subplots(1, 2)
    axes[0].set_title('Fish Orientation vs. Barrier (rad)')
    axes[1].set_title('Distance from Barrier at Escape Termination')
    fig2, axes2 = pl.subplots(1, len(cond_list), sharex=True, sharey=True)
    for i in range(len(cond_list)):
        axes2[i].set_xlim([-100, 100])
        axes2[i].set_ylim([-100, 100])
        axes2[i].vlines(0, -100, 100, colors=(.8, .8, .8), linestyles='dashed')
        axes2[i].set_aspect('equal')
        axes2[i].set_title(cond_list[i])
    cond_data_arrays = []
    for cond_ind, cond_data_as_list in enumerate(cond_collector_list):
        cond_data = cond_data_as_list.convert_to_nparrays()
        cond_data_arrays.append(cond_data)
        sb.tsplot(np.array(cond_data['Heading vs Barrier']),
                  ax=axes[0], estimator=np.nanmean, color=cpal[cond_ind])

        # here add a counter and ask whether r_coords[0] or l_coords[0] are pos or neg.
        correct_moves = 0
        incorrect_moves = 0
        for r_coords, l_coords in izip_longest(
                cond_data['Barrier On Left Trajectories'],
                cond_data['Barrier On Right Trajectories']):
                if r_coords is not None:
                    axes2[cond_ind].plot(r_coords[0], r_coords[1],
                                         color=np.array(cpal[cond_ind]) * .5)
                    if np.mean(r_coords[0]) < 0:
                        correct_moves += 1
                    else:
                        incorrect_moves += 1
                if l_coords is not None:
                    axes2[cond_ind].plot(l_coords[0], l_coords[1],
                                         color=cpal[cond_ind] * 1 / np.max(
                                             cpal[cond_ind]))
                    if np.mean(l_coords[0]) > 0:
                        correct_moves += 1
                    else:
                        incorrect_moves += 1

        axes2[cond_ind].text(
            -95, -80,
            str(correct_moves) + ' Correct', size=10)
        axes2[cond_ind].text(
            -95, -95,
            str(incorrect_moves) + ' Incorrect', size=10)

    pl.tight_layout()
    barfig, barax = pl.subplots(3, 3, figsize=(8, 6))
    barax[0, 0].set_title('% CStart Away from Barrier')
    barax[1, 0].set_title('# Collisions')
    barax[0, 1].set_title('CStart Latency (ms)')
    barax[0, 2].set_title('CStart Angle (deg)')
    barax[1, 1].set_title('Probability of Tap Per Arena Entry')
    barax[1, 2].set_title('Total Time Spent in Barrier Zone')
    barax[2, 0].set_title('Phototaxis to Tap Time')
    barax[2, 1].set_title('CStart Rel to Prevbout')
    sb.violinplot(data=[np.concatenate(bdist, axis=0) for bdist 
                        in [c['Distance From Barrier After Escape']
                        for c in cond_data_arrays]],  ax=axes[1])
    cstart_percentage_data = [cdir[~np.isnan(cdir)] for
                              cdir in [c['Correct CStart Percentage']
                              for c in cond_data_arrays]]
    cstart_rel_to_prevbout = [cdir[~np.isnan(cdir)] for
                              cdir in [c['CStart Rel to Prevbout']
                              for c in cond_data_arrays]]
    sb.barplot(data=cstart_percentage_data, 
               ax=barax[0, 0], estimator=np.nanmean)
    sb.barplot(data=cstart_rel_to_prevbout, ax=barax[2, 1], estimator=np.nanmean)
    collision_percentage_data = [clp[~np.isnan(clp)] for
                                 clp in [c['Collision Percentage']
                                         for c in cond_data_arrays]]
    sb.barplot(data=collision_percentage_data, 
               ax=barax[1, 0], estimator=np.nanmean)
    sb.boxplot(data=[2*clat[~np.isnan(clat)] for
                     clat in [c['CStart Latency']
                                 for c in cond_data_arrays]],
               ax=barax[0, 1])
    sb.boxplot(data=[cang[~np.isnan(cang)] for
                     cang in [c['CStart Angle']
                              for c in cond_data_arrays]],
               ax=barax[0, 2])
    taps_per_entry = [num_entries[~np.isnan(num_entries)] for
                      num_entries in [c['Taps Per Entry Into Arena']
                                      for c in cond_data_arrays]]
    sb.boxplot(data=taps_per_entry,
               ax=barax[1, 1])
    sb.boxplot(data=[dur[~np.isnan(dur)] for
                     dur in [c['Total Time In Center']
                             for c in cond_data_arrays]],
               ax=barax[1, 2])
    sb.boxplot(data=[ptax[~np.isnan(ptax)] for
                     ptax in [c['Phototaxis to Tap Time']
                              for c in cond_data_arrays]],
               ax=barax[2, 0])
    pl.tight_layout()
    pl.show()

    
def parse_obj_by_trial(drct_list, cond, mods):
    os.chdir('/Users/nightcrawler2/Escape-Analysis/')
    for drct in drct_list:
        fish_id = '/' + drct
        pl.ioff()
        area_thresh = 47
        esc_dir = os.getcwd() + fish_id
        print esc_dir
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


def experiment_collector(drct_list, cond_list, *new_exps):
    cond_collector_list = [Condition_Collector(cl) for cl in cond_list]
    if new_exps != ():
        new_exps = new_exps[0]
    os.chdir('/Users/nightcrawler2/Escape-Analysis/')
    for newexp_dirct in new_exps:
        fish_id = '/' + newexp_dirct
        pl.ioff()
        area_thresh = 47
        esc_dir = os.getcwd() + fish_id
        print esc_dir
        plotcstarts = False
        for cond in cond_list:
            try:
                escape_obj = Escapes(cond, esc_dir, area_thresh)
                escape_obj.trial_analyzer(plotcstarts)
                escape_obj.exporter()
            except IOError:
                print("No " + cond + " Trials in fish" + str(esc_dir))
# CATCH FOR SPLITTING ACCORDING TO TRIAL GOES HERE. 
            
    for drct in drct_list:

        for cond_ind, cond_collector in enumerate(cond_collector_list):
            try:
                print('loading cond ' + str(cond_ind+1) + ' trials')
                esc_obj = pickle.load(open(
                    drct + '/escapes_' + cond_list[cond_ind] + '.pkl', 'rb'))
                cond_collector.update_ddict(esc_obj)
            except IOError:
                pass
    return cond_collector_list


# SIMPLY CALL EXPERIMENT_COLLECTOR ON TWO SEPARATE DRCT_LISTS AND THEN + THEM, PUT INTO PLOT_ALL
    

if __name__ == '__main__':

    hd = experiment_collector(['030519_1', '030519_2',
                               '030719_1', '030719_2', '030719_3'], ['l', 'd'],
                              ['030519_1', '030519_2',
                               '030719_1', '030719_2', '030719_3'])

#     all_dist_list = ['043019_2', '050219_3', '050219_4']
# #    parse_obj_by_trial(all_dist_list, 'l', 3)
#     hd = experiment_collector(all_dist_list, ['l0', 'l1', 'l2'])

    # FOR 54 SMALL VS 108 BIG
    # have to write an experiment collector call for each list you want
    # i.e. ec1 = experiment_collector(drct_list1, conds)
    #       ec2 = experiment_collector(drct_list2, conds)
    #       hd_both = ec1 + ec2
    #       plot_all_results(ec1 + ec2)

    # FOR 54 vs 108 BIG
    # drct_list = [alldrcts]
    # parse_obj_by_trial(alldrcts, 'l', 2)
    # ec = experiment_collector(alldrcts, ['l0', 'l1'])
    plot_all_results(hd)

    


    

    
    # escape_cond2 = Escapes('d', esc_dir, area_thresh)
    # escape_cond2.trial_analyzer(plotcstarts)
    # escape_cond2.escapes_vs_barrierloc()
    # esc_dir = os.getcwd() + '/030719_1'
    # pl.ioff()
    # area_thresh = 47
    # plotcstarts = True
    # escape_cond1 = Escapes('l', esc_dir, area_thresh)
    # escape_cond1.trial_analyzer(plotcstarts)
    # escape_cond1.escapes_vs_barrierloc()

#    esc_l = pickle.load(open(drct + '/escapes_l.pkl', 'rb'))

# HOW MANY TIMES THEY VISIT THE TRIGGER ZONE PROVIDED THEYVE ENTERED THE FOREST
