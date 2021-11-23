import numpy as np
import math
import seaborn as sb
import cv2
import toolz
import scipy.ndimage
import pickle
from toolz.itertoolz import sliding_window, partition
from scipy.ndimage import gaussian_filter, uniform_filter
import csv
import os
from collections import Counter
from matplotlib import pyplot as pl
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.image import AxesImage
from matplotlib.colors import Colormap, Normalize



def proximity_calculator(nav_list, condition, b_color, *value_range):
    coords_wrt_barrier = []
    nav_object_collection = []
    for nl_item in nav_list:
        nav_directory = '/Volumes/Esc_and_2P/Escape_Results/' + nl_item
        nav_object = Navigator(condition, nav_directory)
        nav_object.norm_coords_to_barrier()
        coords_wrt_barrier += nav_object.coords_wrt_closest_barrier
        nav_object_collection.append(nav_object)



    coords_wrt_barrier = np.array(coords_wrt_barrier)
    xmax = np.nanmax(coords_wrt_barrier[:, 0])
    xmin = np.nanmin(coords_wrt_barrier[:, 0])
    ymax = np.nanmax(coords_wrt_barrier[:, 1])
    ymin = np.nanmin(coords_wrt_barrier[:, 1])

    scale_factor = 300
    barrier_location = np.array([-xmin, -ymin])
    
    # xmax = scale_factor
    # xmin = -1*scale_factor
    # ymax = scale_factor
    # ymin = -1*scale_factor
    
    proximity_matrix = np.zeros([int(ymax-ymin+1), int(xmax-xmin+1)])
    for coord in coords_wrt_barrier:
        if np.isfinite(coord).all():
            if magvector(coord) < scale_factor:
                proximity_matrix[int(coord[1] - ymin), int(coord[0] - xmin)] += 1

    fig = pl.figure()
    ax = fig.add_subplot(111)
    # was at 2 for all original figures
    filt_proxmat = gaussian_filter(proximity_matrix, 5)
 #   filt_proxmat = proximity_matrix

    if value_range != ():
        min_proxmat, max_proxmat = value_range[0]
    else:
        min_proxmat = np.min(filt_proxmat[filt_proxmat > 0])
        max_proxmat = np.max(filt_proxmat)
 
    sb.heatmap(filt_proxmat, center=3* (max_proxmat - min_proxmat)/ 4, cmap='hot')
#    sb.heatmap(filt_proxmat, center=(max_proxmat - min_proxmat) / 2, cmap='icefire')
#    sb.heatmap(proximity_matrix)
    bounds = pl.Circle(barrier_location,
                       scale_factor, ec='k', fc='None')
    barrier = pl.Circle(barrier_location,
                        nav_object.barrier_diams[0] / 2, ec='None', fc=b_color)
    ax.add_artist(bounds)
    ax.add_artist(barrier)
    pl.show()
    return nav_object_collection, [min_proxmat, max_proxmat]

def proximity_histogram(nav_list, condition, b_color, *norm_max):
    coords_wrt_barrier = []
    nav_object_collection = []
    bound = 200
    barrier_location = [0, 0]
    n_bins = 30
    barrier_prox_stats = []
    for nl_item in nav_list:
        nav_directory = '/Volumes/Esc_and_2P/Escape_Results/' + nl_item
        nav_object = Navigator(condition, nav_directory)
        nav_object.norm_coords_to_barrier()
        coords_wrt_barrier += nav_object.coords_wrt_closest_barrier
        nav_object_collection.append(nav_object)
        density = np.histogram2d(np.array(nav_object.coords_wrt_closest_barrier)[:, 0],
                                 np.array(nav_object.coords_wrt_closest_barrier)[:, 1],
                                 range=[[-bound, bound], [-bound, bound]])[0]
        dim = density.shape[0]
        barrier_prox_stat = np.sum(density[int(dim/2) - int(dim / 5):int(dim/2) + int(dim/5), int(dim/2) - int(dim / 5):
                                       int(dim/2) + int(dim/5)]) / np.sum(density)
        barrier_prox_stats.append(barrier_prox_stat)

    # here do the proximity calculation below. do it on a binsize basis per fish.
    # make a bar graph for it next to the plots
        
    coords_wrt_barrier = np.array(coords_wrt_barrier)
    fig = pl.figure()
    ax = fig.add_subplot(111)
    cmap = 'inferno'

    if norm_max != ():
        norm=colors.Normalize(0, norm_max[0])
        hm = ax.hist2d(coords_wrt_barrier[:, 0], coords_wrt_barrier[:, 1], range=[[-bound, bound], [-bound, bound]],
                       bins=n_bins, cmap=cmap, density=False, norm=norm)
    else:
        hm = ax.hist2d(coords_wrt_barrier[:, 0], coords_wrt_barrier[:, 1], range=[[-bound, bound], [-bound, bound]],
                       bins=n_bins, cmap=cmap, density=False)
    

        
#    sb.heatmap(filt_proxmat, center=3* (max_proxmat - min_proxmat)/ 4, cmap='hot')
    barrier = pl.Circle(barrier_location,
                        nav_object.barrier_diams[0] / 2, ec='None', fc=b_color)
 #   ax.add_artist(bounds)
    ax.add_artist(barrier)
    ax.set_aspect('equal')
    fig.colorbar(hm[3], ax=ax)
    pl.show()
    return nav_object_collection, hm[0], barrier_prox_stats


class Navigator:
    def __init__(self, condition, drc):
        self.barrier_coords = []
        self.barrier_diams = []
        self.condition = condition
        self.drc = drc
        self.xy_coords = []
        self.mags_from_center = []
        self.coords_wrt_closest_barrier = []
        self.get_xy()
        self.load_barrier_info()
        self.inbound_swims = []
        self.outbound_swims = []

    def load_barrier_info(self):
        self.barrier_coords = []
        self.barrier_diams = []
        barrier_file = np.loadtxt(
                self.drc + '/barrierstruct_' + exp_type + '.txt',
                dtype='str')
        for line, j in enumerate(barrier_file[2:]):
            if line % 2 == 0:
                self.barrier_coords.append(x_and_y_coord(j))
            else:
                self.barrier_diams.append(float(j))

    def get_xy(self):
        self.xy_coords = []
        xy_file = np.loadtxt(self.drc + '/all_xycoords_' + exp_type + '.txt',
                             dtype='str')
        xcoords = []
        ycoords = []
        for coordstring in xy_file:
            x, y = x_and_y_coord(coordstring)
            xcoords.append(x)
            ycoords.append(y)
        self.xy_coords = np.array([z for z in zip(xcoords, ycoords)])
        print(self.xy_coords)
        v_from_center = []
        for crd in self.xy_coords:
            vector = np.array(crd)
            center_mag = magvector_center(vector)
            v_from_center.append(center_mag)
        self.mags_from_center = v_from_center

    def norm_coords_to_barrier(self):
        self.coords_wrt_closest_barrier = []
        for coord in self.xy_coords:
            vec_to_barrier = [coord - bc for bc in self.barrier_coords]
            vec_mags = [magvector(v) for v in vec_to_barrier]
            self.coords_wrt_closest_barrier.append(
                vec_to_barrier[np.argmin(vec_mags)])
            
    def plot_xy_experiment(self, facecolors):
        fig, ax = pl.subplots(1, 1)
        ax.set_facecolor('.7')
        ax.grid(False)
        ax.plot(self.xy_coords[:, 0],
                self.xy_coords[:, 1],
                color='k', linewidth=.8)
        for br, bd, f in zip(self.barrier_coords, self.barrier_diams, facecolors):
            barrier_x = br[0]
            barrier_y = br[1]
            barrier_diameter = bd
            barrier_plot = pl.Circle((barrier_x, barrier_y),
                                     barrier_diameter / 2, fc=f, ec=f)
            ax.add_artist(barrier_plot)

        ax.axis('equal')
        pl.show()

    def distance_from_center(self):
        all_mags = []
        for xy in self.xy_coords:
            mag_from_center = magvector_center([xy[0], xy[1]])
            if not math.isnan(mag_from_center):
                all_mags.append(mag_from_center)
        sb.distplot(np.array(all_mags), bins=50)
        pl.show()
        

    def get_crossing_profile(self):
        self.inbound_outbound()
        nearest_b, lines = barrier_center(self.barrier_coords)
        line_functions = fit_barrierline(lines, self.barrier_coords)
        xpaths, ypaths = xy_paths(self.xy_coords[:, 0],
                                  self.xy_coords[:, 1], self.outbound_swims)
        lr = l_or_r(self.barrier_coords, lines)
        # #returns pairwise readouts for which barrier is right or left of other barriers. 
        crossings = crosscoords(xpaths, ypaths, line_functions)
        mid_prox = np.array(midpoint_proximity(crossings, self.barrier_coords,
                                               self.barrier_diams, lines))
        pl.hist(mid_prox, bins=50, color='r')
        rat = barrier_ratios(crossings, self.barrier_coords, lines, lr)
        sb.distplot([r[0] for r in rat], bins=50)
        pl.show()

# # so rat contains the ratios of left to right barriers, normalized so that 0 is the leftmost, and 2 is the rightmost possible.
# # each entry contains a second variable that shows which barrier is on the left and which is on the right. filter accordingly when
# # you start using new types of barriers.

        
        

    def inbound_outbound(self):
        fig = pl.figure()
        ax = fig.add_subplot(111)
        delta_mag = np.diff(self.mags_from_center)
        filt_mag = gaussian_filter(delta_mag, 10)
        sign_switch = np.diff(np.sign(filt_mag)) != 0
        outbound_swims = []
        inbound_swims = []
        if filt_mag[0] < 0:
            switch = True
        else:
            switch = False
        switch_origin = 0
        for ind, delta in enumerate(sign_switch):
            if delta:
                if math.isnan(filt_mag[ind]):
                    continue
                switch_inds = [switch_origin, ind]
                if switch:
                    ax.plot(
                        self.xy_coords[:, 0][switch_inds[0]:switch_inds[1]],
                        self.xy_coords[:, 1][switch_inds[0]:switch_inds[1]],
                        color='k', linewidth=.5)
                    outbound_swims.append(switch_inds)
                else:
                    ax.plot(
                        self.xy_coords[:, 0][switch_inds[0]:switch_inds[1]],
                        self.xy_coords[:, 1][switch_inds[0]:switch_inds[1]],
                        color='m', linewidth=.5)
                    inbound_swims.append(switch_inds)
                switch = not switch
                switch_origin = ind

        for b_ind, (br, bd) in enumerate(zip(self.barrier_coords,
                                             self.barrier_diams)):
            barrier_x = br[0]
            barrier_y = br[1]
            barrier_diameter = bd
            barrier_plot = pl.Circle((barrier_x, barrier_y),
                                     barrier_diameter / 2, fc='k')
            ax.text(barrier_x, barrier_y, str(b_ind), color='w')
            ax.add_artist(barrier_plot)
        ax.axis('equal')
        pl.show()
        self.inbound_swims = inbound_swims
        self.outbound_swims = outbound_swims
    
    

def magvector(vec):
    mag = np.sqrt(np.dot(vec, vec))
    return mag


def outlier_filter(xcoords, ycoords):
    new_x = [xcoords[0]]
    new_y = [ycoords[0]]
    for i, crds in enumerate(zip(xcoords[1:], ycoords[1:])):
        diff_vec = [crds[0] - new_x[-1], crds[1] - new_y[-1]]
        vmag = magvector(diff_vec)
        if i == len(xcoords) - 1:
            return new_x, new_y
        elif vmag < 100:
            new_x.append(crds[0])
            new_y.append(crds[1])
        else:
            new_x.append(new_x[-1])
            new_y.append(new_y[-1])
            xcoords = new_x + xcoords[i+2:]
            ycoords = new_y + ycoords[i+2:]
            try:
                return outlier_filter(xcoords, ycoords)
            except RuntimeError:
                return [], []
    return new_x, new_y


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
    x, y = float(xcoord), 1024 - float(ycoord)
    if magvector_center([x, y]) < 350:
        return x, y
    else:
        return np.nan, np.nan


def magvector_center(vec):
    dist_vec = [vec[0] - 640, vec[1] - 512]
    mag = np.sqrt(np.dot(dist_vec, dist_vec))
    return mag


def magvector_diff(vec1, vec2):
    dist_vec = [vec1[0] - vec2[0], vec1[1] - vec2[1]]
    mag = np.sqrt(np.dot(dist_vec, dist_vec))
    return mag


def barrier_center(bloc):
    two_nearest_barriers = []
    for b in bloc:
        temp_distance = []
        for ob in bloc:
            magvec = magvector_diff(ob, b)
            temp_distance.append(magvec)
        dist_argsort = np.argsort(temp_distance)
        two_nearest_barriers.append([dist_argsort[1], dist_argsort[2]])

    # in addition to two nearest, want to pull out the unique lines you have to construct. do this using the
    # two nearest barrier list.
    lines_to_make = []
    for barrier, nb in enumerate(two_nearest_barriers):
        b_cand1 = [barrier, nb[0]]
        b_cand2 = [barrier, nb[1]]
        if (b_cand1 not in lines_to_make) and (
                b_cand1[::-1] not in lines_to_make):
            lines_to_make.append(b_cand1)
        if (b_cand2 not in lines_to_make) and (
                b_cand2[::-1] not in lines_to_make):
            lines_to_make.append(b_cand2)
    return two_nearest_barriers, lines_to_make

        
def fit_barrierline(linelist, bloc):
    line_functions = []
    for line in linelist:
        point1 = bloc[line[0]]
        point2 = bloc[line[1]]
        slope = float(point2[1]-point1[1]) / (point2[0]-point1[0])
        yint = point2[1] - point2[0]*slope
#        linefunc = lambda (x): x * slope + yint
        linefunc = np.poly1d([slope, yint])
        line_functions.append(linefunc)
    return line_functions
        
# polynomial is a function that transforms x vals into a y val. will input
# x value of coordinates, return a y value, and ask if that's above or below your points yval.
# want this function to return a barrier # length list of 5 poly1d lines. 
    


# Have to first calculate line equations for each barrier pair. Next you take a trajectory from
# xycoords classified into sections of inward and outward swimming. For each section,
# ask whether there is a sign change where the coord is above or below each line. It will only cross one.
# Make sure at that point that the distance to the relevant barriers is below some threshold. 
    
                                     
    






def xy_paths(xc, yc, swim_windows):
    x_paths = []
    y_paths = []
    # this takes a window where the swim path is continually away or towards the center for 1 second or more
    thresh_windows = [win for win in swim_windows if win[1] - win[0] > 50]
    for tw in thresh_windows:
        xpath = xc[tw[0]:tw[1]]
        ypath = yc[tw[0]:tw[1]]
        mags = np.array([magvector_center([x, y]) for x, y in zip(xpath, ypath)])
        if (mags < 200).any():
            x_paths.append(xpath)
            y_paths.append(ypath)
    
    return x_paths, y_paths


def plotpaths(xp, yp):
    fig = pl.figure()
    ax = fig.add_subplot(111)
    for p_ind, path in enumerate(zip(xp, yp)):
        ax.plot(path[0], path[1], linewidth=.5)
        ax.text(path[0][-1], path[1][-1], str(p_ind))
    ax.axis('equal')
    pl.show()


def crosscoords(xpaths, ypaths, polyfunc):
    crosspoints = []
    for path_ind, (xpath, ypath) in enumerate(zip(xpaths, ypaths)):
        for line_id, func in enumerate(polyfunc):
            yline = map(func, xpath)
            comp = np.where(
                np.diff(
                    np.sign([y-yl for y, yl in zip(ypath, yline)])) != 0)[0]
            if comp.any():
                if len(crosspoints) == 0:
                    crosspoints.append([path_ind,
                                        line_id,
                                        xpath[comp[0]], ypath[comp[0]]])
                else:
                    if crosspoints[-1][0] != path_ind:
                        crosspoints.append([path_ind,
                                            line_id,
                                            xpath[comp[0]], ypath[comp[0]]])
                    elif crosspoints[-1][0] == path_ind:

                        if magvector_center(
                                crosspoints[-1][2:]) < magvector_center(
                                    [xpath[comp[0]], ypath[comp[0]]]):
                            continue
                        else:
                            crosspoints[-1] = [path_ind,
                                               line_id,
                                               xpath[comp[0]], ypath[comp[0]]]
                    
    return crosspoints
            
            
            
# want to apply each of the 5 functions in polyfunc to all x coords in xpath. you will get a xpath length vector
# of y coordinates that are on the line between two bariers. you then ask whether there is a sign change in the
#        ypath coords minus the line ycoords. 


# want this function to be specific to the barrier pair, using a negative index if you're closer to the left barrier, pos if right, 0 if centered. 

# 0 means first entry is on the left, second on the right
# think about this algorithm! trickier than you think.
# i think whether the y coord is above or below 512 matters. if you are below 512 and your xcoord is greater, you are on the left. 
def l_or_r(bloc, lines):
    lr = []
    for line in lines:
        b0 = bloc[line[0]]
        b1 = bloc[line[1]]
        if b0[1] > 512:
            if b0[0] < b1[0]:
                lr.append(0)
            else:
                lr.append(1)
        else:
            if b0[0] < b1[0]:
                lr.append(1)
            else:
                lr.append(0)
    return lr


def barrier_ratios(crossings, bloc, lines, lr):
    #ratios will contain the ratio and the barrier pair ordered by l first, r second
    ratios = []
    for cross in crossings:
        line_crossed = cross[1]
        barriers = lines[line_crossed]
        dist_to_barrier1 = magvector_diff(cross[2:], bloc[barriers[0]])
        dist_to_barrier2 = magvector_diff(cross[2:], bloc[barriers[1]])
        if lr[line_crossed] == 0:
            dist_rat = dist_to_barrier1 / dist_to_barrier2
        elif lr[line_crossed] == 1:
            dist_rat = dist_to_barrier2 / dist_to_barrier1
            barriers = barriers[::-1]
        if dist_rat > 1:
            dist_rat = 1 + (1 - 1 / dist_rat)
        ratios.append(
            [dist_rat, barriers])

    return ratios
        

def midpoint_proximity(crossings, bloc, bdiams, lines):
    line_midpoints = []
    max_possible_distance_from_midpoint = []
    barr_diam = np.median(bdiams)

    # DONT NEED A MIDPOINT. JUST NEED THE RATIO OF THE DISTANCE FROM THE TWO BARRIER CENTERS. 
    for b_pairs in lines:
        pt1 = np.array(bloc[b_pairs[0]])
        pt2 = np.array(bloc[b_pairs[1]])
        midpoint = (pt1 + pt2) / 2
        line_midpoints.append(midpoint)
    # max possible is the vector from midpoint to bloc minus bdiam/2
        maxdist = magvector_diff(pt1, midpoint) - barr_diam / 2.0
        max_possible_distance_from_midpoint.append(maxdist)

    crossratios = []
    for cross in crossings:
        line_number = cross[1]
        midpnt = line_midpoints[line_number]
        maxd = max_possible_distance_from_midpoint[line_number]
        dist_from_mp = magvector_diff(cross[2:], midpnt)
        ratio = dist_from_mp / maxd
        crossratios.append(ratio)
    return crossratios


exp_type = 'b'

red_b = ['061419_1', '061419_2', '061419_3',
         '061419_4', '061419_5', '061819_1']

white_b = ['061319_4', '061319_5', '061319_6',
           '061319_7', '061319_8', '061319_9']

red_2xheight_4xwide = ["072221_2", "072221_3", 
                       "072321_3", "072621_1", "072721_1",
                       "072721_3", "072721_4", 
                       "072821_1", "072821_2"]

whiteandred_b = ["061119_1", "061119_2", "061119_3",
                 "061119_4", "061119_5", "061219_1"]

blackandred_b = ["061219_2", "061219_3", "061219_4",
                 "061319_1", "061319_2", "061319_3"]

#navs_white = proximity_calculator(white_b, exp_type, [1, 1, 1])
#navs_red = proximity_calculator(red_b, exp_type, [1, 0, 0], navs_white[1])


navs_white, density_w, bprox_w = proximity_histogram(white_b, exp_type, [1, 1, 1])
navs_red, density_r, bprox_r = proximity_histogram(red_b, exp_type, [1, 0, 0], np.max(density_w))
xs = np.zeros(len(bprox_w)).tolist() + np.ones(len(bprox_r)).tolist()
sb.barplot(xs, bprox_w + bprox_r, color='gray')
ttest_results = scipy.stats.ttest_ind(bprox_w, bprox_r)

#navs_big = proximity_calculator(red_2xheight_4xwide, exp_type, [1, 0, 0])  #, navs_white[1])

#navs_whiteandred = proximity_calculator(whiteandred_b, exp_type, [1, 1, 1], navs_white[1])

#navs_blackandred = proximity_calculator(blackandred_b, exp_type, [0, 0, 0], navs_white[1])

#navtest = proximity_calculator(["061219_2"], exp_type, [0, 0, 0], navs_white[1])

# note that each of the navs above contains a list of Navigator objects as the first index
# you can call the plot_xy_experiment method on each Navigator object to see the trajectory. 

# barrier_loc, barrier_diams = load_barrier_info(exp_type, directory)

# # this function is going to have to take the boundaries of the light / dark switch as an arg
# # so that the correct x and y coords are taken from the experiment. 








    



