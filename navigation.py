import csv
import os
import numpy as np
import math
from matplotlib import pyplot as pl
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.image import AxesImage
from matplotlib.colors import Colormap
import seaborn as sb
import cv2
import toolz
import scipy.ndimage
import pickle
from toolz.itertoolz import sliding_window, partition
from scipy.ndimage import gaussian_filter


def plot_xy_experiment(b_list, b_diams, exp_type, drc):
    fig = pl.figure()
    axes = fig.add_subplot(111, axisbg='.75')
    axes.grid(False)
    for br, bd in zip(b_list, b_diams):
        barrier_x = br[0]
        barrier_y = br[1]
        barrier_diameter = bd
        barrier_plot = pl.Circle((barrier_x, barrier_y),
                                 barrier_diameter / 2, fc='r')
        axes.add_artist(barrier_plot)
    xcoords, ycoords = get_xy(exp_type, drc)
    v_from_center = []
    for crd in zip(xcoords, ycoords):
        vector = np.array(crd)
        center_mag = magvector_center(vector)
        v_from_center.append(center_mag)
    delta_center_mag = [b-a for a, b in sliding_window(2, v_from_center)]
    axes.plot(xcoords, ycoords)
    axes.axis('equal')
    pl.show()
    return xcoords, ycoords, v_from_center, delta_center_mag

    
def get_xy(exp_type, drc):
    xy_file = np.loadtxt(drc + '/all_xycoords_' + exp_type + '.txt',
                         dtype='string')
    xcoords = []
    ycoords = []
    for coordstring in xy_file:
        x, y = x_and_y_coord(coordstring)
        xcoords.append(x)
        ycoords.append(y)
    return xcoords, ycoords


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



def load_barrier_info(exp_type, drc):
    barrier_file = np.loadtxt(
            drc + '/barrierstruct_' + exp_type + '.txt',
            dtype='string')
    barrier_coords = []
    diam_list = []
    for line, j in enumerate(barrier_file[2:]):
        if line % 2 == 0:
            barrier_coords.append(x_and_y_coord(j))
        else:
            diam_list.append(float(j))
    return barrier_coords, diam_list

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
    
                                     
    
def inbound_outbound(xcoords, ycoords, delta_mag, bloc, bdiam):

    # nans here are being detected as changes
    fig = pl.figure()
    ax = fig.add_subplot(111)
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
                ax.plot(xcoords[switch_inds[0]:switch_inds[1]],
                        ycoords[switch_inds[0]:switch_inds[1]],
                        color='k', linewidth=.5)
                outbound_swims.append(switch_inds)
            else:
                ax.plot(xcoords[switch_inds[0]:switch_inds[1]],
                        ycoords[switch_inds[0]:switch_inds[1]],
                        color='m', linewidth=.5)
                inbound_swims.append(switch_inds)
            switch = not switch
            switch_origin = ind
            
    for b_ind, (br, bd) in enumerate(zip(bloc, bdiam)):
        barrier_x = br[0]
        barrier_y = br[1]
        barrier_diameter = bd
        barrier_plot = pl.Circle((barrier_x, barrier_y),
                                 barrier_diameter / 2, fc='k')
        ax.text(barrier_x, barrier_y, str(b_ind), color='w')
        ax.add_artist(barrier_plot)
    ax.axis('equal')
    pl.show()
    return inbound_swims, outbound_swims


def distance_from_center(xc, yc):
    all_mags = []
    for x, y in zip(xc, yc):
        mag_from_center = magvector_center([x, y])
        if not math.isnan(mag_from_center):
            all_mags.append(mag_from_center)
    sb.distplot(np.array(all_mags), bins=50)
    pl.show()
    



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
    
exp_type = 'l'
directory = os.getcwd() + '/Fish1'
barrier_loc, barrier_diams = load_barrier_info(exp_type, directory)

# this function is going to have to take the boundaries of the light / dark switch as an arg
# so that the correct x and y coords are taken from the experiment. 

xcoords, ycoords, vmag, delta_mag = plot_xy_experiment(barrier_loc,
                                                       barrier_diams,
                                                       exp_type,
                                                       directory)
distance_from_center(xcoords, ycoords)
inbound, outbound = inbound_outbound(
    xcoords, ycoords, delta_mag, barrier_loc, barrier_diams)

nearest_b, lines = barrier_center(barrier_loc)
line_functions = fit_barrierline(lines, barrier_loc)
xpaths, ypaths = xy_paths(xcoords, ycoords, outbound)
lr = l_or_r(barrier_loc, lines)
#returns pairwise readouts for which barrier is right or left of other barriers. 
crossings = crosscoords(xpaths, ypaths, line_functions)
#r = np.array(
 #   midpoint_proximity(crossings, barrier_loc, barrier_diams, lines))
 
#multiple crossings get all F'ed up with inbound I think.
#pl.hist(r, bins=50, color='r')

# so rat contains the ratios of left to right barriers, normalized so that 0 is the leftmost, and 2 is the rightmost possible.
# each entry contains a second variable that shows which barrier is on the left and which is on the right. filter accordingly when
# you start using new types of barriers.

rat = barrier_ratios(crossings, barrier_loc, lines, lr)
sb.distplot([r[0] for r in rat], bins=50)
pl.show()
