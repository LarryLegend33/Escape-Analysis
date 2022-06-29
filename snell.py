import numpy as np
import seaborn as sns
from matplotlib import pyplot as pl


def snell_final(barrier_height, barrier_distance, water_depth):
    water_refractive_index = 1.333
    theta_prime = np.arctan(barrier_distance / barrier_height)
    theta_range = np.arange(0, np.pi / 2, .01)
    da_over_dw = (barrier_height - water_depth) / water_depth
    candidate_theta_primes = []
    for theta in theta_range:
        # Snells Law
        psi_a = np.arcsin(water_refractive_index * np.sin(theta))
        tprime_cand = np.arctan(
            (np.tan(theta) + da_over_dw * np.tan(psi_a)) / (1 + da_over_dw))
        candidate_theta_primes.append(tprime_cand)
    diffs = [np.abs(t-theta_prime) for t in candidate_theta_primes]
    theta_perceived = theta_range[np.nanargmin(diffs)]
    if np.degrees(theta_perceived) > 48.7:
        print("Caught by Snell Bro")
        perceived_barrier_height = np.nan
    else:
        perceived_barrier_height = barrier_distance / np.tan(theta_perceived)
        print(perceived_barrier_height)
    return perceived_barrier_height


def calculate_perceived_height(barrier_height, barrier_distance, water_depth):
    a_range = np.arange(0, barrier_distance, .001)
    water_refractive_index = 1.33
    underwater_triangle = []
    overwater_triangle = []
    for a in a_range:
        underwater_triangle.append(
            water_refractive_index * np.sin(np.arctan(a / water_depth)))
        overwater_triangle.append(
            np.sin(
                np.arctan(
                    (barrier_distance - a) / (barrier_height - water_depth))))
    diffs = [np.abs(b-a) for a, b in zip(overwater_triangle, underwater_triangle)]
    min_a = a_range[np.argmin(diffs)]
    pl.plot(a_range, underwater_triangle, color='b')
    pl.plot(a_range, overwater_triangle, color='m')
    theta_2 = np.arctan(min_a / water_depth)
    theta_1 = np.arctan((barrier_distance - min_a) / (barrier_height - water_depth))
    perceived_height = barrier_distance / np.tan(theta_2)
    print("Location of Ray Intersection With Water")
    print(min_a)
    print("Perceived Height of Barrier")
    print(perceived_height)
#    print(theta_2)
#    print(theta_1)
    pl.show()
    return min_a, perceived_height

def angle_ratios(theta1):
    water_refractive_index = 1.33
    theta2 = np.arcsin(np.sin(theta1) / water_refractive_index)
    print (theta1 / theta2)


def snells_law_2(barrier_height, barrier_distance, water_depth):
    water_refractive_index = 1.33
    loc_of_unbent_ray_entry = water_depth * barrier_distance / barrier_height
    theta_unbent_ray = np.arctan(loc_of_unbent_ray_entry / water_depth)
    # this ray WILL be bent. it will hit the ground at a distance C from the fish
    theta_bent = np.arcsin(
        np.sin(
            np.arctan(barrier_distance / barrier_height)) / water_refractive_index)
    # so far everything here is predetermined by the known variables.
    c = water_depth * np.tan(theta_bent)
    distance_ray_hits_ground_from_fish = loc_of_unbent_ray_entry - c
    print("Distance Unbent Ray Hits From Fish")
    print(loc_of_unbent_ray_entry)
    print("Distance Ray Hits Ground From Fish")
    print(distance_ray_hits_ground_from_fish)
    # pretty sure this is correct. should be impossible if the ratio is greater than 1.33, because you can't grow any more. (i.e ratio is stuck at 1.33 min)
    # test this with angle ratios. 
    # if distance_ray_hits_ground_from_fish > loc_of_unbent_ray_entry / water_refractive_index:
    theta_required_for_visibility = np.arctan((loc_of_unbent_ray_entry - distance_ray_hits_ground_from_fish) / water_depth)
    print("Theta Ratio")
    print(theta_unbent_ray / theta_required_for_visibility)
    print("Theta Required")
    print(theta_required_for_visibility)
#    if distance_ray_hits_ground_from_fish > loc_of_unbent_ray_entry / water_refractive_index:
    if theta_ratio > 1.83:
        new_barrier_height = barrier_height - .1
        if new_barrier_height < water_depth:
            print("Can't see any nonsubmerged portion")
        else:
            print("Ray Caught By Snell Bro")
            return snells_law_2(barrier_height - .1, barrier_distance, water_depth)
    else:
        print("Perceived Barrier Height at Groundpoint Intersection")
        perceived_barrier_height_at_intersection = (barrier_distance - distance_ray_hits_ground_from_fish) / np.tan(theta_bent)
        print(perceived_barrier_height_at_intersection)
        print("Perceived Barrier Height at Fish")
        perceived_barrier_height_at_fish = (barrier_distance + distance_ray_hits_ground_from_fish) / np.tan(theta_unbent_ray)
        print(perceived_barrier_height_at_fish)
        return distance_ray_hits_ground_from_fish



