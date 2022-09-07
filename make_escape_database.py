import os
import shutil as sh

# first copy complete folder.
# next delete everything that doesn't say ".txt".



virtual = ['091119_4', '091019_1', '091019_2', '091119_1', '091119_2',
           '091119_3', '091119_5', '091319_1', '091319_2', '091319_3',
           '091319_4', '092719_1', '092719_2', '092719_3', '092719_4',
           '100219_1', '100219_2', '100219_3', '100219_4', '100219_6',
           '100219_7', '100219_5']

four_w = ['072319_1', '072319_2', '072319_3', '072319_4',
          '072319_5', '072319_6', '072319_7', '072419_3',
          '072419_4', '072419_5', '072419_6', '072419_7',
          '072419_8', '072419_9', '072519_1', '072619_1',
          '072619_2', '072619_3', '072619_4']

# [0, 0, 0, 

four_b = ['022619_2', '030519_1', '030519_2', '030719_1',
          '030719_2', '030719_3', '032619_1', '032819_1',
          '032919_1', '040319_1', '040419_2', '040519_2',
          '041719_1', '041819_1', '041919_2', '042319_1',
          '042719_1', '102319_1', '102319_2', '102419_1',
          '102519_1', '110219_1', '110219_2']

wik_mauthner_l = ['052721_1', '060421_1',
                  '060421_3', '060421_4', '060421_5',
                  '021320_1', '021320_2', '021320_3',
                  '022120_1', '022120_2', '022120_3', '061021_1',
                  '061021_3', '061021_6']

wik_mauthner_r = ['052721_2', '052721_3', '052821_2', '052821_3',
                  '052821_4', '060321_1', '060321_2',
                  '060321_4', '060321_5', '060321_6', '060321_7',
                  '060421_8', '061021_4', '061021_5']


# 0 is l first, 1 is n first
# [1, 1, 1, 1, 1, 1, 1, 1,  

red24mm_4mmdist = ['061121_1', '061121_2', '061121_3', '061121_4',
                   '061121_5', '061121_6', '061121_7', '061421_1',
                   '061421_2', '061421_3', 
                   '061521_2', '061521_3', '061521_4', '061521_5',
                   '061521_6']


red12mm_4mmdist = ["061721_1", "061721_2", "061721_3", "061721_4", "061721_5",
                   "061721_6", "061721_7", "061821_1", "061821_2", "061821_3", 
                   "061821_4", "061821_5", "061821_6",  "062221_1", "062221_2",
                   "062221_3", "062221_4", "062221_5", "062221_6"]



red48mm_8mmdist = ["062521_3", "062521_4", "063021_2",
                   "063021_3", "063021_4", "063021_5",
                   "063021_6", "070121_1", "070121_2", 
                   "070221_4", "070221_5", "070221_6", "070621_4",
                   "070721_4", "070721_5", "070721_6", "070921_2",
                   "070921_4"]

# started getting rid of fish where the taps weren't happening here, but then
# wrote a filter for LED hits into the analysis.

red48mm_8mmdist_2h = [#"070821_1", "070821_2", "070821_8", "070821_9",
    #                  "070921_6",
                      "071221_1", #"071221_2",  "071221_9"

                      #note that 7/8 and 7/9 appear to have no taps! 1221_3 has taps and cstarts,                            # 1221_9 also has no taps. 
                      "071221_3",
                      "071221_5", "071221_6", "071221_7", "071221_8",
                      "071321_3", "071421_1", "071421_2",
                      "071421_3", "071421_4", "071421_5", "071421_7"]


red12mm_4mmdist_2h = ["072921_1", "072921_2", "072921_4", "073021_1",
                      "073021_2", "073021_3", "073021_4", "073021_5",
                      "073021_7", "073021_8", "073021_9", "080221_2",
                      "080221_3", "080221_4", "080221_5", "080221_6",
                      "080221_7", "080321_1", "080321_2", "080321_3"]



# 15 feels right. not really enough data in the 5 wins for conclusive results, and
# spreading to 15 puts enough data for clear error bars. if you bin by 5, the 15 degree
# window is already at 80% in four_b. 10 is definitely too close.

# TO DO NEXT: make sure your h to b plots are right. they are so consistent that they must be
# revealing something fundamental. good work today though!


# h_vs_b_plot is completely correct as long as the angle calculations are good. bout that lead into the
# barrier zone is calculated by taking the gray coords right before barrierzone entry.


src_path = r"/Volumes/Esc_and_2P/Escape_Results/"
folders_in_src = os.listdir(src_path)
dst_path = os.getcwd() + "/data_repository/"
mauthners = [wik_mauthner_l, wik_mauthner_r]
white_and_virtual_list = [four_w, virtual]
red_b_list = [four_b, red24mm_4mmdist, red12mm_4mmdist_2h, red12mm_4mmdist, red48mm_8mmdist_2h, red48mm_8mmdist]
all_fish = []
for fishdir in white_and_virtual_list + red_b_list + mauthners:
    for f in fishdir:
        all_fish.append(f)

for pth in all_fish:
    os.mkdir(dst_path+pth)
    try:
        files_in_fish = os.listdir(src_path+pth)
        for f in files_in_fish:
            if f[-4:] == ".pkl" or f[-4:] == '.txt':
                sh.copyfile(src_path+pth+'/'+f, dst_path+pth+'/'+f)
    except FileNotFoundError:
        print(pth)
        print("not found")


