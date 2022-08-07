from STM_Python_Modules.Model_Resource.KeypointDetection_Module import KeypointDetection

'''
img_array = []  # small scale 2D image from scan, val range in (0,255)
x = []  # global x of small scale img (center) in nm
y = []  # global y of small scale img (center) in nm
'''

def KpDet(img_array, position):
    scan_size = [4.0, 4.0]    # scan size of the small scale img [W,H] in nm

    KpDet_model = KeypointDetection(img_array, position, scan_size)

    Keypoints_List = KpDet_model.prediction()

    return Keypoints_List
''' Keypoints_List: A list of [boolean, x, y], 
                    where boolean indicates whether it is target module,
                    x,y is global position of the reaction site in meter
'''