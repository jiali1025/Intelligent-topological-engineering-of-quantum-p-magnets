from STM_Python_Modules.Model_Resource.ObjectDetection_Module import ObjectDetection
'''
img_array = []  # 2D image from scan, val range in (0,255)
x = []  # global x of large scale img (center) in nm
y = []  # global y of large scale img (center) in nm
'''
 # scan size of the large scale img [W,H] in nm


def ObjDet(img_array, position):
    scan_size = [30.0, 30.0]

    ObjDet_model = ObjectDetection(img_array, position, scan_size)

    Position_List = ObjDet_model.prediction()

    return Position_List

''' Position_List:  A 1-D List of [4*N], 
                    where N is num of molecules found, 4 is global position of molecule in nanometer
                    in [x1, y1, W, H, x2, y2, W, H, ... ] in m, 
                    * W, H is currently set to 3nm as default
'''

