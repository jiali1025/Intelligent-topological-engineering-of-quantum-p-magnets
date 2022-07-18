from ReactionClassification_Module import ReactionClassification

'''
img_array = []  # small scale 2D image from scan, val range in (0,255)
x = []  # global x of small scale img (center) in nm
y = []  # global y of small scale img (center) in nm
'''
scan_size = [40, 40]


def RxnClass(img_array, x, y):
    Rnx_model = ReactionClassification(img_array, x, y, scan_size)

    retry_info = Rnx_model.prediction()

    return retry_info


''' retry_info: [boolean, x, y], 
                where boolean indicate whether to retry (true means no reaction, need retry),
                x,y the global position in nm for new reaction site if the reaction is not successful.
                when moving on (boolean = false), 0 will be assigned to x,y
'''
