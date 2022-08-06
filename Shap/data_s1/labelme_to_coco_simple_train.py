# import package
import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = r'C:\Users\Laptop\Desktop\DECT\New'

# set path for coco json to be saved
save_json_path = r'C:\Users\Laptop\Desktop\DECT\New\eval13.json'

# conert labelme annotations to coco
labelme2coco.convert(labelme_folder, save_json_path)