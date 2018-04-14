import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# path_data = '/home/sam/fast_ws/src/ork_yolo/data/imagenet'
path_data = current_dir + '/'

# Percentage of images to be used for the test set
percentage_test = 10;

# Create and/or truncate train.txt and test.txt
file_train = open('train.txt', 'w')

# Populate train.txt
counter = 1
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    file_train.write(path_data + title + '.jpg' + "\n")

