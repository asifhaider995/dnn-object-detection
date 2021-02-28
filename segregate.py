import os
import cv2

def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input 'image' of size 
    equal to 'window_size'. The first image returned top-left 
    co-ordinate (0, 0) and are increment in both x and y directions
    by the 'step_size' supplied.
    So, the input parameters are-
    image - Input image
    window_size - Size of Sliding Window 
    step_size - incremented Size of Window
    The function returns a tuple -
    (x, y, im_window)
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield image[y: y + window_size[1], x: x + window_size[0]]

def compile():
    path = os.getcwd()
    # raw_path = os.path.join(path, 'data', 'dataset', 'primary', 'raw')
    processed_path = os.path.join(path, 'data','phase2', 'processed')
    train_path = os.path.join(path, 'data','phase2', 'total')
    pos_path = os.path.join(processed_path, 'pos')
    neg_path = os.path.join(processed_path, 'neg')
    count = 0
    for im in os.listdir(pos_path):
        img = cv2.imread(os.path.join(pos_path, im))
        count += 1
        filename = "human-"+str(count)+".png"
        file_path = os.path.join(train_path, filename)
        cv2.imwrite(file_path, img)
    count = 0
    for im in os.listdir(neg_path):
        img = cv2.imread(os.path.join(neg_path, im))
        count += 1
        filename = "null-"+str(count)+".png"
        file_path = os.path.join(train_path, filename)
        cv2.imwrite(file_path, img)
def main():
    step_s = (60,60)
    ws = (300, 800)
    path = os.getcwd()
    raw_path = os.path.join(path, 'data', 'dataset', 'primary', 'raw')
    processed_path = os.path.join(path, 'data','dataset', 'primary', 'processed')

    # pos_path = os.path.join(processed_path, 'pos')
    
    images = os.listdir(raw_path)
    img = cv2.imread(os.path.join(raw_path, images[13]))
    # img = cv2.resize(img, (1280, 720))
    count = 5456
    for im in sliding_window(img, ws, step_s):
        count += 1
        filename = os.path.join(processed_path, 'sample'+str(count)+".png")
        cv2.imwrite(filename, im)
    
    img = cv2.resize(img, (1080, 576))
    cv2.imshow("Test", img)
    cv2.waitKey(0)

if __name__ == '__main__':
    compile()