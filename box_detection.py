import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import sys

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

#Function which will read in the image
def read_img(img_name):
    img = cv2.imread(img_name, 0)
    img_color= cv2.imread(img_name)
    return img,img_color

# Adaptive thresholding
def thresh_img(grayscale_img):
    inverted_thresh_img = cv2.adaptiveThreshold(grayscale_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    return inverted_thresh_img

def gen_struct_elem(img):
    # length of minimum line
    struct_elem_length = np.array(img).shape[1] // 100
    # to get the vertical lines
    verticle_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1, struct_elem_length))
    # to get the horizontal lines
    horizontal_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (struct_elem_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return struct_elem_length,verticle_struct,horizontal_struct,kernel


def get_lines(inverted_thresh_img,verticle_kernel,horizontal_kernel):
    # Morphological operations to extract vertical lines
    img_temp1 = cv2.erode(inverted_thresh_img, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    # Morphological operation to extracxt horizontal lines
    img_temp2 = cv2.erode(inverted_thresh_img, horizontal_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, horizontal_kernel, iterations=3)
    return verticle_lines_img,horizontal_lines_img

def get_boxes(verticle_lines_img,horizontal_lines_img,kernel):
    img_boxes = cv2.addWeighted(verticle_lines_img, 0.5, horizontal_lines_img, 0.5, 0.0)
    img_boxes = cv2.erode(~img_boxes, kernel, iterations=2)
    (thresh, img_boxes) = cv2.threshold(img_boxes, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_boxes

def gen_points(x,y,w,h):
    return {"points": [[x, y], [x, y+h], [x+w, y], [x+w, y+h]]}




def main():

    input_args = sys.argv

    # This will read-in the image and convert it to grayscale for thresholding
    img,img_color = read_img(input_args[1])

    # Adaptive thresholding is used and the image is inverted to binarize the image
    inverted_thresh_img = thresh_img(img)

    # generate appropriate struct elems for line detections
    struct_elem_length, verticle_struct, horizontal_struct, kernel = gen_struct_elem(inverted_thresh_img)

    # returns the horizontal and vertical lines in the image
    verticle_lines_img, horizontal_lines_img = get_lines(inverted_thresh_img,verticle_struct,horizontal_struct)


    # returns the images adding the vertical and horizontal boxes
    img_boxes = get_boxes(verticle_lines_img,horizontal_lines_img,kernel)



    im_, contours, hierarchy = cv2.findContours(img_boxes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")


    idx = 0
    all_boxes = []
    for i in range(0,len(contours)):

        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(contours[i])
        #setting the font
        font = cv2.FONT_HERSHEY_SIMPLEX
        #writing the box number onto the image
        cv2.putText(img_color, str(i), (x+50, y+50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # setting a threshold for the box size
        if (w > 30 and h > 30):
            idx += 1
            #drawing the boxes onto the image
            cv2.drawContours(img_color,contours,idx,(248,125,208),3)
            all_boxes.append(gen_points(x, y, w, h))

    #creating the json and output file
    with open("data_file.json", "w") as write_file:
        temp = {"boxes":all_boxes}
        json.dump(temp, write_file)
    cv2.imwrite("output.jpg",img_color)



            



if __name__ == '__main__':
    main()