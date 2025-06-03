import cv2
import numpy as np  
import sys
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.ops as ops

# #not required
# def high_pass_filter(image):
#     #identity matrix
#     id_mat = np.eye(3, dtype=np.float32)
#     filtered_img = cv2.filter2D(image, ddepth=cv2.CV_8U, kernel=id_mat, anchor=(-1,-1), delta=0, borderType=cv2.BORDER_DEFAULT)
    
#     return filtered_img

def process_main_pipeline(image):
    #KERNEL FOR EVERY OPERATION
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8,8))
    d_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    #will append here later
    objects = []
    
    #GRAYSCALE
    stock_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    normalised_stock_gray = cv2.normalize(stock_gray, None , 0, 255, cv2.NORM_MINMAX)
    #CLAHE c
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_clahe = clahe.apply(normalised_stock_gray)

    stock_gray_blur = cv2.medianBlur(enhanced_clahe, 5)
    stock_gray_blur_1 = cv2.medianBlur(stock_gray_blur, 5)
  
    
    # cv2.imshow("blurred", stock_gray_blur_1)
    # cv2.waitKey(0)
    # cv2.destroyWindow("blurred")

    #OTSUS THRESHOLDING
    _, binarised = cv2.threshold(stock_gray_blur_1, 0, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
    # cv2.imshow("otsu", binarised)
    # cv2.waitKey(0)
    # cv2.destroyWindow("otsu")
    
    # canny edge
    v = np.median(stock_gray_blur)
    stock_gray_blur_edge = cv2.Canny(stock_gray_blur, int(0.66 * v), int(1.3 * v))
    # cv2.imshow("canny",stock_gray_blur_edge)
    # cv2.waitKey(0)
    # cv2.destroyWindow("canny")
    
    dialated_stock = cv2.dilate(stock_gray_blur_edge, d_kernel, iterations=1)

    # cv2.imshow("dialated",dialated_stock)
    # cv2.waitKey(0)
    # cv2.destroyWindow("dialated")   

    #extract contours from the grayscale images
    ( contours, hierarchy ) = cv2.findContours(dialated_stock.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for i,contour in enumerate(contours):
    #     if hierarchy[0][i][3] != -1:
    #         #then its an outer contour
    #         continue

        
    #     x,y,w,h = cv2.boundingRect(contour)
    #     image_area = image.shape[0] * image.shape[1]
    #     rect_area = w * h

    #     if 0.0004 < rect_area / image_area < 0.036:
    #         cropped = image[y:y+h, x:x+w]
    #         objects.append(((x, y , x+w, y+h), cropped))

    # return objects

    #if the detected contour is an outer contour (by heirarchy ) then add that to the list
    for i,contour in enumerate(contours):
        if hierarchy[0][i][3] == -1:
            #then its an outer contour
            x,y,w,h = cv2.boundingRect(contour)
            image_area = image.shape[0] * image.shape[1]
            rect_area = w * h

            #condition to filter very small or very large boxes
            if 0.005 < rect_area / image_area < 0.05 and image.shape[0] // 18 < h < image.shape[0] // 4 and image.shape[1] // 18 < w < image.shape[1] // 4:
                score = rect_area/image_area
                # cropped = image[y:y+h, x:x+w]
                objects.append(((x, y , x+w, y+h), score))

    #APPLYING NMS TO REMOVE OVERLAPPING BOXES

    #corner case for empty tray
    if len(objects) > 0:
        boxes, scores = zip(*objects)
        boxes_nms, scores_nms = nonmax(boxes, scores, 0.2)
    else:
        return 0

    # return list(zip(boxes_nms, scores_nms))

    return len(list(zip(boxes_nms, scores_nms)))


#NON MAX SUPPRESSION TO PREVENT BOX OVERLAP
def nonmax(box, score, threshold):
    box_tensor = torch.tensor(box, dtype=torch.float32)
    score_tensor = torch.tensor(score, dtype=torch.float32)
    pts = ops.nms(box_tensor, score_tensor, threshold)
    return [box[i] for i in pts] , [score[i] for i in pts]


def draw_region(detections, image):
    print("annotating....\n")
    copied_image = image.copy()
    count = 0
    for detection, scores in detections:
        (x0, y0, x1, y1) = detection
        #RGB SO 0 255 0 -> GREEN
        cv2.rectangle(copied_image, (x0, y0), (x1, y1), (255, 0, 0), 2)
        cv2.putText(copied_image, str(count+1), (x0+30, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        count+=1
    plt.imshow(copied_image)
    plt.title("Detected Items")
    plt.axis("off")
    plt.savefig("outputs/annotation.png")



###FOR DEVELOPMENT 
# if __name__ == "__main__":
#     #use the processing in main in function_app.property

#      #for command line argument as the file name
#     print("no of arguments: ", len(sys.argv))
#     n = len(sys.argv)
#     if n==2:

#         stock_src = cv2.imread(f"images_test/{sys.argv[1]}")
#         image_fn = str(sys.argv[1])

#         print(stock_src.shape[0], stock_src.shape[1])
#         # resized_src = cv2.resize(stock_src, (1024, 1024), interpolation=cv2.INTER_AREA)

#         detections, name = process_main_pipeline(stock_src, image_fn)
        
#         print("TOTAL: ")
#         print(name, ": ",len(detections))

#         draw_region(detections, stock_src)

#     else:
#         print("enter the file name")

