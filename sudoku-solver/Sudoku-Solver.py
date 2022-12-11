
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import imutils
from solver import *

#co tong cong 9 class de du doan nen tao mang tu chua tu 0 den  9
classes = np.arange(0, 10)

#su dung load_model de tai model da duoc pre-train
model = load_model('model-OCR.h5')
input_size = 48


def get_perspective(img, location, height = 900, width = 900):
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    # su dung thuat toan chuyen doi phoi canh?
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result

#ham get_InvPerspective nguoc lai voi ham get_perspective no lay hinh anh va chieu no vao vung mat phang khac da chon
def get_InvPerspective(img, masked_num, location, height = 900, width = 900):
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[3], location[1], location[2]])

    # su dung thuat toan chuyen doi phoi canh?
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(masked_num, matrix, (img.shape[1], img.shape[0]))
    return result



#tim bang bang cach su dung phat hien duong vien`
def find_board(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #chuyen doi anh sang mau` xam de phat hien duong` vien`
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20) #loai bo nhieu~ bang cv2.bilateralFilter
    edged = cv2.Canny(bfilter, 30, 180) #su dung cv2.Canny() de phat hien cac canh trong hinh` anh
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #phat hien cac diem lien tuc nam trong canh bang cv2.findContours
    ##cac diem nay duoc goi la cac duong bao
    contours  = imutils.grab_contours(keypoints)

    newimg = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)


    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None
    
    # tim duong vien hinh` chu nhat
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break
    result = get_perspective(img, location)
    return result, location


# chia bang thanh 81 hinh anh rieng le
def split_boxes(board):
    rows = np.vsplit(board,9)  #np.vsplit() chia anh theo chieu doc
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)  #np.hsplit() chia anh theo chieu ngang
        for box in cols:
            box = cv2.resize(box, (input_size, input_size))/255.0
            boxes.append(box)
    cv2.destroyAllWindows()
    return boxes

#hien thi so thanh hinh` anh
def displayNumbers(img, numbers, color=(0, 255, 0)):
    W = int(img.shape[1]/9)
    H = int(img.shape[0]/9)
    for i in range (9):
        for j in range (9):
            if numbers[(j*9)+i] !=0:
                cv2.putText(img, str(numbers[(j*9)+i]), (i*W+int(W/2)-int((W/4)), int((j+0.7)*H)), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
    return img

# doc hinh anh bang ham cv2.imread()
img = cv2.imread('sudoku1.jpg')


# trich xuat bang tu hinh anh dau vao
board, location = find_board(img)


gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
rois = split_boxes(gray)
rois = np.array(rois).reshape(-1, input_size, input_size, 1)

#du doan chu so cua moi o
# nhan du doan
prediction = model.predict(rois)

predicted_numbers = []
# nhan lop tu du doan
for i in prediction: 
    index = (np.argmax(i)) # tra ve gia tri cua so luong toi da cua mang
    predicted_number = classes[index]
    predicted_numbers.append(predicted_number)

#chuong trinh giai ma tran 9x9 la dang 2D, nhung danh sach du doan la 1D nen ta can phai dinh hinh lai danh sach
board_num = np.array(predicted_numbers).astype('uint8').reshape(9, 9)



#giai bang
try:
    solved_board_nums = get_board(board_num) #dua ma tran vao ham get_board tu chuong trinh giai

    # tao mot mang nhi phan cua cac so  duoc du doan. 0 co nghia la so chua duoc giai cua sudoku va 1 co nghia la so da cho
    binArr = np.where(np.array(predicted_numbers)>0, 0, 1)
    # chi nhan cac so da giai cho bang da giai
    flat_solved_board_nums = solved_board_nums.flatten()*binArr
    #tao mask
    mask = np.zeros_like(board)
    #hien thi so da giai trong mask o cung vi tri o trong tren bang
    solved_board_mask = displayNumbers(mask, flat_solved_board_nums)
    inv = get_InvPerspective(img, solved_board_mask, location)
    combined = cv2.addWeighted(img, 0.7, inv, 1, 0)
    cv2.imshow("Ket qua", combined)

except:
    print("Giai phap khong ton tai.Cac chu so doc sai mo hinh")

cv2.imshow("Anh dau vao`", img)
cv2.waitKey(0)
cv2.destroyAllWindows()