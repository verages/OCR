# -*- coding: utf-8 -*-
'''
Author: wayne
LastEditors: wayne
email: linzhihui@szarobots.com
Date: 2021-12-25 14:32:07
LastEditTime: 2021-12-27 15:55:04
Description: eva
'''

from rec.recognizer import TextRecognizer
import cv2
import os


text_recognizer = None
def main(full_path,text_recognizer):
    r = (280, 10, 25, 25)
    # global text_recognizer
    # if text_recognizer is None:
    #     text_recognizer = TextRecognizer(model_path)
    if (os.path.isfile(full_path) and full_path.endswith(".mp4")):
        print("open file:",full_path)
        cap = cv2.VideoCapture(full_path)
        success, frame = cap.read()
        while success:
            img_list = []
            raw_image = frame
            img = cv2.resize(frame,(640, 362))
            new_roi = img[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
            img_list.append(new_roi)
            rec_res = text_recognizer(img_list)
            print(str(rec_res[0]))
            cv2.putText(raw_image, str(rec_res[0][0]), (100, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 10, (0,0,255), 2)
            cv2.imshow("test",new_roi)
            c=cv2.waitKey(int(1000/50) )  # 延迟
            # cv2.imwrite("./raw.jpg",frame)    
            if(c==27):   
                   # ESC键退出
                    break
            # out.write(raw_image)
            success, frame = cap.read() 
        cap.release()


if __name__ == '__main__':     
    test_mp4 = "./video"
    model_path = "./ch_ocr.pth"
    text_recognizer = TextRecognizer(model_path)
    if(test_mp4.endswith(".mp4")):
        main(test_mp4,text_recognizer)
    else:
        for file in os.listdir(test_mp4):
            full_path = os.path.join(test_mp4, file)
            main(full_path,text_recognizer)