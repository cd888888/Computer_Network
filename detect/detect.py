import cvcv2 as cv
import numpy as np


'''
#通过面积判断像素点是否是定位点
def judge_qrPoint(con1,con2,con3):
    base1=cv.contourArea(con1)/49
    base2=cv.contourArea(con2)/25
    base3=cv.contourArea(con3)/9
    base=min(base1,base2,base3)
    if(base1-base<=base and base2-base<=base and base3-base<=base):
        return True
    return False
'''
filename="img/6.jpg"
input_img=cv.imread(filename)

gray_img=cv.cvtColor(input_img,cv.COLOR_BGR2GRAY)#将图片通过库转换成灰度图像

ret,binary_img=cv.threshold(gray_img,127,255,cv.THRESH_BINARY)#将图片转换成黑白二值化
#cv.imshow("binary_img",binary_img)

#thresholdImage=cv.Canny(binary_img,100,200)#通过hierarchy层级关系判断定位点
contours,hierarchy=cv.findContours(binary_img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

#print(hierarchy)
points= []#存的是二维码的四个顶点
qrcoder = cv.QRCodeDetector()
ret,points=qrcoder.detect(binary_img)

print(points)
cv.drawContours(input_img, [np.int32(points)], -1, (0, 0, 255), 2)#将二维码圈出


#print("GRAY:", gray_img[500, 500])#输出某点的灰度值 0是黑色，255是白色

cv.circle(input_img,(500,500),1,(0, 0, 255),1)#在原图中圈出定位点

#仿射变换
ptr1=points
ptr2=np.float32([[0,0],[205,0],[205,205],[0,205]])
print(ptr1)
print(ptr2)
M=cv.getPerspectiveTransform(ptr1,ptr2)
res=cv.warpPerspective(input_img,M,(300,300))
cv.imshow("input_img",res)
cv.imwrite("img/a.jpg",res)

cv.waitKey(0)
cv.destroyAllWindows()