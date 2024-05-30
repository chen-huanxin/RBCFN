import cv2

img = cv2.imread('src.jpeg')
print(img.shape)
x_local = img[48:(48 + 128), 48:(48 + 128),:]
print('x_local type: ')
print(type(x_local))   
print(x_local.shape)   
cv2.imwrite('pic1.jpg', x_local)

x_local2 = img[ 80:(80 + 64), 80:(80 + 64), :]
print('x_local2 type: ')
print(type(x_local2))   # class 'torch.Tensor'
print(x_local2.shape)  # [1,3,64,64]
cv2.imwrite('pic2.jpg', x_local2)