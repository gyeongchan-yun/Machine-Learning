import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()

# zip - list를 묶어주는 함수. 반복문에서 list들이 동시에 slicing 됨.
for label, img in zip(digits.target[:10], digits.images[:10]):
    plt.subplot(2, 5, label + 1)  # 하나의 plot에 여러개의 subplot이 존재하도록 만들수 있음. 2 x 5 행렬처럼 설정.
    plt.axis('off')
    plt.imshow(img, cmap='Greys', interpolation='nearest')
    plt.title('Digit:{0}'.format(label))

plt.show()