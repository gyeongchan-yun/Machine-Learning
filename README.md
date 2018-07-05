Machine-Learning
================

* **코드 환경**:

   * **IDE**: Pycharm 2017.3.4
   
   * **Interpreter**: Anaconda3/envs/tensorflow/python.exe
   
   * **Package**: numpy, matplotlib.pyplot, scikit-learn, scipy
   
* **package 설치 방법** (Pycharm 에서)
   * File -> settings -> Project Interpreter -> 초록색 + button 클릭 -> package 검색 -> 'install package' buttion 클릭
   

## 1. 신경망 첫걸음 
**MNIST data 쓰는 예제들** 
   
   mnist_dataset/ 안에 밑의 파일들 다운로드.
   
   학습 데이터: http://www.pjreddie.com/media/files/mnist_train.csv
   
   테스트 데이터: http://www.pjreddie.com/media/files/mnist_test.csv
   
   100개 학습 데이터: https://git.io/vySZ1
   
   10개 테스트 데이터: https://git.io/vySZP
   
   (raw file로 브라우저에 나타날시, 오른쪽 마우스 클릭 -> 다른이름으로 저장)
   
## 2. 머신러닝 부트캠프 with python 

  * **분류(Classification)**
    * 정답 데이터에서 분류 규칙을 배워 미지의 데이터에서도 분류할 수 있게 하는 것. 분류를 레이블(label)이라 표현.
    * 지도학습 데이터의 속성값을 **설명 변수** , 분류를 **목적 변수**, 정답에 해당하는 분류값을 **레이블** 이라고 표현.
    * **과적합(overfitting)**: 학습 데이터에만 너무 적응해서 미지의 데이터에 적합하지 않은(일반화 성능이 떨어진) 상태.
    * 홀드 아웃 검증(holdout validation): 전체 데이터에서 **테스트 데이터**와 **학습 데이터**를 구분.
    * k분할 교차 검증(k-fold cross validation): 대상 데이터를 k개 분할하여 1개의 테스트 데이터, k-1개의 학습 데이터로 구분
  
  * **회귀(Regression)**
    * 주어진 데이터에서 수치를 예측. 정답 데이터의 규칙을 배워 미지의 데이터에서 대응하는 수치를 예측.
    
  * **클러스터링(Clustering)**
    * 데이터의 성질(feature)에 맞게 클러스터를 만듬. 정답이 필요하지 않음.
    
 * **딥러닝(Deep Learning)**
   * 머신러닝의 하나로 신경망의 일종
   * 오차역전파법(Backpropagation)
      * 단점: 1. 과적합(overfitting)이 쉽게 발생   2. 하이퍼 파라미터의 조율이 어려움
   * 합성곱 신경망(Convolutional Neural Network)
      * 결합 방법에 중첩을 도입
      * 이미지 처리에 자주 사용
   * 순환 신경망(Recurrent Neural Network)
      * 전후 계속성이 중요한 데이터나 동영상
   
