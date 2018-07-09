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
    * classifier 성능
      * confusion matrix
      
          실제\예측   | Positive | Negative
        -------- | ---------| --------
        Positive | True Positive(TP) | False Negative(FN)
        Negative | False Negative(FP) | True Negative(TN)
      
      * 정답률(Accuracy): 전체 예측안에 정답이 있는 비율
      
        ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7BTP%20&plus;%20TN%7D%7BTP%20&plus;%20FP%20&plus;%20FN%20&plus;%20TN%7D)
      
      * 적합률(Precision): classifier가 positive로 예측했을 때, 진짜로 positive인 비율
      
        ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7BTP%7D%7BTP%20&plus;%20FP%7D)
        
      * 재현율(Recall): 진짜 positive인것을 classifier가 얼마나 positive로 예측했는지 나타내는 비율
      
        ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7BTP%7D%7BTP%20&plus;%20FN%7D)
        
      * F값(F-measure): 적합률과 재현률의 조화 평균
      
        ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B2%7D%7B%5Cfrac%7B1%7D%7BPrecision%7D%20&plus;%20%5Cfrac%7B1%7D%7BRecall%7D%7D)
    * 여러가지 classifier
      * 결정 트리 (Decision Tree)
        * 데이터를 여러 등급으로 분류하는 지도 학습
      * 앙상블 학습 (Ensemble Learning)
        * 정의: 여러개의 classifier를 생성하고 이들의 예측을 조합해 성능이 높은 classifier를 만드는 학습 방법.
        * 배깅 (Bagging)
          * 학습 데이터를 빼고 중복을 허용해(뽑은 샘플을 다시 집어 넣음) 그룹 여러개로 분할하고 그룹마다 각각의 학습기로 생성.
          * 각각의 약한 학습기(weak learner)가 출력한 결과를 다수결로 결정
          * 랜덤 포레스트 (Random Forest)
            * 전체 학습 데이터 중에서 중복 이나 누락을 허용해 학습 데이터 셋을 여러개 추출하며, 그 속성을 사용해 결정트리 생성
            * 장점: 1. 학습과 판별을 빠르게 처리한다.  2. 학습 데이터의 노이즈에도 강함 3. 회귀나 클러스터링에도 사용가능
            * 단점: 학습 데이터가 적을 때 과적합 발생이 쉬움
        * 부스팅 (Boosting)
          * 난이도가 높은 데이터를 추출하여 이를 분류하는데 특화된 분류기를 순차적으로 만들어가는 형태
          * 에이다부스트 (AdaBoost)
            * 각각의 다른 데이터에 강점을 가지는 약한 학습기를 선택해서 여러개 구축
            * 난이도 높은 학습데이터와 성능이 높은 약한 학습기에 가중치를 주어서 정확도를 높임
            * 장단점: 분류 정밀도가 많이 높지만, 학습 데이터의 노이즈에 쉽게 영향을 받음
      * 서포트 벡터 머신 (SVM)
        * 분할선부터 근접 샘플 데이터까지의 margin(거리의 2제곱)의 합을 최대화하는 직선을 찾는다 (선형 분류)
        * 장단점: 학습 데이터의 노이즈에 강하고 분류 성능이 매우 좋지만, 분류 속도가 느림
      
      
  
  * **회귀(Regression)**
    * 주어진 데이터에서 수치를 예측. 정답 데이터의 규칙을 배워 미지의 데이터에서 대응하는 수치를 예측.
    * 분류
      * 선형 회귀 (Linear Regression)
        * Y = w0 + w1x1 + w2x2 + ... wnxn
        * 선형이 1차식이라고만 할수 없다. 예를 들어, n=2, x1=x, x2= 이면 Y = w0 + w1x + w2x^2
      * 비선형 회귀
     
      * 단순 회귀 (Simple Regression)
        * 입출력 관계가 변수 하나로 구성된 식 (ex) y = ax + b
      * 다중 회귀 (Multiple Regression)
        * 변수를 2개 이상 쓰는 회귀
    * 모델
      * 최소 제곱법
      * SVM
      * 랜덤 포레스트
      * K-근접법 (K-nearest neighbor)
    * 평가 
      * 결정 계수
      
      ![equation](https://latex.codecogs.com/gif.latex?R%5E%7B2%7D%20%3D%201%20-%20%5Cfrac%7BY%7D%7BX%7D)
      Y: 관측 값과 예측 값 차의 제곱 합 X: 관측 값과 측정값 전체 평균의 제곱 합
      * 1에 가까울 수록 좋은 모델
    
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
   
