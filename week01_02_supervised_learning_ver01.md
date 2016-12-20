# Introduction #

## 02 Supervised Learning ##

### 1. Lecture note ###

이 강의에서는 가장 일반적인 기계학습 유형인 supervised learning<sup>지도 학습</sup>이 무엇인지 알아보도록 하겠습니다. 공식적인 정의는 뒤에서 이야기하도록 하고 쉽게 설명하기 위해 예를 들어보겠습니다. 

주택 가격을 예측해 보고 싶다고 가정해 봅시다. 몇 년 전 한 학생이 관련 데이터 세트를 오리곤<sup>Oregon</sup>주 포틀랜드<sup>Portland</sup>시 기관에서 가져왔습니다. 그 데이터를 그래프로 나타내면 아래와 같습니다.

![housing_price_plot](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/SupervisedLearning_housing_price_plot.PNG?raw=true) **<그림1>**  

수평(x)축은 주택의 사이즈(square feet, 평방피트)를 뜻하고 수직(y)축은 주택 가격($1,000)을 뜻합니다. 


----------


이 데이터를 토대로 볼 때, 당신의 친구가 750평방피트의 집을 갖고 있는데 얼마에 팔아야 할지 궁금해 한다고 합시다. 이 때 학습 알고리즘은 어떤 도움을 줄 수 있을까요? 

학습 알고리즘이 할 수 있는 한 가지는 데이터를 관통하는 일직선을 만드는 것, 즉 직선을 데이터에 적합(fit)시키는 것입니다. 

![housing_price_straightline](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/SupervisedLearning_housing_price_straightline.png?raw=true) **<그림2>**  

이러한 직선으로 본다면 친구의 집은 약 $150,000에 팔릴 것으로 예상할 수 있습니다.


----------


더 나은 방법도 가능합니다. 예를 들어, 데이터에 직선을 적합시키기 보다는 이차함수(quadratic function) 또는 2차 다항식 함수(second-order polynomial)를 만드는 게 더 나을 수도 있습니다.

![housing_price_quadraticfunction](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/SupervisedLearning_housing_price_quadraticfunction.png?raw=true) **<그림3>**  

포물선 형태의 이차함수 그래프를 만들어서 예측을 해보면, 친구의 집을 약 $200,000에 팔 수 있을 것으로 예측할 수 있습니다. 


----------

우리가 나중에 논의할 내용 중 하나는 데이터에 직선을 적합시켜야 하는 경우와 이차함수를 적합시켜야 하는 경우를 구별하는 방법입니다. 여기서 친구의 집을 더 잘 팔리게 해줄 정답이란 건 없습니다. 그러나 각각의 방법들은 학습 알고리즘을 배우는 데 있어서 좋은 예시입니다. 

자, 지금까지 살펴본 예시가 바로 supervised learning<sup>지도 학습</sup> 알고리즘입니다. supervised라는 용어는 데이터에 이미 정답이 주어졌기 때문에 붙은 것입니다. 즉, 학습 알고리즘을 적용하기 전에 데이터 세트에는 이미 x값(주택 사이즈)에 해당하는 정답인 y값(주택 가격)이 주어져 있었습니다. 집 크기가 얼마였을 때 실제로 얼마에 팔렸다라는 정답을 알고 있는 상태이기 때문에 잘못되지 않도록 이미 감독되어(supervised) 있는 학습인  거죠. 그래서 알고리즘은 이미 주어진 (x, y) 쌍을 토대로 위에서 언급한 친구의 집 사이즈와 같은 새로운 x값을 넣어서 주택 가격인 y값을 예측했던 것입니다. 

전문용어를 사용하면, 이러한 주택 가격 예측 문제는 regression problem<sup>회귀 문제</sup>라고 부릅니다. 회귀 문제는 주택 가격처럼 연속적인 출력 값(continuous value output)을 예측해야 하는 문제를 뜻합니다. 엄밀히 따지자면 가격은 cent 단위로 반올림할 수 있어서 비연속적인 값(discrete values)이지만 일반적으로 볼 때 우리는 주택 가격을 실수(real number)로서, 값을 가지고 있는 스칼라 값(scalar value)으로서, 연속적인 값을 가진 숫자로 간주합니다. 그리고 **regression이라는 용어는 연속적인 값을 가진 특성을 예측하는 경우를 뜻합니다**.

supervised learning의 또 다른 예시를 알아볼까요? 제 친구들 몇 명과 함께 작업했던 내용입니다. 의료 기록을 보면서 유방에 있는 종양이 악성(malignant)인지 양성(benign)인지 예측하고자 합니다. 환자의 가슴에 혹이나 종양이 발견되었을 때, 악성 종양(malignant tumor)은 매우 위험하고 심각한 종양을 뜻하고 양성 종양(benign tumor)은 무해한 종양을 뜻합니다. 그러므로 악성인지 양성인지 판단하는 문제는 굉장히 중요하고 많은 사람들이 관심을 기울이는 문제입니다. 

----------

그럼 주어진 데이터가 아래 **<그림4>**처럼 생겼다고 해봅시다. 수평(x)축에는 종양의 크기를, 수직(y)축에는 0과 1을 표기합니다. 0은 악성종양이 아니다(No), 1은 악성종양이 맞다(Yes)를 뜻합니다. 

![Breast cancer plot](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/SupervisedLearning_Breast%20cancer%20plot.png?raw=true) **<그림4>**  

데이터에 양성 종양(파란색)과 악성 종양(빨간색)이 각각 5개씩 있습니다. 

----------

자 여기서 문제 들어갑니다. 정말 비극적인 일이지만 가상의 친구가 유방에 종양이 있고 종양의 사이즈가 분홍색 화살표에 위치한다고 합시다. 

![Breast cancer friend example](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/SupervisedLearning_Breast%20cancer%20friend%20example.png?raw=true)
**<그림5>**  

이때 친구의 종양이 악성인지 양성인지 판단하려고 할 때 각각의 확률이 얼마나 될까요? 기계학습 분야에서는 이러한 유형의 문제를 전문용어로 classification problem<sup>분류 문제</sup>이라고 합니다. classification이라는 용어는 이산 출력 값(discrete value output)을 예측해야 하는 문제를 뜻합니다. 이산 출력 값은 '0 또는 1', '악성 또는 양성'과 같이 출력 값이 분류할 수 있는 경우일 때 사용됩니다. 이산(離散)이라는 말은 연속적인(continuous) 값과 달리 출력 값이 비연속적이고 흩어져 있음을 의미합니다. 출력 값은 보통 2가지이나 3가지 이상인 경우도 있습니다. 

예를 들어, 유방암의 종류가 3가지라고 합시다. 그러면 총 4가지 출력 값이 나오게 됩니다.   

+ type 1 malignant tumor 
+ type 2 malignant tumor
+ type 3 malignant tumor
+ Not tumor (=benign) 


----------

classification problem을 그래프로 나타내는 다른 방법도 있습니다. 이번에는 다른 심볼을 사용해보겠습니다. 아까와 마찬가지로 예측 변수로서 종양의 크기라는 특성(attribute)을 사용해서 악성인지 양성이니 예측하려고 합니다. 

![classification straight plot](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/SupervisedLearning_classification%20straight%20plot.png?raw=true)**<그림6>**  

동그라미(O) 심볼은 양성을, 엑스자(X) 심볼은 악성을 뜻합니다. 이 **<그림6>**은 위의 **<그림5>**를 일직선으로 맵핑(배치)한 것입니다. 

----------

지금까지는 한 가지 특성(feature, or attribute)만 사용해서 예측하려고 했습니다. 보통 기계학습 문제에서는 한가지 이상의 특성을 사용합니다. 다음 예시로 넘어가 보죠.

종양의 크기만 아는 것이 아니라 환자의 나이도 안다고 해봅시다. 그러면 데이터는 **<그림7>**처럼 보일 것입니다. 

![breast cancer two attriubtes](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/SupervisedLearning_Breast%20cancer%20two%20attributes.png?raw=true) **<그림7>**   

파란색 동그라미는 benign tumor를, 빨간색 엑스자는 malignant tumor를 뜻하고 친구의 종양은 분홍색 동그라미에 위치한다고 합시다. 친구의 종양은 benign일까요 malignant일까요?

----------

기계학습 알고리즘이 이러한 문제를 해결하는 방법은, malignant tumor와 benign tumor를 구별하는 일직선을 그래프에 그리는 것입니다. **<그림8>**처럼 말이죠.

![Breast cancer ML separate line](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/SupervisedLearning_Breast%20cancer%20machine%20learning%20separate%20line.png?raw=true)
**<그림8>** 
 
이 알고리즘에 의하면 친구의 종양은 다행스럽게도 benign tumor일 확률이 높다고 예측됩니다. 지금까지는 2가지 특성까지 사용했지만 실제 현장에서는 아래와 같이 훨씬 더 많은 특성들이 사용됩니다.

+ clump thickness<sup>혹의 두께</sup>
+ uniformatiy of cell size<sup>종양 세포 크기의 균일함</sup>
+ uniformatiy of cell shape<sup>종양 세포 모양의 균일함</sup>
+ 등등

----------

지금 이 슬라이드에서는 5가지 특성들을 나열했지만 현장에서는 무한히 많은 특성들을 기계학습 알고리즘에 사용할 수 있습니다. 그러면 무한한 특성들의 데이터를 다뤄야 할 때 저장의 한계가 있을 텐데 컴퓨터 메모리가 부족하게 되진 않을까요? 나중에 Support Vector Machine(SVM) 알고리즘에 대해 이야기할 시간이 있을 텐데, 이 알고리즘은  수학적인 기술을 활용해서 컴퓨터가 무한한 개수의 특성들을 다룰 수 있게 만들어 줍니다. 즉 가능하다는 거죠.

 
### 2. Recap ###

Q1. supervised learning이란?  
A1-1. 기계를 학습시킬 때 문제지와 정답지를 함께 주는 것  
A1-2. 기계가 과거의 문제지와 정답지를 토대로 새로운 문제의 정답을 예측하는 학습 유형

Q2. supervised learning 예시?  
A2. 집값 예측, 종양 판단 예측

Q3. supervised learning 문제 유형?
A3. regression problems, classification problems

Q4. regression problem이란?  
A4. 연속적인 값(continuous value output)을 예측하는 문제 유형 

Q5. regression problem 예시?  
A5. 집값 예측

Q6. classification problem이란?
A6. 비연속적인 값(discrete value output)을 예측하는 문제 유형

Q7. classification problem 예시?  
A7. 종양 판단 예측

Q8. 다음 문제를 푸시오.
![wrapup question](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/SupervisedLearning_wrap%20up%20question.png?raw=true)

A8. 비공개

### 3. Summary document ###

**Supervised Learning**  

In supervised learning, we are given a data set and already know what our correct output should look like, having the idea that there is a relationship between the input and the output. 

Supervised learning problems are categorized into "regression" and "classification" problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories. 

**Example 1:**

Given data about the size of houses on the real estate market, try to predict their price. Price as a function of size is a continuous output, so this is a regression problem. We could turn this example into a classification problem by instead making our output about whether the house "sells for more or less than the asking price." Here we are classifying the houses based on price into two discrete categories. 

**Example 2:** 

(a) Regression - Given a picture of a person, we have to predict their age on the basis of the given picture 

(b) Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.

</br></br></br>
끝.