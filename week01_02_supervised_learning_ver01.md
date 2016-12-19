# Introduction #

## 02 Supervised Learning ##

이 강의에서는 가장 일반적인 기계학습 유형인 supervised learning<sup>지도 학습</sup>이 무엇인지 알아보도록 하겠습니다. 공식적인 정의는 뒤에서 이야기하도록 하고 쉽게 설명하기 위해 예를 들어보겠습니다. 

주택 가격을 예측해 보고 싶다고 가정해 봅시다. 몇 년 전 한 학생이 관련 데이터 세트를 오리곤<sup>Oregon</sup>주 포틀랜드<sup>Portland</sup>시 기관에서 가져왔습니다. 그 데이터를 그래프로 나타내면 아래와 같습니다.

![housing_price_plot](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/SupervisedLearning_housing_price_plot.PNG?raw=true) **<그림1>**  

수평(x)축은 주택의 사이즈(square feet, 평방피트)를 뜨하고 수직(y)축은 주택 가격($1,000)을 뜻합니다. 


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

Here's another supervised learning example, some friends and I were actually working on this earlier. 


----------

----------

----------

----------

...작성중...

가로축은 각각의 주택의 크기를 평방피트 단위로 표시합니다. 세로축은 각각 주택의 가격을 천 달러 단위로 표시합니다. 이 데이터를 바탕으로 친구가 750평방피트의 주택을 소유하고 있고 팔려고 한다면 어느정도의 가격을 받아야 할까.에 대해 학습 알고리즘을 통해 이를 해결할 수 있을까요? 첫째로 학습알고리즘으로 할 수 있는 것은 데이터를 지나는 직선을 긋고 그 직선을 참고하면, 주택가는 150,000달러정도가 될것이라 보일것입니다. 그렇지만, 학습알고리즘은 이 이상의 것을 할 수 있습니다. 예를 들면, 직선을 데이터에 맞추어 긋는 대신, 이차함수나 이차 방정식을 적용할 수 있습니다. 이를 적용하게 되면 예상치는 여기가 되고 아마도 우리는 집을 $200,000정도에 팔 수 있을 것 같군요. 우리가 나중에 다룰것중 하나는 데이터에 직선을 그릴지 이차 곡선을 그릴지, 어떻게 선택하고 결정하느냐 입니다.