# Introduction #

## 03 Unsupervised Learning ##

### 1. Lecture note ###

자, 이번 시간에는 머신러닝의 두 번째 주요 문제 유형인 unsupervised learning<sup>비지도학습</sup>에 대해서 알아보겠습니다.

![supervised_learning_recall](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/UnsupervisedLearning_supervised_learning_recall.png?raw=true) **<그림1>**

지난 시간에는 supervised learning<sup>지도학습</sup>에 대해서 학습했죠? supervised learning에서는 데이터에 레이블이 매겨져 있었습니다. 특정 데이터가 positive인지 negative인지 답을 알려주는 레이블이 있었죠. 가령 tumor<sup>종양</sup>에 대한 데이터에서는 그것이 benign<sup>양성</sup>인지 malignant<sup>악성</sup>인지 레이블링이 되어 있었습니다.

#### __unsupervised learning이란? ####

+ @@@데이터의 구조를 발견하기 위한 기계학습

![no label](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/UnsupervisedLearning_no%20label.png?raw=true) **<그림2>**

그런데 unsupervised learning에서는 레이블이 매겨져 있지 않은 데이터를 다룹니다. 엄밀히 말하면, 모든 데이터가 하나의 레이블로 매겨져 있거나 아예 레이블이 없는 것이죠. supervised learning에서는 데이터 세트와 그 데이터로 예측하고자 하는 것이 무엇인지 알려주는 레이블이 있었다면, unsupervised learning에서는 그저 데이터 세트만 주어집니다. 그리고 다음과 같이 질문합니다. 

    Q. 주어진 데이터에서 구조(structure)를 발견할 수 있을까?

#### __clustering algorithm이란? ####

![unsup.learning_red circle clusters](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/UnsupervisedLearning_red%20circle%20clusters.png?raw=true) **<그림3>**

unsupervised learning 알고리즘을 돌리면 **<그림3>**처럼  데이터가 2개 집단으로 구분될 수 있습니다. 이렇게 데이터의 집단을 나누는 알고리즘을 clustering algorithm<sup>클러스터링 알고리즘</sup>이라고 합니다. cluster는 모여 있는 집단을 뜻합니다. 이 알고리즘은 unsupervised learning에서 활용되는 알고리즘에 속하는 한 종류로써 다양한 곳에서 많이 쓰입니다.

#### __clustering algorithm을 적용한 예시는? (1/2) ####

 + Google News > 같은 주제를 다룬 기사별로 집단을 나누는 클러스터링

클러스터링 알고리즘를 사용한 예시 중 하나가 바로 구글 뉴스입니다. 지금 바로 [Google News](https://news.google.co.kr/?edchanged=1&ned=us&authuser=0 "Google News 링크")를 클릭해서 확인해보세요.

![unsup.learning_googleNews01](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/UnsupervisedLearning_googleNews01.png?raw=true) **<그림4>**

미리 캡쳐해둔 그림을 보겠습니다. **<그림4>**에서 빨간색으로 표시된 기사는 "BP사의 기름 유출 사태"에 대한 뉴스입니다. 하단의 초록색 글씨들은 같은 주제를 다룬 다른 언론사들의 기사를 연결해주는 링크입니다. 전 세계 뉴스 기사는 굉장히 많을 텐데 같은 주제에 대한 다른 언론사들의 기사를 한 번에 모을 수 있는 방법은 무엇일까요?

구글 뉴스가 한 것은 뉴스 기사 수천 수만 개를 찾고 그것들을 자동으로 클러스터링한 것입니다. 그렇게 하면 같은 주제를 가진 모든 뉴스 기사가 한 곳에 모이는 거죠. 

+ genomics > 유사한 유전자를 가진 개체별로 집단을 나누는 클러스터링

다음 예시는 genomics<sup>유전체학</sup>입니다.

![unsup.learning_genomics](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/UnsupervisedLearning_genomics.png?raw=true) **<그림5>**

**<그림5>**는 DNA microarray 데이터입니다. 유전체학은 여러 사람의 DNA 데이터를 모아두고 같은 유전자를 얼마나 가지고 있는지 측정합니다. 엄밀히 말하면, 특정 유전자가 발현되는 정도를 측정하는 것입니다. 위 그림에 나타난 빨간색, 녹색, 회색 등은 다른 사람들이 특정 유전자를 얼마나 갖고 있는지 정도를 나타냅니다. 

이제 여러분이 할 일은 클러스터링 알고리즘을 돌려서 여러 사람을 카테고리로 또는 타입으로 나누면 됩니다.

이런 작업이 unsupervised learning에 속하는 이유는 사전에 어떤 사람은 type1이고 또 다른 사람은 type2에 해당한다는 정보 없이 알고리즘을 돌리기 때문입니다. 그저 많은 데이터가 있음을 알려주기만 합니다.

#### __clustering algorithym을 적용한 예시는? (2/2) ####

계속해서 unsupervised learning과 클러스터링 알고리즘을 적용한 또 다른 예시들을 알아보겠습니다.

![unsup.learning_applications](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/UnsupervisedLearning_applications.png?raw=true) **<그림6>**

+ 컴퓨터 클러스터(computing cluster) 구성 > 함께 작동하는 컴퓨터가 무엇인지 알아내는 클러스터링

컴퓨터 클러스터는 여러 대의 컴퓨터들이 연결되어 하나의 시스템처럼 동작하는 컴퓨터들의 집합을 말합니다. 보통 아주 큰 데이터 센터에 있는 컴퓨터들이 컴퓨터 클러스터로 구성되어 있습니다. 이때 어떤 컴퓨터끼리 함께 작동하는지 아는 것이 중요합니다. 왜냐하면 함께 작동하는 컴퓨터를 함께 두어야 데이터 센터의 작업 효율이 높아지기 때문입니다. 

+ 소셜 네트워크 분석(social network analysis) > 유대 관계가 높은 친구 집단 또는 서로를 알고 있는 집단을 알아내는 클러스터링

다음은 소셜 네트워크 분석입니다. 당신이 이메일을 가장 많이 보낸 친구에 대해서 알고 있거나, 페이스북에 등록된 친구 또는 Google+ circles에 대한 데이터를 갖고 있다면 유대 관계가 끈끈한 친구 집단을 자동으로 알아낼 수 있을까요?  또 서로를 알고 있는 집단이 어디인지 알아낼 수 있을까요?

+ 시장 세분화(market segmentation) > 유사한 성질별로 시장을 세분화 하는 클러스터링  

회사들은 방대한 고객 정보를 갖고 있습니다. 고객 데이터를 보고 자동으로 세분 시장을 발견하고 고객을 다른 시장 부문에 자동으로 그룹화하여 서로 다른 시장 부문을 자동으로 더 효율적으로 판매 또는 마케팅 할 수 있습니까? 이 역시 어떤 세분 시장에 어떤 고객이 속하는지 사전에 알 수 없으므로 unsupervised learning 문제 유형이다.

+ 천문 데이터 분석(astronomical data analysis) > 

마지막 예시는 놀랍게도 천문 데이터 분석입니다. 천문학에서 클러스터링 알고리즘은 은하계가 어떻게 생성되었는지 이론적인 근거를 제공해주는 데 사용됩니다. 

지금까지 든 예시들은 모두 unsupervised learning에 속하는 여러 종류 중 한 가지인 클러스터링 알고리즘의 예시입니다. 

#### __칵테일 파티 문제 (1) 개요 ####

또 다른 예시 하나를 말씀드리겠습니다. 바로 칵테일 파티 문제<sup>cocktail party problem</sup>입니다. 


> 칵테일 파티 문제는 칵테일 파티 효과<sup>cocktail party effect</sup>와 관련이 있습니다. 이 효과는 파티의 참석자들이 시끄러운 주변 소음이 있는 방에 있어도 상대방과 이야기를 선택적으로 집중하여 잘 받아들이는 현상에서 유래한 말입니다. 즉 주변 환경에 개의치 않고 자신에게 의미 있는 정보만을 선택적으로 받아들이는 선택적 지각<sup>selective perception</sup>을 뜻합니다. 다른 말로 자기 관련 효과<sup>self-referential effect</sup>, 잔치집 효과라고도 합니다.

칵테일 파티에 가본 적 있나요? 방 안에 사람이 가득하고 모두가 여러 자리에 앉아 있으며 다들 이야기를 나누고 있습니다. 사람들의 목소리가 동시에 겹쳐서(overlapping) 들리기 때문에 당신 앞에 있는 사람의 말을 듣는 것도 쉽지가 않습니다. 

자, 여기서 칵테일 파티에 두 사람만이 참석했다고 해봅시다. 소규모 칵테일 파티이고 두 사람은 동시에 이야기를 하고 있습니다. 그리고 방 안에 2개의 마이크로폰을 설치합니다. 

![usup.learning_cocktailparty](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/UnsupervisedLearning_cocktailparty.png?raw=true) **<그림7>**

2개의 마이크로폰은 두 사람으로부터 각각 다른 거리에 떨어져 있습니다. 마이크로폰과 사람과의 거리가 각각 다르다보니, 마이크로폰은 두 사람의 목소리를 다른 조합으로 녹음하게 됩니다. 대화자#1은 마이크로폰#1에 더 크게 들리고 대화자#2는 마이크로폰#2에 더 크게 들립니다. 대화자와 마이크로폰 사이의 상대적 거리가 각각 다르기 때문이죠. 하지만 모든 마이크로폰은 두 대화자의 목소리가 겹쳐서 녹음이 됩니다.

#### __칵테일 파티 문제 (2) 2가지 레코딩 파일 ####

실제 녹음본을 들려드리겠습니다. 

![unsup.learning_cocktailparty_actual_recording](https://raw.githubusercontent.com/datalater/ML_AndrewNg_study/bae61cd3ed1c2f1fc1793ec4b407c05a2a0e9824/images/UnsupervisedLearning_cocktailparty_actual_recording.png) **<그림8>**

+ 마이크로폰#1  
One (uno), two (dos), three (tres), four (cuatro), five (cinco), six (seis), seven (siete), eight (ocho), nine (nueve), ten (y diez).

두 사람이 1부터 10까지 숫자를 세는 데 각자 다른 언어로 말하는 것입니다. 마이크로폰#1은 영어 카운팅이 스페인어 카운팅보다 더 크게 들렸습니다.

+ 마이크로폰#2  
Uno (one), dos (two), tres (three), cuatro (four), cinco (five), seis (six), siete (seven), ocho (eight), nueve (nine) y diez (ten)

마이크로폰#2는 스페인어 카운팅이 영어 카운팅보다 더 크게 들렸습니다. 

#### __칵테일 파티 문제 (3) unsupervised learning 알고리즘 활용과 출력 ####

여기서 우리는 위 2가지 마이크로폰 레코더를 가져온 후 unsupervised learning 알고리즘 중 하나인 칵테일 파티 알고리즘<sup>cocktail party algorithm</sup>을 돌려서 데이터의 구조를 파악할 수 있습니다. 

알고리즘이 수행할 작업은 오디오 레코딩을 듣고 "두 오디오 레코딩이 함께 추가된 것 같다" 또는 "두 오디오 레코딩이 함께 합쳐져서 우리가 가진 레코딩을 생성한 것 같다"고 말하는 것입니다. 그리고 알고리즘은 추가되거나 합쳐져서 다른 레코딩을 형성하는 두 개의 오디오 소스를 분리하는 작업을 할 수 있습니다. 실제로 작업을 수행한 칵테일 파티 알고리즘의 첫 번째 출력입니다.

+ 출력#1  
One, two, three, four, five, six, seven, eight, nine, ten.

출력#1은 레코딩에서 영어 목소리만 분리해냈습니다.

+ 출력#2  
Uno, dos, tres, quatro, cinco, seis, siete, ocho, nueve y diez.

출력#2는 스페인어 목소리만 따로 분리했습니다.


#### __칵테일 파티 문제 (4) 배경 음악과 사람 목소리 분리하기 ####

이번에는 다른 상황입니다. 라디오 목소리와 영어 카운팅이 혼합되어 있습니다.

+ 마이크로폰#1  
(Radio BGM) One, two, three, four, five, six, seven, eight, nine, ten.

마이크로폰#1은 Radio에서 음악 소리가 흘러나오고 있고 한 사람이 영어로 1부터 10까지 카운팅하고 있는데, 카운팅 목소리가 크게 들립니다.

+ 마이크로폰#2  
Radio BGM (One, two, three, four, five, six, seven, eight, nine, ten)

마이크로폰#2는 카운팅 목소리가 작고 Radio 음악 소리가 더 크게 들립니다.

두 녹음파일을 가지고 칵테일 파티 알고리즘을 돌리면, 알고리즘은 두 오디오 레코딩이 함께 합쳐져 있음을 인지하고 분리해낼 수 있습니다.

+ 출력#1  
One, two, three, four, five, six, seven, eight, nine, ten.

출력#1에서는 사람 목소리가 크게 들리고 Radio BGM의 소리는 매우 작아졌습니다.

+ 출력#2  
Radio BGM

출력#2에서는 사람의 목소리가 거의 또는 아예 들리지 않습니다. 오직 Radio BGM 소리만 들립니다. 

#### __ 프로그래밍 환경의 중요성과 Octave ####

이렇게 오디오 프로세싱 알고리즘을 만들려면 수없이 많은 코드 줄을 써야 하거나 이와 관련된 JAVA 라이브러리를 활용해서 매우 복잡한 프로그램을 만들어야만 가능할지도 모릅니다.

그런데 이 알고리즘은 한 줄의 코드로 작성할 수 있습니다.

![one_line_code](https://github.com/datalater/ML_AndrewNg_study/blob/master/images/UnsupervisedLearning_cocktail_party_algorithm_one_line_code.png?raw=true) **<그림9>**  

사실 이 한 줄의 코드를 짜기 위해 연구자들은 긴 세월을 보내야 했습니다. 쉬운 문제라고 얘기하려는 것이 아닙니다. 올바른 프로그래밍 환경을 사용한다면 많은 학습 알고리즘들이 정말로 짧은 코드의 프로그램이 될 수 있다는 것을 말하려는 것입니다. 그리고 이는 우리 수업에서 Octave 프로그램을 활용하려는 이유이기도 합니다.

Octave는 무료 오픈 소스 소프트웨어이며 Octave나 Matlab 같은 도구를 사용하면 많은 학습 알고리즘이 단 몇 줄만으로도 실행할 수 있게 됩니다. 추후 수업에서 Octave를 사용하는 법을 배울 것이고 몇 가지 알고리즘들을 Octave에서 직접 실행해볼 것입니다. 또는 Matlab을 사용해볼 수도 있습니다.

실제로 실리콘 밸리에서 많은 기계 학습 알고리즘을 사용할 때 처음에 Octave를 사용해서 프로토타입을 만듭니다. Octave에 있는 소프트웨어는 학습 알고리즘을 믿을 수 없을 정도로 빠르게 실행할 수 있도록 도와주기 때문입니다.

**<그림9>**에 보면 SVD 함수가 있습니다. SVD 함수는 singular value decomposition의 약자로서 linear algebra routine<sup>선형 대수를 활용하는 프로그램 루틴</sup>인데 Octave에 이미 내장되어 있습니다.

이런 것들을 C++이나 Jave로 하려면 코드 줄이 많아져야 하고 그 코드들은 C++이나 Java의 복잡한 라이브러리와 연관되어 있을 것입니다. 물론 C++이나 Java, 또는 Python으로도 기계 학습 알고리즘들을 실행할 수 있지만 Octave보다는 훨씬 더 복잡한 과정을 거쳐야만 할 것입니다.

제가 거의 10년 간 기계 학습을 가르치면서 느낀 점은 Octave를 프로그래밍 환경으로 사용하는 학생이 훨씬 더 빠르게 배운다는 점입니다. Octave를 학습 도구와 프로토타이핑 도구로 사용하면 자연스럽게 학습 알고리즘을 프로토타입으로 만드는 방법을 훨 씬 더 빠르게 배우게 될 것입니다.

실제로 실리콘 밸리의 큰 회사에서 일하는 사람들은 Octave를 활용해서 학습 알고리즘에 대한 첫 프로토타입을 만들고, 프로토타입이 제대로 작동한 후에야 C++이나 JAVA 또는 다른 무언가에 마이그레이트(migrate, 프로그램을 다른 시스템으로 바꾸는 것)시킵니다. Octave에서 프로토타입을 만들고 나서 C++이나 다른 언어로 마이그레이트 시키는 방법은 아예 처음부터 C++로 시작하는 방법보다 훨씬 더 빠르게 학습 알고리즘을 작동시키는 길입니다. 

교육자로서 말씀드립니다. 프로토타입을 만들 때 Octave를 활용하는 것이 다른 언어보다 훨씬 더 생산성이 높습니다. 제 말을 믿으셔도 좋습니다.

#### review question ####

복습 문제를 내드리겠습니다. 이번 시간에는 unsupervised learning에 대해서 배웠습니다. unsupervised learning은 알고리즘에게 방대한 데이터를 주고 데이터에 내재되어 있는 구조를 발견하라고 묻는 학습 환경(learning setting)입니다. 

다음 질문에 답해주십시오.



### 2. Recap ###

Q1. 

### 3. Summary document ###



</br></br></br>
끝.