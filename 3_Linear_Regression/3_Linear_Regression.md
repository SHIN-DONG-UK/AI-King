# 키워드

- Data Definition
- Hypothesis
- Compute loss
- Gradient discent

## 1. Linear regression

- 리니어 리그레션이 뭐냐
- 1시간 공부했을 때 2점을 얻었고, 2시간 공부했을 때 4점을 얻었으며, 3시간을 공부했을 때 6점을 얻었다고 하자.
- X = [1;2;3], Y = [2;4;6]으로 나타낼 수 있다
- 이제 궁금한거다
- 1시간 30분 공부하면 점수가 몇 점일까?
- 10시간 공부하면 몇 점 정도 나올까?
- 이들 학습 데이터로 직선을 찾는 방법이 linear regression임

## 2. Data Definition

<img width="399" alt="스크린샷 2024-09-30 오후 7 24 43" src="https://github.com/user-attachments/assets/bcadde55-1875-4533-a9db-5036175962be">


- 위와 같이 x_train과 y_train을 구분해서 적어준다

```python
import torch

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
```

## Hypothesis

<img width="310" alt="스크린샷 2024-09-30 오후 7 24 26" src="https://github.com/user-attachments/assets/ff2cf188-81c0-4ded-91d9-6cf74a406f1f">


- W와 b를 학습시킨다
- 처음에는 어떤 값을 입력받아도 0을 예측하도록 W와 b를 0으로 초기화한다
- requires_grad = True로 설정하여 학습 대상임을 표기한다

```python
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
hypothesis = x_train * W + b
```

## Compute loss

- 학습을 잘 하려면, 우리 모델이 얼마나 정답에 가까운지 알아야 함
- 정답과 얼마나 가까운지의 숫자를 cost 혹은 loss라고 함

<img width="739" alt="스크린샷 2024-09-30 오후 7 14 57" src="https://github.com/user-attachments/assets/ef1fb723-3fcc-4b6d-a915-a1bd06ac8032">



- 손실 함수(loss function) : “정답에 얼마나 가까운지”를 계산하는 함수(Mean Squared Error, MSE)
- MSE : 예측값과 정답값의 차이를 제곱해서 평균낸거

```python
cost = torch.mean((hypothesis - y_train) ** 2)
```

## Gradient descent

- 경사 하강법이라고 함
- 위에서 “정답에 얼마나 가까운지”를 계산하는 함수를 통해 수치를 얻을 수 있음
- 이 수치가지고 weight와 bias라는 두 개의 변수를 조정해야 함
- 조정하는 방법을 gradient descent라고 함
- 여기서는 SGD라는 알고리즘 쓸거임
- 이게 뭔지는 아직 안알랴쥼
- pytorch에서는 아래와 같이 사용함

```python
# [W, b]는 학습할 텐서
# lr은 learning rate
optimizer = optim.SGD([W,b], lr=0.01)

# zero_grad()로 gradient 초기화
# backward()로 gradient 계산
# step()으로 개선
optimizer.zero_grad()
cost.backward()
optimizer.step()
```

## How does gradient descent minimize cost?

### 잘 학습된 데이터란?

- “정답에 얼마나 가까운지”의 숫자가 0에 가까울 수록 잘 학습된 것
- 예를 들어, bias를 삭제하고 $y = w x$라는 수식으로 예측한다고 하자
- 그럼 w가 1에 가까울수록 잘 학습된 것
- loss function을 MSE로 하면 w와 cost의 관계는 다음 그림과 같다

<img width="375" alt="스크린샷 2024-09-30 오후 7 43 33" src="https://github.com/user-attachments/assets/1efe7f64-61b9-4e7a-bc24-72d7b534ff0a">


- 그럼 여기서 우리가 해야할 것은?
- 곡선을 내려가자
- 기울기가 양수면 기울기가 작아져야하고, 기울기가 음수면 기울기가 커져야 함
- 기울기가 가파르면 크게 변하고, 기울기가 작으면 작게 변해야 기울기가 0에 수렴하여 오차가 작아질 수 있다
    
<img width="651" alt="스크린샷 2024-09-30 오후 9 11 24" src="https://github.com/user-attachments/assets/be3db8bd-c0d8-406a-bf3b-f39a6e9b57b9">

    
- 이렇게 기울기로 어떻게 w가 변해야 하는지 정할 수 있다
- 이러한 기울기를 “Gradient”

```python
gradient = 2 * torch.mean((W * x_train - y_train) * x_train)
lr = 0.1
W -= lr * gradient
```

### torch.optim으로 gradient를 개선할 수 있음!

- 앞에서 사용했던 optim.SGD함수가 이제 이해가 됨
- optim.SGD는 gradient descent를 수행하여 W를 최적화하는 함수임

## What’s Next?

- 지금까지는 하나의 정보로부터 추측하는 모델을 만들었음
    - 수업 참여도 → 시험 점수
    - 총 수면 시간 → 집중력
- 하지만 대부분의 추측은 많은 정보를 추합해서 이루어짐
    - 쪽지 시험 성적들 → 중간고사 성적
    - 암의 위치, 넓이, 모양 → 치료 성공률
- 여러 개의 정보로부터 결론을 추측하는 모델은 어떻게 만들까?
