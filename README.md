# ML-Based Automatic Berthing System

<p align="center">
  <img src="imgs/berthing_trajectory_03.png"/>
</p>

# 사용된 강화학습 알고리즘
[Proximal Policy Optimization (PPO) (2017)](https://arxiv.org/abs/1707.06347) <br>

<small>* 자세한 이론/사항은 구글에 PPO 라고 검색하면 관련 튜토리얼 자료들이 많이나오니, 참고하시면 될듯합니다.</small>

# 강화학습의 전반적인 흐름
<p align="center">
  <img src="imgs/RL-flow2.png"/>
</p>

- Agent: 강화학습을 하는 '주체'. (게임으로 치면, 게임플레이어)
- Environment: 강화학습이 진행되는 '환경'. (게임으로 치면, 플레이어가 돌아다니는 세상(환경) 정도)
- <img src="https://render.githubusercontent.com/render/math?math=s_t">: 시간 t일때의 상태 (state)
- <img src="https://render.githubusercontent.com/render/math?math=r_t">: 시간 t일때의 보상 (reward)
- <img src="https://render.githubusercontent.com/render/math?math=a_t">: 시간 t일때의 행동 (action)
- <img src="https://render.githubusercontent.com/render/math?math=\pi">: 행동을 결정하는 정책 (행동정책)

agent가 현재 상태(state)를 고려하여, 현재의 행동정책(policy)에 따라, 현재 상태에서의 행동(action)을 결정한다.<br>
결정된 행동으로 환경(environment)와 상호작용을 한다. (= 시뮬레이션 상에서 한 타임스텝 넘어간다는 얘기)<br>
보상(reward)과 다음상태(next state)를 얻는다.<br>
위 루프를 계속해서 반복하여, maximum 보상을 받을 수 있도록, 행동정책을 업데이트 해나간다.<br>

# 자동접안 시스템의 전반적인 흐름
<p align="center">
  <img src="imgs/RL_overall_flow1.png"/>
</p>
위는 강화학습 기반 자동접안 시스템의 전박전인 흐름을 보여준다.<br>
여기서 actor, critic이란 각각의 딥뉴럴네트워크(Deep Neural Network, DNN)이다.<br>

- actor는 정책을 mapping하는 DNN (즉, actor가 action을 output함)
- critic은 Q-value를 output하는 DNN 이다. (Q-value란 현재 상태(state)에서 미래에 얼마만큼의 보상을 받을수 있는지를 나타내는 값이다 == 현재 상태가 얼마나 좋은지를 나타내는 값)

<small>* 자세한 사항은 구글에 actor-critic method라고 검색하면 많은 자료가 나오니 참고하면 될듯합니다.</small>

# 강화학습 트레이닝 절차 (pseudo code)
1. random init posx, posy, heading_angle (이때 random하게 생성되는 위치는..)
2. ...
트레이닝 다되면, 테스트는 ...

# 보상함수 정의
보상함수는 r_t 를 얻기위한 함수이다. 트레이닝을 사용한

# PPO의 트레이닝을 위한 Hyper-parameter 세팅
Same as [Here](https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html)
- Parameters 아래에 `class stable_baselines.ppo2.PPO2(...)` 가 있는데, `(...)` 안의 default argument값들을 트레이닝에 사용하였음.
- hyper-parameter 세팅이 PPO논문과 유사하므로, 보고서작성시 논문에 있는 hyper-parameter을 참고하였음 이라고 적어도 무난할 듯 합니다.

# Target Ship 제원
<p align="center">
  <img src="imgs/ship_property.png"/>
</p>
- n, rudder angle 제약


# 트레이닝 결과
<p align="center">
  <img src="imgs/reward_hist.png"/>
</p>
위는 트레이닝동안의 보상(reward) history를 보여준다. <br>
위에서 언급했던대로, 강화학습의 최종목표는 보상(reward)을 maximize하는 것이다. 위의 그래프에서 볼 수 있듯이, 트레이닝이 진행되면서 agent가 획득하는 reward가 점점더 높아지는것을 확인할 수 있다.

# Google Colab에서 실행해보기
Google Colab allows users to use Python on Chrome. Therefore, no installation is required for users.<br>
Click the following link: https://colab.research.google.com/drive/1aIaVj3iYTQVR0WzTayTkqp6cFnJeW98V?usp=sharing

사용법:

여기서 보여지는 trajectory 결과에서 x, y축은 배의 LBP로 나누어 scaling 해준 값이다.
