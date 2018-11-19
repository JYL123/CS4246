## Naive-Reinforcement-Learning-With-Atari-Games 

## Game Environment
Atari game environments in the this project are all available on [Open AI](https://gym.openai.com/envs/#atari). The project focuses on [centipede-ram-v0](https://gym.openai.com/envs/Centipede-ram-v0/), BreakoutDeterministic-v4, and [Taxi-v2](https://gym.openai.com/envs/Taxi-v2/). We have selected these games to observe naive reinforcement learning algorithms (without neutral networks) on how well the algorihtms can guide the agent to play markovian games (Taxi-v2), and non-markovian games (BreakoutDeterministic-v4, Centipede-ram-v0). 

* [check observation variables](https://gym.openai.com/docs/#observations)
* [check environment variables](https://gym.openai.com/docs/#environments)
* [more information about variables in openai](https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym)

## Algorithms 
### Naive Q-Learning
* [check this website for more details](https://en.wikipedia.org/wiki/Q-learning)
* [check this page for implementation details](https://github.com/JYL123/Naive-Reinforcement-Learning-With-Atari-Games/blob/master/qlearning.py)
### SARSA
* [check thsi website](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action)
* [check this page for implementation details](https://github.com/JYL123/Naive-Reinforcement-Learning-With-Atari-Games/blob/master/sarsa.py)
### Monte Carlo Tree Search 
* [check this website](http://mcts.ai/about/)
* [check this page for implementation details](https://github.com/JYL123/Naive-Reinforcement-Learning-With-Atari-Games/blob/master/uct.py)

## Results
* [SARSA plays Taxi-v2](https://github.com/JYL123/Naive-Reinforcement-Learning-With-Atari-Games/blob/master/plot/sarsa_on_Taxi.jpg)

![Alt text](https://github.com/JYL123/Naive-Reinforcement-Learning-With-Atari-Games/blob/master/plot/sarsa_on_Taxi.jpg)

* [Q-learning plays Taxi-v2](https://github.com/JYL123/Naive-Reinforcement-Learning-With-Atari-Games/blob/master/plot/q-learning_on_Taxi_v2.jpg)

![Alt text](https://github.com/JYL123/Naive-Reinforcement-Learning-With-Atari-Games/blob/master/plot/q-learning_on_Taxi_v2.jpg)

* For more information, please check this [report](https://github.com/JYL123/Naive-Reinforcement-Learning-With-Atari-Games/blob/master/report/report.pdf).

## Note

* `pip install gym` and `pip install gym[atari]` to run atari game environments
