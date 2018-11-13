 ```
 CS4246
```
## MCT algorithm 
* [check this website](http://mcts.ai/about/)

## Variables

* [check observation variables](https://gym.openai.com/docs/#observations)
* [check environment variables](https://gym.openai.com/docs/#environments)
* [more information about variables in openai](https://www.oreilly.com/learning/introduction-to-reinforcement-learning-and-openai-gym)

## Note
* `pip install gym` and `pip install gym[atari]` to run `breakout env`
* You need to change the file path on your PC in order for `uct.py` `read` and `write` file properly. This also means that `uct.py` cannot be trained on the server. `read` and `write` make sure that we can trian and get data discretely. Comparing to `uct.py`, `qlearning.py` and `sarsa.py` run much faster, thus we can get result in one shot.
