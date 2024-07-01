from .envfeature import holdem2ENV
from .predictor import Predictor

def main():
    predictor = Predictor()
    predictor.init_model(model_path='./checkpoint/38000_model/model.ckpt')
    env = holdem2ENV()
    env.reset()
    obs = env.getobs()
    print(type(obs))
    print(obs.shape)
    actions = predictor.get_prob(obs[:-1])
    print(actions)

if __name__ == '__main__':
    main()