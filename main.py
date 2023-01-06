import gymnasium, math, imageio, os, time, numpy
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display

def main():
    numpy.random.seed(42)
    environment = gymnasium.make("CartPole-v1")
    # fakeDisplay = False
    # createDisplay(fakeDisplay)
    
    for generation in range(1):
        state = environment.reset()
        for step in range(200):
            environment.render(mode = "human")
            #Sample a random action
            action = environment.action_space.sample()
            new_state, reward, terminated, truncated, info = environment.step(action)
            state = new_state
            if terminated or truncated:
                print("Episode finished after {} timesteps".format(step+1))
                break
    environment.close()
    
def discrete_states(state, env, buckets=(1, 1, 6, 12)):
    """Function to turn continuous states in the CartPole Game Environment into Discrete States so that the state
        can be used with Q-Learning

    Args:
        state: Current State
        env: The CartPole Game Environment
        buckets : Buckets to discretisize the continuous state. Defaults to (1, 1, 6, 12).
    Return:
        Discrete version of the Current State
    """
    upperBounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50) / 1.]
    lowerBounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50) / 1.]
        
    ratios = [(state[i] + abs(lowerBounds[i])) / (upperBounds[i] - lowerBounds[i]) for i in range(len(state))]    
    state_ = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(state))]
    state_ = [min(buckets[i] - 1, max(0, state_[i])) for i in range(len(state))]
    
    return tuple(state_)
    
def QLearning(env, numberOfEpisodes):
    """Function to train the agent using Q-Learning

    Args:
        env: The CartPole Game Environment
        numberOfEpisodes: The number of Episodes
    Return:
        Q-Table and Reward List
    """
    #Gamma is the Discount factor
    gamma = 0.98
    Q_table = numpy.zeros((1,1,6,12) + (env.action_space.n,)) #env.action_space.n returns the number of valid actions
    
    total_reward = []
    for e in range(numberOfEpisodes):
        state=env.reset()
        state = discrete_states(state, env)
        alpha = exploration_rate = get_rate(e)
        
        episode_reward = 0
        done = False
        while done is False:
            action = epsilon_greedy_policy(state, env, Q_table, exploration_rate)
            new_state, reward, terminated, truncated, _ = env.action_space(action)
            new_state = discrete_states(new_state, env)
            
            Q_table = update_Q(Q_table, state, action, reward, new_state, alpha, gamma)
            
            state = new_state
            episode_reward += reward
        total_reward.append(episode_reward)
    print("Finished Training.")
    return Q_table, total_reward
def update_Q(Q_table, state, action, reward, new_state, alpha, gamma):
    Q_table[state][action] = Q_table[state][action] + alpha * (reward + gamma * numpy.max(Q_table[new_state]) - Q_table[state][action])
    return Q_table

def epsilon_greedy_policy(state, env, Q_table, exploration_rate):
    """A Policy in which the agent chooses the action for maximum reward most of the time but occasionally takes a random action to learn

    Args:
        state: current state of the CartPole Environment
        env: CartPole Game Environment
        Q_table: Table that stores the determination of how good a particular action is at a state
        exploration_rate: A small value close to 0
    Return:
        Action for agent to take
    """
    if numpy.random.random() < exploration_rate:
        return env.action_space.sample()
    else:
        return numpy.argmax(Q_table[state]) #Choose action with maximum expected reward based on current state
    
def get_rate(e):
    return max(0.1, min(1., 1. - numpy.log10((e + 1) / 25.)))
    
def createDisplay(fake_display):
    if fake_display is False:
        display = Display(visible=0, size =(700,450))
        display.start()
        is_ipython = 'inline' in plt.get_backend()
        if is_ipython:
            from IPython import display
        plt.ion()
        fake_display=True

if __name__ == "__main__":
    main()