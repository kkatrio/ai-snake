from snake_world import Environment, Actions
from dqnsnake import DQNAgent

def train_snake():

    numberOfCells = 6 # in each axis
    startingPosition = (1, 1)
    headDirection = 2 # SOUTH
    foodPosition = (2, 1)

    #worldSize = 800 # only for visualization

    env = Environment(numberOfCells, worldSize=0)
    state_size = env.state_size #(numberOfCells x numberOfCells)
    action_size = Actions.action_size # 3
    #print('state_size: ', state_size, 'action_size: ', action_size)
    agent = DQNAgent(state_size=state_size, action_size=action_size, head_starting_position=startingPosition) # builds network

    episodes = 50
    maxsteps = 100

    for e in range(episodes):

        #print("--------------")
        #print("episode: ", e)
        #print("--------------")
        state = env.reset(startingPosition, headDirection, foodPosition)
        #print('state shape: ', state.shape)
        #state = np.reshape(state, [1, state_size])

        for t in range(maxsteps):

            #print("--------------")
            #print("step : ", t)
            #print("--------------")
            
            action = agent.get_action(state)

            # step to the next state
            next_state, reward, done = env.step(action)

            # calculate reward from this step
            reward = reward if not done else -100
            #rewards.append(reward)

            #next_state = np.reshape(next_state, [1, state_size])

            # save S,A,R,S' to experience
            agent.store_transition(state, action, reward, next_state, done)

            # use alternative policy to train model
            agent.train()

            state = next_state

            if done:
                print('episode:', e, 'done in', t, 'steps', ' experience length:', len(agent.experience), " epsilon:", agent.epsilon)
                #print(' ')
                #steps.append(t)
                break

            if t is maxsteps - 1:
                print('maxsteps reached!')
                print('episode:', e, 'reached in', t, 'steps', ' experience length:', len(agent.experience), " epsilon:", agent.epsilon)
                #print(' ')
                #steps.append(t)


    #epochs = np.arange(episodes)
    #plt.plot(epochs, steps)

    #plt.show()

if __name__ == "__main__":
    train_snake()
