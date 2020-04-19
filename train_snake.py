from snake_world import Environment, Actions
from dqnsnake import DQNAgent

def train_snake():

    numberOfCells = 10 # in each axis
    startingPosition = (4, 4) # head
    headDirection = 0 # NORTH
    foodPosition = (2, 1)

    #worldSize = 800 # only for visualization

    env = Environment(numberOfCells, worldSize=0)
    state_size = env.state_size #(numberOfCells x numberOfCells) # todo: not great that the state size is taken form the environment. It should be given somewhat more generically to the agent and the env.
    action_size = Actions.action_size # 3
    #print('state_size: ', state_size, 'action_size: ', action_size)
    agent = DQNAgent(state_size=state_size, action_size=action_size, head_starting_position=startingPosition) # builds network

    episodes = 1
    maxsteps = 500 # todo: use while
    decay = 0.9 / episodes * 2

    for e in range(episodes):

        state = env.reset(startingPosition, headDirection, foodPosition)
        #print('state array reset: \n', state)

        state = agent.get_channels(state)

        print('state before predict: ', state)
        print('dtype: ', state.dtype)

        for t in range(maxsteps):

            # state in this level is just a 2D array
            action = agent.get_action(state)

            print("step : ", t, 'action: ', action)

            # step to the next state
            next_state, reward, done = env.step(action)

            print('next state array after step: \n', next_state)

            #print('next state array after step: \n', next_state)
            print('reward returned: ', reward)

            # we store the next_state in (1,H,W,C)
            full_next_state = agent.get_channels(next_state)
            assert(full_next_state.shape == (1, numberOfCells, numberOfCells, agent.numberOfChannels))

            # save S,A,R,S' to experience
            agent.store_transition(state, action, reward, full_next_state, done)

            # use alternative policy to train model - rely on experience only
            #agent.train()

            state = full_next_state

            if done:

                if agent.epsilon > 0.01:
                    agent.epsilon -= decay

                #print('episode:', e, 'done in', t, 'steps', ' experience length:', len(agent.experience), " epsilon:", agent.epsilon)
                print('episode: {:5d} steps: {:3d} epsilon: {:.5f}'.format(e, t, agent.epsilon))
                #print(' ')
                #steps.append(t)
                break

            if t is maxsteps - 1:
                print('maxsteps reached!')
                #print('episode:', e, 'reached in', t, 'steps', ' experience length:', len(agent.experience), " epsilon:", agent.epsilon)
                print('episode: {:5d} steps: {:3d} epsilon: {:.5f}'.format(e, t, agent.epsilon))
                #print(' ')
                #steps.append(t)


    #epochs = np.arange(episodes)
    #plt.plot(epochs, steps)

    #plt.show()

if __name__ == "__main__":
    train_snake()
