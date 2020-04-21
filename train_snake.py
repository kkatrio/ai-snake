from snake_world import Environment, Actions
from dqnsnake import DQNAgent

def train_snake():

    numberOfCells = 10 # in each axis
    startingPosition = (4, 4) # head
    headDirection = 0 # NORTH
    foodPosition = (3, 6)
    max_steps_allowed = 1000

    #worldSize = 800 # only for visualization

    env = Environment(numberOfCells, worldSize=0)
    state_size = env.state_size #(numberOfCells x numberOfCells) # todo: not great that the state size is taken form the environment. It should be given somewhat more generically to the agent and the env.
    action_size = Actions.action_size # 3
    #print('state_size: ', state_size, 'action_size: ', action_size)
    agent = DQNAgent(state_size=state_size, action_size=action_size)

    episodes = 50
    #maxsteps = 200
    decay = 0.9 / episodes * 2 # changes epsilon : explore vs exploit

    for e in range(episodes):

        state = env.reset(startingPosition, headDirection, foodPosition)
        #print('state array reset: \n', state)

        state = agent.get_channels(state)
        loss = 0.0
        step = 0
        done = False

        #for t in range(maxsteps):
        while not done:
            #print('--- step: ', step)

            # state in this level is just a 2D array
            action = agent.get_action(state)

            #print('action chosen: ', action)

            # step to the next state
            next_state, reward, done = env.step(action)

            #print('state array after step ', step, ' : \n', next_state)
            #print('reward returned: ', reward)

            # we store the next_state in (1,H,W,C)
            full_next_state = agent.get_channels(next_state)
            assert(full_next_state.shape == (1, numberOfCells, numberOfCells, agent.numberOfChannels))

            # save S,A,R,S' to experience
            agent.store_transition(state, action, reward, full_next_state, done)

            # use alternative policy to train model - rely on experience only
            loss += agent.train()
            #print('loss after each step: ', loss)

            state = full_next_state

            # limit max steps - avoid something bad
            step += 1
            if step >= max_steps_allowed:
                done = True

        # next episode
        if agent.epsilon > 0.1:
            agent.epsilon -= decay # agent slowly reduces exploring

        print('episode: {:5d} step: {:3d} epsilon: {:.5f} loss: {:8.4f}'.format(e, step, agent.epsilon, loss))

    #epochs = np.arange(episodes)
    #plt.plot(epochs, steps)

    #plt.show()

if __name__ == "__main__":
    train_snake()
