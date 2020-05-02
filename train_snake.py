from snake_world import Environment, Actions
from dqnsnake import DQNAgent
import matplotlib.pyplot as plt


def train_snake():

    numberOfCells = 10 # in each axis
    startingPosition = (4, 5) # head
    headDirection = 0 # NORTH
    foodPosition = (3, 6)
    max_steps_allowed = 1000

    #worldSize = 800 # only for visualization

    env = Environment(numberOfCells, worldSize=0)
    state_size = env.state_size #(numberOfCells x numberOfCells) # todo: not great that the state size is taken form the environment. It should be given somewhat more generically to the agent and the env.
    action_size = Actions.action_size # 3
    #print('state_size: ', state_size, 'action_size: ', action_size)
    agent = DQNAgent(state_size=state_size, action_size=action_size)

    episodes = 5
    decay = 0.9 / episodes * 2 # changes epsilon : explore vs exploit

    epochs = []
    losses = []
    steps_list = []

    fig1 = plt.figure('loss')
    ax1 = plt.gca()
    sc1, = ax1.plot(epochs, losses)
    sc1.set_marker('.')
    sc1.set_markerfacecolor('b')
    sc1.set_markeredgecolor('b')
    sc1.set_color('b')

    fig2 = plt.figure('steps')
    ax2 = plt.gca()
    sc2, = ax2.plot(epochs, steps_list)
    sc2.set_marker('.')
    sc2.set_markerfacecolor('r')
    sc2.set_markeredgecolor('r')
    sc2.set_color('r')

    for e in range(episodes):

        state = env.reset(startingPosition, headDirection, foodPosition)
        #print('state array reset: \n', state)

        agent.reset_convolutional_layers()
        full_state = agent.get_convolutional_layers(state)
        loss = 0.0
        steps = 0
        done = False

        episode_reward = 0

        #maxsteps = 3
        #debug_actions = [0, 1, 2, 2, 2, 1, 1, 1, 0, 0, 1, 2, 2, 2, 1, 1, 0, 0]

        #for step in range(maxsteps):
        while not done:
            #print('--- step: ', step)

            # state in this level is just a 2D array
            action = agent.get_action(full_state)
            #action = debug_actions[step]
            #action = agent.get_raction()
            #print('action chosen: ', action)

            # step to the next state
            next_state, reward, done = env.step(action)

            #print('state array after step ', step, ' : \n', next_state)
            #print('reward returned: ', reward)
            #print('next state: ', next_state)

            # we store the next_state in (1,H,W,C)
            full_next_state = agent.get_convolutional_layers(next_state)

            #print('full next state: \n:', full_next_state)

            assert(full_next_state.shape == (1, numberOfCells, numberOfCells, agent.numberOfLayers))

            # save S,A,R,S' to experience
            # full states are a snapshot - copies of the state
            agent.save_transition(full_state, action, reward, full_next_state, done)
            episode_reward += reward

            #if len(agent.experience) > 1:
            #    l = len(agent.experience)
            #    print(agent.experience[l - 1] - agent.experience[l - 2])


            # use alternative policy to train model - rely on experience only
            current_loss = agent.train()
            #print('current_loss: ', current_loss)
            loss += current_loss


            full_state = full_next_state


            # limit max steps - avoid something bad
            steps += 1
            if steps >= max_steps_allowed:
                done = True

            #if done:
            #    print('done!')
            #    break

        # next episode
        if agent.epsilon > 0.1:
            agent.epsilon -= decay # agent slowly reduces exploring

        #agent.print_memory

        print('episode: {:5d} steps: {:3d} epsilon: {:.3f} memory {:4d} loss: {:8.4f} reward: {:3d}'.format(e, steps, agent.epsilon, len(agent.experience), loss, episode_reward))

        plt.draw()
        epochs.append(e)
        losses.append(loss)
        sc1.set_data(epochs, losses)
        ax1.relim()
        ax1.autoscale_view()

        steps_list.append(steps)
        sc2.set_data(epochs, steps_list)
        ax2.relim()
        ax2.autoscale_view()

        plt.pause(0.01)

    plt.waitforbuttonpress()


if __name__ == "__main__":
    train_snake()
