from dqnsnake.agent.snake_world import Environment, Actions
from dqnsnake.agent.dqnsnake import DQNAgent
import matplotlib.pyplot as plt


def train_snake():

    numberOfCells = 10 # in each axis
    startingPosition = (4, 5) # head
    foodPosition = (3, 6)
    max_steps_allowed = 1000

    #worldSize = 800 # only for visualization

    env = Environment(numberOfCells, worldSize=0)
    state_size = env.state_size #(numberOfCells x numberOfCells) # todo: not great that the state size is taken form the environment. It should be given somewhat more generically to the agent and the env.
    action_size = Actions.action_size # 3
    agent = DQNAgent(state_size=state_size, action_size=action_size, batch_size=3, memory_limit=2000)

    episodes = 4
    decay = 0.9 / episodes * 2 # changes epsilon : explore vs exploit

    epochs = []
    losses = []
    steps_list = []

    with open('training_data', 'w') as f:

        for e in range(episodes):

            state = env.reset(startingPosition)
            #print('state array reset: \n', state)

            agent.reset_convolutional_layers()
            full_state = agent.get_convolutional_layers(state)
            loss = 0.0
            steps = 0
            done = False

            episode_reward = 0

            #for step in range(maxsteps):
            while not done:

                # state in this level is just a 2D array
                action = agent.get_action(full_state)
                #action = debug_actions[step]
                #action = agent.get_raction()
                #print('action chosen: ', action)

                # step to the next state
                next_state, reward, done = env.step(action)

                #print('state array after step ', steps, ' : \n', next_state)
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

            print('episode: {:5d} steps: {:3d} epsilon: {:.3f} loss: {:8.4f} reward: {:3d} fruits: {:2d}'.format(e, steps, agent.epsilon, loss, episode_reward, env.fruits_eaten))
            f.write('{:5d} {:3d} {:8.4f} {:4d} {:2d}\n'.format(e, steps, loss, episode_reward, env.fruits_eaten))

        agent.model.save('trained_snake.model')


if __name__ == "__main__":
    train_snake()
