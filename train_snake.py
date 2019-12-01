
if __name__ == "__main__":
    #env = gym.make('CartPole-v1')
    Environment env

    

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    steps = []
    episodes = 100
    maxsteps = 300
    agent = Agent(state_size, action_size)

    for e in range(episodes):

        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for t in range(maxsteps):

            #env.render()

            action = agent.get_action(state)

            # step to the next state
            next_state, reward, done, _ = env.step(action)

            # calculate reward from this step
            reward = reward if not done else -100
            #rewards.append(reward)

            next_state = np.reshape(next_state, [1, state_size])

            # save S,A,R,S' to memory
            agent.store_transition(state, action, reward, next_state, done)

            # use alternative policy to train model
            if(len(agent.memory)) >= agent.batch_size:
                agent.replay()

            state = next_state

            if done:
                print('episode:', e, 'done in', t, 'steps', ' memory length:', len(agent.memory), " epsilon:", agent.epsilon)
                #steps.append(t)
                break

            if t is maxsteps - 1:
                print('maxsteps reached!')
                print('episode:', e, 'reached in', t, 'steps', ' memory length:', len(agent.memory), " epsilon:", agent.epsilon)
                #steps.append(t)


    #epochs = np.arange(episodes)
    #plt.plot(epochs, steps)

    #plt.show()
