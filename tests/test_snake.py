from dqnsnake.agent.dqnsnake import DQNAgent
from dqnsnake.agent.snake_world import Environment, Actions


def test_walk():
    numberOfCells = 10
    startingPosition = (4, 5)
    headDirection = 0 # NORTH
    foodPosition = (3, 6)

    env = Environment(numberOfCells, worldSize=0)
    env.reset(startingPosition, headDirection, foodPosition)

    debug_actions = [0, 1, 2, 2, 2, 1, 1, 1, 0, 0, 1, 2, 2, 2, 1, 1, 0, 0]

    for step in range(len(debug_actions) - 1):

        action = debug_actions[step]
        _, _, done = env.step(action)

        assert(not done)

    action = debug_actions[-1]
    _, _,  done = env.step(action)

    assert(done)


def test_deterministic_training():

    numberOfCells = 10 # in each axis
    startingPosition = (4, 5) # head
    headDirection = 0 # NORTH
    foodPosition = (3, 6)

    env = Environment(numberOfCells, worldSize=0)
    state_size = env.state_size #(numberOfCells x numberOfCells) # todo: not great that the state size is taken form the environment. It should be given somewhat more generically to the agent and the env.
    action_size = Actions.action_size # 3
    #print('state_size: ', state_size, 'action_size: ', action_size)
    agent = DQNAgent(state_size=state_size, action_size=action_size, deterministic=True)

    state = env.reset(startingPosition, headDirection, foodPosition)
    agent.reset_convolutional_layers()
    full_state = agent.get_convolutional_layers(state)
    loss30 = -1
    loss50 = -1
    action30 = -1
    action50 = -1
    done = False

    maxsteps = 50

    for step in range(maxsteps):
        action = agent.get_exploration_action()
        next_state, reward, done = env.step(action)
        full_next_state = agent.get_convolutional_layers(next_state)
        assert(full_next_state.shape == (1, numberOfCells, numberOfCells, agent.numberOfLayers))
        agent.save_transition(full_state, action, reward, full_next_state, done)
        current_loss = agent.train(determinitic=True)

        if (step == 29):
            action30 = action
            loss30 = current_loss

        full_state = full_next_state

        # on step 10 it touches the wall - it does not break because we don't check that
        # that's why if you test it with deterministic == False, you get index out of range error in the state
        #if (step >= 10):
        #    assert(done)

    loss50 = current_loss
    action50 = action

    assert(loss30 == 0.009624761529266834) # is this accurate generally?
    assert(loss50 == 0.002301788656041026)
    assert(action30 == 1)
    assert(action50 == 0)

def test_deterministic_multiepisode_training():

    numberOfCells = 10 # in each axis
    startingPosition = (4, 5) # head
    headDirection = 0 # NORTH
    foodPosition = (3, 6)

    env = Environment(numberOfCells, worldSize=0)
    state_size = env.state_size #(numberOfCells x numberOfCells) # todo: not great that the state size is taken form the environment. It should be given somewhat more generically to the agent and the env.
    action_size = Actions.action_size # 3
    agent = DQNAgent(state_size=state_size, action_size=action_size, deterministic=True)

    losses = [-1, -1, -1, -1]
    done = False

    episodes = 4
    maxsteps = 9

    for e in range(episodes):

        state = env.reset(startingPosition, headDirection, foodPosition)
        agent.reset_convolutional_layers()
        full_state = agent.get_convolutional_layers(state)
        loss = 0

        for step in range(maxsteps):
            action = agent.get_exploration_action()
            next_state, reward, done = env.step(action)
            full_next_state = agent.get_convolutional_layers(next_state)
            assert(full_next_state.shape == (1, numberOfCells, numberOfCells, agent.numberOfLayers))
            agent.save_transition(full_state, action, reward, full_next_state, done)

            current_loss = agent.train(determinitic=True)
            loss += current_loss

            full_state = full_next_state

        losses[e] = loss

    assert(losses[0] == 3.9530042931437492)
    assert(losses[1] == 0.044061796041205525) # hmmm
    assert(losses[2] == 0.1294256257242523)
    assert(losses[3] == 2.783756474032998)
