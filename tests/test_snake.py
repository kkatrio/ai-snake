from dqn.agent import DQNAgent
from dqn.snake_world import Environment, Actions


def test_walk():
    numberOfCells = 10
    startingPosition = (4, 5)
    foodPosition = (3, 6)

    env = Environment(numberOfCells)
    env.reset(startingPosition, foodPosition)

    debug_actions = [0, 1, 2, 2, 2, 1, 1, 2, 2, 0, 1, 2, 2, 0, 1, 1, 0, 0]

    for step in range(len(debug_actions) - 1):

        action = debug_actions[step]
        print('action: ', action)
        state, _, done = env.step(action)
        if done:
            print('done!')
        print(state)

        assert(not done)

    action = debug_actions[-1]
    _, _,  done = env.step(action)

    assert(done)

def test_smoke():
    # just runs the code - no assetions

    numberOfCells = 10 # in each axis
    startingPosition = (4, 5) # head
    foodPosition = (3, 6)

    env = Environment(numberOfCells)
    agent = DQNAgent(state_size=env.state_size, action_size=Actions.action_size, deterministic=True, batch_size=24, memory_limit=2000)
    state = env.reset(startingPosition, foodPosition)
    agent.reset_convolutional_layers()
    full_state = agent.get_convolutional_layers(state)

    maxsteps = 2

    for step in range(maxsteps):
        action = agent.get_exploration_action()
        next_state, reward, done = env.step(action, food_position=(1, 1))
        full_next_state = agent.get_convolutional_layers(next_state)
        assert(full_next_state.shape == (1, numberOfCells, numberOfCells, agent.numberOfLayers))
        agent.save_transition(full_state, action, reward, full_next_state, done)
        current_loss = agent.train()

        if (step == 0):
            action1 = action
            loss1 = current_loss

        full_state = full_next_state

    loss2 = current_loss
    action2 = action
    #assert(False)

def test_single_training():

    numberOfCells = 10 # in each axis
    startingPosition = (4, 5) # head
    foodPosition = (3, 6)

    env = Environment(numberOfCells, deterministic=True)
    agent = DQNAgent(state_size=env.state_size, action_size=Actions.action_size, deterministic=True, batch_size=24, memory_limit=2000)
    state = env.reset(startingPosition, foodPosition)
    agent.reset_convolutional_layers()
    full_state = agent.get_convolutional_layers(state)
    loss10 = -1
    action10 = -1

    maxsteps = 10

    for step in range(maxsteps):
        action = agent.get_exploration_action()
        next_state, reward, done = env.step(action, food_position=(1, 1))
        assert(not done)
        full_next_state = agent.get_convolutional_layers(next_state)
        assert(full_next_state.shape == (1, numberOfCells, numberOfCells, agent.numberOfLayers))
        agent.save_transition(full_state, action, reward, full_next_state, done)
        current_loss = agent.train()
        full_state = full_next_state

    loss10 = current_loss
    action10 = action

    assert(loss10 == 0.006804642267525196)
    assert(action10 == 0)

def test_multiepisode_training():

    numberOfCells = 10 # in each axis
    startingPosition = (4, 5) # head
    foodPosition = (3, 6)

    env = Environment(numberOfCells, deterministic=True)
    state_size = env.state_size
    action_size = Actions.action_size # 3
    agent = DQNAgent(state_size=state_size, action_size=action_size, deterministic=True, batch_size=24, memory_limit=2000)

    losses = [-1, -1, -1, -1]
    done = False

    episodes = 4
    maxsteps = 9

    for e in range(episodes):

        state = env.reset(startingPosition, foodPosition)
        agent.reset_convolutional_layers()
        full_state = agent.get_convolutional_layers(state)
        loss = 0

        for step in range(maxsteps):
            action = agent.get_exploration_action()
            next_state, reward, done = env.step(action, food_position=(1, 1)) # generation on (1, 1) happens once over the test
            full_next_state = agent.get_convolutional_layers(next_state)
            assert(full_next_state.shape == (1, numberOfCells, numberOfCells, agent.numberOfLayers))
            agent.save_transition(full_state, action, reward, full_next_state, done)

            current_loss = agent.train()
            loss += current_loss

            full_state = full_next_state

        losses[e] = loss

    assert(losses[0] == 3.9618697417899966)
    assert(losses[1] == 0.044194952584803104)
    assert(losses[2] == 0.1333141174982302)
    assert(losses[3] == 2.834151452407241)

def test_food_regeneration():

    numberOfCells = 10 # in each axis
    startingPosition = (4, 5) # head
    foodPosition = (3, 6)

    env = Environment(numberOfCells)
    state = env.reset(startingPosition, foodPosition)
    env.regenerate_food((8, 8))
    assert(env[(8, 8)] == 1)

    for i in range(59): # plus the (8, 8) = 60
        env.regenerate_food()
    print(env._cells)
    assert(env[startingPosition] == 2)
    assert(env[(5, 5)] == 3)
    assert(env[(6, 5)] == 3)
