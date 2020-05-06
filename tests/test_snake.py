from dqnsnake.agent.dqnsnake import DQNAgent
from dqnsnake.agent.snake_world import Environment, Actions


def test_walk():
    numberOfCells = 10
    startingPosition = (4, 5)
    foodPosition = (3, 6)

    env = Environment(numberOfCells, worldSize=0)
    env.reset(startingPosition, foodPosition)

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
    foodPosition = (3, 6)

    env = Environment(numberOfCells, worldSize=0)
    agent = DQNAgent(state_size=env.state_size, action_size=Actions.action_size, deterministic=True, batch_size=24, memory_limit=2000) # todo: tf summary
    state = env.reset(startingPosition, foodPosition)
    agent.reset_convolutional_layers()
    full_state = agent.get_convolutional_layers(state)
    loss30 = -1
    loss50 = -1
    action30 = -1
    action50 = -1

    maxsteps = 50

    for step in range(maxsteps):
        action = agent.get_exploration_action()
        next_state, reward, done = env.step(action, food_regeneration=False, food_position=(1, 1))
        full_next_state = agent.get_convolutional_layers(next_state)
        assert(full_next_state.shape == (1, numberOfCells, numberOfCells, agent.numberOfLayers))
        agent.save_transition(full_state, action, reward, full_next_state, done)
        current_loss = agent.train()

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

    assert(loss30 == 0.009624761529266834) # is this accurate really?
    assert(loss50 == 0.002301788656041026)
    assert(action30 == 1)
    assert(action50 == 0)

def test_deterministic_multiepisode_training():

    numberOfCells = 10 # in each axis
    startingPosition = (4, 5) # head
    foodPosition = (3, 6)

    env = Environment(numberOfCells, worldSize=0)
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
            next_state, reward, done = env.step(action, food_regeneration=False, food_position=(1, 1))
            full_next_state = agent.get_convolutional_layers(next_state)
            assert(full_next_state.shape == (1, numberOfCells, numberOfCells, agent.numberOfLayers))
            agent.save_transition(full_state, action, reward, full_next_state, done)

            current_loss = agent.train()
            loss += current_loss

            full_state = full_next_state

        losses[e] = loss

    assert(losses[0] == 3.9530042931437492)
    assert(losses[1] == 0.044061796041205525) # hmmm
    assert(losses[2] == 0.1294256257242523)
    assert(losses[3] == 2.783756474032998)

def test_food_regeneration():

    numberOfCells = 10 # in each axis
    startingPosition = (4, 5) # head
    foodPosition = (3, 6)

    env = Environment(numberOfCells, worldSize=0)
    state = env.reset(startingPosition, foodPosition)
    env.regenerate_food((8, 8))
    assert(env[(8, 8)] == 1)

    for i in range(200):
        env.regenerate_food()
    print(env._cellType)
    assert(env[startingPosition] == 2)
    assert(env[(5, 5)] == 3)
    assert(env[(6, 5)] == 3)

def test_mechanics():
    numberOfCells = 10 # in each axis
    startingPosition = (2, 2) # head
    foodPositions = [(2, 7), (7, 7), (7, 2), (2, 2)]

    env = Environment(numberOfCells, worldSize=0)
    state = env.reset(startingPosition, foodPositions[0])

    for i in range(20):
        if (i % 5 == 0):
            action = 2
            env.regenerate_food(foodPositions[i % 4])
        else:
            action = 0
        state, reward, done = env.step(action, food_regeneration=False)
        print(state)
        print('reward: ', reward, ' snake size: ', env.snake.size)

    assert(env.snake.size == 7)


def test_growing():

    numberOfCells = 10 # in each axis
    startingPosition = (2, 2) # head
    foodPositions = [(3, 1), (3, 3), (5, 3), (6, 1), (9, 9), (9, 9)] # last two values are dummy, just to satisfy the indices below

    env = Environment(numberOfCells, worldSize=0)
    agent = DQNAgent(state_size=env.state_size, action_size=Actions.action_size, deterministic=True, batch_size=24, memory_limit=2000)

    episodes = 2
    food_index = 0
    step = 0
    for e in range(episodes):

        agent.reset_convolutional_layers()
        state = env.reset(startingPosition, foodPositions[food_index])
        full_state = agent.get_convolutional_layers(state)
        done = False
        loss = 0
        previous_size = 3 # keep track of snake size
        while not done:

            #print('\nfood_index: ', food_index, ' if we eat, we will place the next fruit in: ', foodPositions[food_index + 1])
            action = agent.get_action(state)
            next_state, reward, done = env.step(action, food_regeneration=False, food_position=foodPositions[food_index + 1])

            print('head is at: ', env.snake.head, '  snake size: ', reward)

            if env.snake.size > previous_size: # eaten
                previous_size = env.snake.size
                food_index += 1

            full_next_state = agent.get_convolutional_layers(next_state)
            agent.save_transition(full_state, action, reward, full_next_state, done)
            current_loss = agent.train()
            loss += current_loss
            #print('loss: ', loss)

            full_state = full_next_state

            if step == 1:
                assert(env.snake.size == 4)
            if step == 6:
                assert(env.snake.size == 4)
            if step == 8:
                assert(env.snake.size == 5)
            if step == 11:
                assert(env.snake.size == 6)
            if step == 12:
                assert(loss == 20.45940124988556)

            step += 1
