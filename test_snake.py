from snake_world import Environment, Actions
import random

def test_snake():

    numberOfCells = 10 # in each axis
    startingPosition = (4, 4) # head
    headDirection = 0 # NORTH
    foodPosition = (3, 6)

    #worldSize = 800 # only for visualization

    env = Environment(numberOfCells, worldSize=0)
    action_size = Actions.action_size # 3
    #print('state_size: ', state_size, 'action_size: ', action_size)

    episodes = 30000
    #maxsteps = 200
    decay = 0.9 / episodes * 2 # changes epsilon : explore vs exploit

    for e in range(episodes):

        step = 0
        print('episode: ', e)

        state = env.reset(startingPosition, headDirection, foodPosition)
        #print('state array reset: \n', state)

        done = False

        #for t in range(maxsteps):
        while not done:

            action = random.randrange(action_size)
            #print('action chosen: ', action)

            # step to the next state
            next_state, reward, done = env.step(action)

            #print('state array after step ', step, ' : \n', next_state)
            #print('reward returned: ', reward)

            state = next_state
            step += 1

        print('steps done: ', step)


if __name__ == "__main__":
    test_snake()
