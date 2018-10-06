#######################################################################
# Copyright (C)                                                       #
# 2016 Shangtong Zhang(zhangshangtong.cpp@gmail.com)                  #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Nicky van Foreest(vanforeest@gmail.com)                        #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# actions: hit or stand
ACTION_HIT = 0
ACTION_STAND = 1  #  "strike" in the book
actions = [ACTION_HIT, ACTION_STAND]

# policy for player
# policyPlayer = np.zeros(22)
# for i in range(12, 20):
#     policyPlayer[i] = ACTION_HIT
# policyPlayer[20] = ACTION_STAND
# policyPlayer[21] = ACTION_STAND

# # function form of target policy of player
# def targetPolicyPlayer(usableAcePlayer, playerSum, dealerCard):
#     return policyPlayer[playerSum]

# function form of behavior policy of player
def behaviorPolicyPlayer(usableAcePlayer, playerSum, dealerCard):
    if np.random.binomial(1, 0.5) == 1:
        return ACTION_STAND
    return ACTION_HIT

# policy for dealer
policyDealer = np.zeros(22)
for i in range(12, 17):
    policyDealer[i] = ACTION_HIT
for i in range(17, 22):
    policyDealer[i] = ACTION_STAND

# get a new card
def getCard():
    card = np.random.randint(1, 14)
    card = min(card, 10)
    return card

# play a game
# @policyPlayerFn: specify policy for player
# @initialState: [whether player has a usable Ace, sum of player's cards, one card of dealer]
# @initialAction: the initial action
def play(policyPlayerFn, initialState=None, initialAction=None):
    # player status

    # sum of player
    playerSum = 0

    # trajectory of player
    playerTrajectory = []

    # whether player uses Ace as 11
    usableAcePlayer = False

    # dealer status
    dealerCard1 = 0
    dealerCard2 = 0
    usableAceDealer = False

    if initialState is None:
        # generate a random initial state

        numOfAce = 0

        # initialize cards of player
        while playerSum < 12:
            # if sum of player is less than 12, always hit
            card = getCard()

            # if get an Ace, use it as 11
            if card == 1:
                numOfAce += 1
                card = 11
                usableAcePlayer = True
            playerSum += card

        # if player's sum is larger than 21, he must hold at least one Ace, two Aces are possible
        if playerSum > 21:
            # use the Ace as 1 rather than 11
            playerSum -= 10

            # if the player only has one Ace, then he doesn't have usable Ace any more
            if numOfAce == 1:
                usableAcePlayer = False

        # initialize cards of dealer, suppose dealer will show the first card he gets
        dealerCard1 = getCard()
        dealerCard2 = getCard()

    else:
        # use specified initial state
        usableAcePlayer = initialState[0]
        playerSum = initialState[1]
        dealerCard1 = initialState[2]
        dealerCard2 = getCard()

    # initial state of the game
    state = [usableAcePlayer, playerSum, dealerCard1]

    # initialize dealer's sum
    dealerSum = 0
    if dealerCard1 == 1 and dealerCard2 != 1:
        dealerSum += 11 + dealerCard2
        usableAceDealer = True
    elif dealerCard1 != 1 and dealerCard2 == 1:
        dealerSum += dealerCard1 + 11
        usableAceDealer = True
    elif dealerCard1 == 1 and dealerCard2 == 1:
        dealerSum += 1 + 11
        usableAceDealer = True
    else:
        dealerSum += dealerCard1 + dealerCard2

    # game starts!

    # player's turn
    while True:
        if initialAction is not None:
            action = initialAction
            initialAction = None
        else:
            # get action based on current sum
            action = policyPlayerFn(usableAcePlayer, playerSum, dealerCard1)

        # track player's trajectory for importance sampling
        playerTrajectory.append([action, (usableAcePlayer, playerSum, dealerCard1)])

        if action == ACTION_STAND:
            break
        # if hit, get new card
        playerSum += getCard()

        # player busts
        if playerSum > 21:
            # if player has a usable Ace, use it as 1 to avoid busting and continue
            if usableAcePlayer == True:
                playerSum -= 10
                usableAcePlayer = False
            else:
                # otherwise player loses
                return state, -1, playerTrajectory

    # dealer's turn
    while True:
        # get action based on current sum
        action = policyDealer[dealerSum]
        if action == ACTION_STAND:
            break
        # if hit, get a new card
        card = getCard()
        if (card == 1):
            card = 11
            usableAceDealer = True
        dealerSum += card
        # dealer busts
        if dealerSum > 21:
            if usableAceDealer == True:
            # if dealer has a usable Ace, use it as 1 to avoid busting and continue
                dealerSum -= 10
                usableAceDealer = False
            else:
            # otherwise dealer loses
                return state, 1, playerTrajectory

    # compare the sum between player and dealer
    if playerSum > dealerSum:
        return state, 1, playerTrajectory
    elif playerSum == dealerSum:
        return state, 0, playerTrajectory
    else:
        return state, -1, playerTrajectory

# Monte Carlo with Exploring Starts
def monteCarloES(nEpisodes):
    # (playerSum, dealerCard, usableAce, action)
    stateActionValues = np.zeros((10, 10, 2, 2))
    # initialze counts to 1 to avoid division by 0
    stateActionPairCount = np.ones((10, 10, 2, 2))

    # behavior policy is greedy
    def behaviorPolicy(usableAce, playerSum, dealerCard):
        usableAce = int(usableAce)
        playerSum -= 12
        dealerCard -= 1
        # get argmax of the average returns(s, a)
        return np.argmax(stateActionValues[playerSum, dealerCard, usableAce, :]
                      / stateActionPairCount[playerSum, dealerCard, usableAce, :])

    # play for several episodes
    for episode in range(nEpisodes):
        if episode % 100000 == 0:
            print('episode:', episode)
        # for each episode, use a randomly initialized state and action
        initialState = [bool(np.random.choice([0, 1])),
                       np.random.choice(range(12, 22)),
                       np.random.choice(range(1, 11))]
        initialAction = np.random.choice(actions)
        _, reward, trajectory = play(behaviorPolicy, initialState, initialAction)
        for action, (usableAce, playerSum, dealerCard) in trajectory:
            usableAce = int(usableAce)
            playerSum -= 12
            dealerCard -= 1
            # update values of state-action pairs
            stateActionValues[playerSum, dealerCard, usableAce, action] += reward
            stateActionPairCount[playerSum, dealerCard, usableAce, action] += 1

    return stateActionValues / stateActionPairCount

# Figure 5.3
def figure5_3():
    stateActionValues = monteCarloES(500000)
    stateValueUsableAce = np.zeros((10, 10))
    stateValueNoUsableAce = np.zeros((10, 10))
    # get the optimal policy
    actionUsableAce = np.zeros((10, 10), dtype='int')
    actionNoUsableAce = np.zeros((10, 10), dtype='int')
    for i in range(10):
        for j in range(10):
            stateValueNoUsableAce[i, j] = np.max(stateActionValues[i, j, 0, :])
            actionNoUsableAce[i, j] = np.argmax(stateActionValues[i, j, 0, :])
    print('No usable ace')
    print(stateValueNoUsableAce)
    print(actionNoUsableAce)

    for i in range(10):
        for j in range(10):
            stateValueUsableAce[i, j] = np.max(stateActionValues[i, j, 1, :])
            actionUsableAce[i, j] = np.argmax(stateActionValues[i, j, 1, :])
    print('Usable ace')
    print(stateValueUsableAce)
    print(actionUsableAce)


if __name__ == "__main__":
    # onPolicy()
    figure5_3()
    # offPolicy()

