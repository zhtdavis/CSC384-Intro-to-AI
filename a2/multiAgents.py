# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random

import util
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        oldFood = currentGameState.getFood()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        height = newFood.height
        width = newFood.width

        for ghostPos in successorGameState.getGhostPositions():
            dist = abs(ghostPos[0]-newPos[0])+abs(ghostPos[1]-newPos[1])
            if dist <= 2:
                return 0

        distance2Food = 0
        closetFood = float("inf")
        for i in range(width):
            for j in range(height):
                if newFood[i][j] == True:
                    dist = abs(newPos[0]-i)+abs(newPos[1]-j)
                    distance2Food += dist
                    closetFood = min(closetFood, dist)

        if distance2Food == 0:
            return currentGameState.getScore()*100

        ans = 1/distance2Food+1/closetFood
        if oldFood != newFood:
            ans += 1
        return ans

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        _, action = self.helper(gameState, 0, 0)
        return action

    def helper(self, gameState, agent, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), ""

        if agent == gameState.getNumAgents()-1:
            depth = depth+1

        if agent == 0:
            ansScore = -float("inf")
            ansAction = ""
            successorActions = gameState.getLegalActions(agent)
            for action in successorActions:
                successorState = gameState.generateSuccessor(agent, action)
                score, _ = self.helper(successorState, (agent+1) % gameState.getNumAgents(), depth)
                if score > ansScore:
                    ansScore = score
                    ansAction = action
            return ansScore, ansAction
        else:
            ansScore = float("inf")
            ansAction = ""
            successorActions = gameState.getLegalActions(agent)
            for action in successorActions:
                successorState = gameState.generateSuccessor(agent, action)
                score, _ = self.helper(successorState, (agent+1) % gameState.getNumAgents(), depth)
                if score < ansScore:
                    ansScore = score
                    ansAction = action
            return ansScore, ansAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        _, action = self.helper(gameState, 0, 0, -float("inf"), float("inf"))
        return action

    def helper(self, gameState, agent, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), ""

        if agent == gameState.getNumAgents()-1:
            depth = depth+1

        if agent == 0:
            ansAction = ""
            successorActions = gameState.getLegalActions(agent)
            for action in successorActions:
                successorState = gameState.generateSuccessor(agent, action)
                score, _ = self.helper(successorState, (agent+1) % gameState.getNumAgents(), depth, alpha, beta)
                if score > alpha:
                    alpha = score
                    ansAction = action
                if alpha >= beta:
                    break
            return alpha, ansAction
        else:
            ansAction = ""
            successorActions = gameState.getLegalActions(agent)
            for action in successorActions:
                successorState = gameState.generateSuccessor(agent, action)
                score, _ = self.helper(successorState, (agent+1) % gameState.getNumAgents(), depth, alpha, beta)
                if score < beta:
                    beta = score
                    ansAction = action
                if alpha >= beta:
                    break
            return beta, ansAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        _, action = self.helper(gameState, 0, 0)
        return action

    def helper(self, gameState, agent, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), ""

        if agent == gameState.getNumAgents()-1:
            depth = depth+1

        if agent == 0:
            ansScore = -float("inf")
            ansAction = ""
            successorActions = gameState.getLegalActions(agent)
            for action in successorActions:
                successorState = gameState.generateSuccessor(agent, action)
                score, _ = self.helper(successorState, (agent+1) % gameState.getNumAgents(), depth)
                if score > ansScore:
                    ansScore = score
                    ansAction = action
            return ansScore, ansAction
        else:
            minScore = float("inf")
            ansAction = ""
            sumScore = 0
            successorActions = gameState.getLegalActions(agent)
            for action in successorActions:
                successorState = gameState.generateSuccessor(agent, action)
                score, _ = self.helper(successorState, (agent+1) % gameState.getNumAgents(), depth)
                sumScore += score
                if score < minScore:
                    minScore = score
                    ansAction = action
            return sumScore/len(successorActions), ansAction


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    food = currentGameState.getFood()
    height = food.height
    width = food.width
    pacmanPos = currentGameState.getPacmanPosition()
    scaredTimes = [ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()]

    scared = False
    for time in scaredTimes:
        if time > 0:
            scared = True

    distance2Food = 0
    for i in range(width):
        for j in range(height):
            if food[i][j] == True:
                dist = abs(pacmanPos[0]-i)+abs(pacmanPos[1]-j)
                distance2Food += dist

    if distance2Food == 0:
        return currentGameState.getScore()*100

    distance2Ghost = 0
    for ghostPos in currentGameState.getGhostPositions():
        dist = abs(pacmanPos[0]-ghostPos[0])+abs(pacmanPos[1]-ghostPos[1])
        distance2Ghost += dist

    if distance2Ghost == 0 and scared:
        return currentGameState.getScore()*100
    elif distance2Ghost == 0 and not scared:
        return 0

    # 6/6
    ans = 1000/distance2Food+currentGameState.getScore()/2
    if scared:
        ans += 100/distance2Ghost
    else:
        ans += 10/distance2Ghost
    return ans


# Abbreviation
better = betterEvaluationFunction
