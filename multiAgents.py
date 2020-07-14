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
import math

from util import manhattanDistance
from game import Directions
import random, util, math, collections

from game import Agent


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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    """
    This method provides the method for evaluating a game state after a specific action. 
    Here it always returns the worst possible outcome (negative infinity) if the action is 'stop' or if 
    you are in the same position as a ghost (where the ghost is not scared of you). 
    For the remaining possibilities the negative manhattan distance of the nearest food is returned. So 
    the closer the food, the better.
    """

    def evaluationFunction(self, currentGameState, action):
        # you'd never receive an advantage, no chance at receiving pallets etc.
        if action == 'Stop':
            return float("-inf")

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newGhostStates = successorGameState.getGhostStates()

        curfood = currentGameState.getFood()
        foodList = curfood.asList()

        for state in newGhostStates:
            if state.getPosition() == newPos and (state.scaredTimer == 0):
                return float("-inf")

        foodDistance = float("-inf")
        for foodPos in foodList:
            foodDistance = max(foodDistance, -manhattanDistance(newPos, foodPos))

        return foodDistance


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        totalAgents = gameState.getNumAgents()

        def isOver(gameState, depth):
            return depth == self.depth or gameState.isLose() or gameState.isWin()

        def maxAgent(gameState, index, depth):
            if isOver(gameState, depth):
                return self.evaluationFunction(gameState)

            score = -math.inf
            for action in gameState.getLegalActions(index):
                gs = gameState.generateSuccessor(index, action)

                follow = 0
                # only if pacman itself plays the game
                if index + 1 == totalAgents:
                    follow = maxAgent(gs, 0, depth + 1)
                else:
                    follow = minAgent(gs, index + 1, depth)

                if follow > score:
                    score = follow

            return score

        def minAgent(gameState, index, depth):
            if isOver(gameState, depth):
                return self.evaluationFunction(gameState)

            score = math.inf
            for action in gameState.getLegalActions(index):
                gs = gameState.generateSuccessor(index, action)

                follow = 0
                if index + 1 == totalAgents:
                    follow = maxAgent(gs, 0, depth + 1)
                else:
                    follow = minAgent(gs, index + 1, depth)

                if follow < score:
                    score = follow

            return score

        bestAction = None

        # NOTE: self.index is always, ALWAYS 0.
        # The following code will reflect this.
        score = -math.inf
        for action in gameState.getLegalActions(self.index):
            # GameState, if this specific action gets taken
            gs = gameState.generateSuccessor(self.index, action)

            follow = minAgent(gameState=gs, index=self.index + 1, depth=0)

            if follow > score:
                score = follow
                bestAction = action
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        totalAgents = gameState.getNumAgents()

        def isOver(gameState, depth):
            return depth == self.depth or gameState.isLose() or gameState.isWin()

        def maxAgent(gameState, index, depth, alpha, beta):
            if isOver(gameState, depth):
                return self.evaluationFunction(gameState)

            score = -math.inf
            for action in gameState.getLegalActions(index):
                gs = gameState.generateSuccessor(index, action)

                follow = 0
                # only if pacman itself plays the game
                if index + 1 == totalAgents:
                    follow = maxAgent(gs, 0, depth + 1, alpha, beta)
                else:
                    follow = minAgent(gs, index + 1, depth, alpha, beta)

                score = max(score, follow)
                alpha = max(alpha, follow)

                if follow > beta:
                    return score

            return score

        def minAgent(gameState, index, depth, alpha, beta):
            if isOver(gameState, depth):
                return self.evaluationFunction(gameState)

            score = math.inf
            for action in gameState.getLegalActions(index):
                gs = gameState.generateSuccessor(index, action)

                follow = 0
                if index + 1 == totalAgents:
                    follow = maxAgent(gs, 0, depth + 1, alpha, beta)
                else:
                    follow = minAgent(gs, index + 1, depth, alpha, beta)

                score = min(score, follow)
                beta = min(beta, follow)

                if follow < alpha:
                    return score

            return score

        bestAction = None

        # NOTE: self.index is always, ALWAYS 0.
        # The following code will reflect this,
        # therefore paxcman (a max agent) will start.

        # (personal note) max sets alpha, prunes on beta
        # (personal note) min sets beta,  prunes on alpha
        score = -math.inf

        # Beta won't be set in the first node
        alpha = -math.inf
        for action in gameState.getLegalActions(self.index):
            # GameState, if this specific action gets taken
            gs = gameState.generateSuccessor(self.index, action)

            follow = minAgent(gameState=gs, index=self.index + 1, depth=0, alpha=alpha, beta=math.inf)
            alpha = max(alpha, follow)

            if follow > score:
                score = follow
                bestAction = action
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):

        def isOver(depth, gameState):
            return depth >= self.depth or gameState.isWin() or gameState.isLose()

        # Helper function to determine if next agent is using max or min
        def agent(gameState, index, depth):
            if index == gameState.getNumAgents():
                # Start with maxagent all over again and increase depth
                return maxAgent(gameState, index=0, depth=depth + 1)
            else:
                return minAgent(gameState, index, depth)

        def minAgent(gameState, index, depth):
            if isOver(depth, gameState):
                return self.evaluationFunction(gameState)

            moves = gameState.getLegalActions()

            totalScore = 0.0
            for nextAction in moves:
                nextState = gameState.generateSuccessor(index, nextAction)

                score = agent(nextState, index+1, depth)

                totalScore += score

            return totalScore / len(moves)

        def maxAgent(gameState, index, depth):
            if isOver(depth, gameState):
                return self.evaluationFunction(gameState)

            totalScore = -math.inf
            for action in gameState.getLegalActions():
                nextState = gameState.generateSuccessor(index, action)
                score = agent(nextState, index+1, depth)
                totalScore = max(totalScore, score)

            return totalScore

        bestAction = None
        bestScore = -math.inf
        for action in gameState.getLegalActions():
            nextState = gameState.generateSuccessor(0, action)
            score = agent(nextState, index=1, depth=0)
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
