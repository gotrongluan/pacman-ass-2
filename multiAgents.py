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


from util import manhattanDistance
from game import Directions
from game import Actions
import random, util

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPosition = successorGameState.getPacmanPosition()
        newGhostPositions = successorGameState.getGhostPositions()
        if successorGameState.isWin():
          return 1000000
        if successorGameState.isLose():
          return -1000000
        score = 0

        #common
        newFoods = successorGameState.getFood().asList()
        manhatToFoods = [manhattanDistance(newPosition, foodPosition) for foodPosition in newFoods]
        minManhatToFood = min(manhatToFoods) if len(manhatToFoods) > 0 else 0
        score += 10 if currentGameState.getNumFood() > successorGameState.getNumFood() else -minManhatToFood
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimers = [ghostState.scaredTimer for ghostState in newGhostStates]
        newManhatToGhosts = [manhattanDistance(newPosition, newGhostPositions[i]) for i in range(len(newGhostPositions)) if newScaredTimers[i] == 0]
        newMinManhatToGhost = min(newManhatToGhosts) if len(newManhatToGhosts) > 0 else -1
        if 2 < newMinManhatToGhost <= 6:
          score -= newMinManhatToGhost
        elif newMinManhatToGhost == 2:
          score -= 500
        elif newMinManhatToGhost == 1:
          return -10000
        elif newMinManhatToGhost == 0:
          return -1000000

        #branch
        newCapsules, curCapsules = successorGameState.getCapsules(), currentGameState.getCapsules()

        if len(newCapsules) == 0:
          #Had Eaten
          if len(curCapsules) > 0:
            return 500
          else:
            newManhatToScaredGhosts = [manhattanDistance(newPosition, newGhostPositions[i]) for i in range(len(newGhostPositions)) if newScaredTimers[i] > 0]
            curGhostStates = currentGameState.getGhostStates()
            curScaredTimers = [ghostState.scaredTimer for ghostState in curGhostStates]
            curPositiveScaredTimers = [t for t in curScaredTimers if t > 0]
            if newManhatToScaredGhosts != []:
              score -= 10 * newManhatToScaredGhosts[0]
            else:
              if curPositiveScaredTimers != []:
                score += 200
        else:
          capsule = newCapsules[0]
          score -= 4.5 * manhattanDistance(newPosition, capsule)
        return score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
    
    def getListActionsSets(self, listActions, turn = 0):
        return [[]] if turn == len(listActions) else [[x] + y for x in listActions[turn] for y in self.getListActionsSets(listActions, turn + 1)]

    def generateMaxState(self, minState, actionSet):
        curState = minState
        for i in range(len(actionSet)):
          if curState.isWin() or curState.isLose():
            return curState
          curState = curState.generateSuccessor(i + 1, actionSet[i])
        return curState

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def getMinValue(self, minState, depth):
        if minState.isWin() or minState.isLose():
          return self.evaluationFunction(minState)
        legalGhostsActions = [minState.getLegalActions(i) for i in range(1, minState.getNumAgents())]
        actionSets = self.getListActionsSets(legalGhostsActions, 0)
        minValue = float("inf")
        for actionSet in actionSets:
          maxState = self.generateMaxState(minState, actionSet)
          minValue = min(minValue, self.getMaxValue(maxState, depth - 1))
        return minValue
    
    def getMaxValue(self, maxState, depth):
        if depth == 0 or maxState.isWin() or maxState.isLose():
          return self.evaluationFunction(maxState)
        else:
          legalPacmanActions = maxState.getLegalActions(0)
          maxValue = -float("inf")
          for action in legalPacmanActions:
            minState = maxState.generateSuccessor(0, action)
            maxValue = max(maxValue, self.getMinValue(minState, depth))
          return maxValue

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
        legalActions = gameState.getLegalActions(0)
        minStates = [gameState.generateSuccessor(0, action) for action in legalActions]
        minValues = [self.getMinValue(minState, self.depth) for minState in minStates]
        bestMinValue = max(minValues)
        bestIndices = [index for index in range(len(minValues)) if minValues[index] == bestMinValue]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalActions[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getMinValue(self, minState, alpha, beta, depth):
        if minState.isWin() or minState.isLose():
          return self.evaluationFunction(minState)
        legalGhostsActions = [minState.getLegalActions(i) for i in range(1, minState.getNumAgents())]
        actionSets = self.getListActionsSets(legalGhostsActions, 0)
        minValue = float("inf")
        for actionSet in actionSets:
          maxState = self.generateMaxState(minState, actionSet)
          minValue = min(minValue, self.getMaxValue(maxState, alpha, beta, depth - 1))
          if minValue < alpha:
            return minValue
          beta = min(beta, minValue)
        return minValue
    
    def getMaxValue(self, maxState, alpha, beta, depth):
        if depth == 0 or maxState.isWin() or maxState.isLose():
          return self.evaluationFunction(maxState)
        else:
          legalPacmanActions = maxState.getLegalActions(0)
          maxValue = -float("inf")
          for action in legalPacmanActions:
            minState = maxState.generateSuccessor(0, action)
            maxValue = max(maxValue, self.getMinValue(minState, alpha, beta, depth))
            if maxValue > beta:
              return maxValue
            alpha = max(alpha, maxValue)
          return maxValue

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions(0)
        alpha, beta = -float("inf"), float("inf")
        maxValue = -float("inf")
        chosenAction = None
        for action in legalActions:
          minState = gameState.generateSuccessor(0, action)
          minValue = self.getMinValue(minState, alpha, beta, self.depth)
          if minValue > maxValue:
            maxValue = minValue
            chosenAction = action
          alpha = max(alpha, minValue)
        return chosenAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getChanceValue(self, chanceState, depth):
        if chanceState.isWin() or chanceState.isLose():
          return self.evaluationFunction(chanceState)
        legalGhostsActions = [chanceState.getLegalActions(i) for i in range(1, chanceState.getNumAgents())]
        actionSets = self.getListActionsSets(legalGhostsActions, 0)
        chanceValue = 0
        p = float(1) / len(actionSets)
        for actionSet in actionSets:
          maxState = self.generateMaxState(chanceState, actionSet)
          chanceValue += p * self.getMaxValue(maxState, depth - 1)
        return chanceValue
    
    def getMaxValue(self, maxState, depth):
        if depth == 0 or maxState.isWin() or maxState.isLose():
          return self.evaluationFunction(maxState)
        else:
          legalPacmanActions = maxState.getLegalActions(0)
          maxValue = -float("inf")
          for action in legalPacmanActions:
            chanceState = maxState.generateSuccessor(0, action)
            maxValue = max(maxValue, self.getChanceValue(chanceState, depth))
          return maxValue

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        legalActions = gameState.getLegalActions(0)
        chanceStates = [gameState.generateSuccessor(0, action) for action in legalActions]
        chanceValues = [self.getChanceValue(chanceState, self.depth) for chanceState in chanceStates]
        bestchanceValue = max(chanceValues)
        bestIndices = [index for index in range(len(chanceValues)) if chanceValues[index] == bestchanceValue]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return legalActions[chosenIndex]

mazeTable = dict()

def mazeToSet(p, s, gameState):
    global mazeTable
    mazes = list()
    p = (int(p[0]), int(p[1]))
    for e in s:
      e = (int(e[0]), int(e[1]))
      if (p, e) not in mazeTable.keys():
        maze = mazeDistance(p, e, gameState)
        mazeTable[(p, e)] = maze
        mazes.append(maze)
      else:
        mazes.append(mazeTable[(p, e)])
    return mazes

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    global mazeTable
    if currentGameState.isLose():
      return -float("inf")
    score = 0
    if currentGameState.isWin():
      score += 1000000
    foods = currentGameState.getFood().asList()
    pacmanPosition = currentGameState.getPacmanPosition()
    
    #MinDistance to Foods
    if len(foods) > 0:
      manhatToFoods = [manhattanDistance(pacmanPosition, food) for food in foods]
      minManhatToFood = min(manhatToFoods)
      index = manhatToFoods.index(minManhatToFood)
      score -= mazeDistance(pacmanPosition, foods[index], currentGameState)

    #Num of food
    numFood = currentGameState.getNumFood()
    score -= 20 * numFood

    #Live Ghost
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()

    liveGhostPositions = [ghostPositions[g] for g in range(len(ghostPositions)) if scaredTimers[g] == 0]
    mazeToLiveGhosts = mazeToSet(pacmanPosition, liveGhostPositions, currentGameState)
    minMazeToLiveGhost = min(mazeToLiveGhosts) if mazeToLiveGhosts != list() else -1
    if 2 < minMazeToLiveGhost <= 6:
      score -= minMazeToLiveGhost
    elif minMazeToLiveGhost == 2:
      score -= 700
    elif minMazeToLiveGhost == 1:
      return -10000
    elif minMazeToLiveGhost == 0:
      return -1000000
    
    #Capsules
    capsules = currentGameState.getCapsules()
    numCapsule = len(capsules)
    score -= numCapsule * 250
    if numCapsule > 0:
      mazeToCapsules = mazeToSet(pacmanPosition, capsules, currentGameState)
      minMazeToCapsule = min(mazeToCapsules)
      score -= 5 * minMazeToCapsule
    
    #Scared Ghosts
    scaredGhostPositions = [ghostPositions[g] for g in range(len(ghostPositions)) if scaredTimers[g] > 0]
    
    numScaredGhost = len(scaredGhostPositions)
    score -= numScaredGhost * 90
    if numScaredGhost > 0:
      mazeToScaredGhosts = mazeToSet(pacmanPosition, scaredGhostPositions, currentGameState)
      minMazeToScaredGhost = min(mazeToScaredGhosts)
      score -= 6 * minMazeToScaredGhost
    return score

def breadthFirstSearch(problem):
    fringe = util.Queue()
    start = problem.getStartState()
    visited = set()
    fringe.push((start, []))
    while not fringe.isEmpty():
      state, path = fringe.pop()
      if problem.isGoalState(state):
        return path
      if state not in visited:
        visited.add(state)
        successors = problem.getSuccessors(state)
        for successor, action, stepCost in successors:
          fringe.push((successor, path + [action]))
    return None

class PositionSearchProblem:
    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

def mazeDistance(point1, point2, gameState):
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(breadthFirstSearch(prob))

# Abbreviation
better = betterEvaluationFunction
