# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
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
    #print 'a',legalMoves
    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    #print 'score',scores
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    #print 'c',bestIndices
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
    #print 'a',successorGameState
    newPos = successorGameState.getPacmanPosition()
    #print 'b',newPos
    newFood = successorGameState.getFood()
    #print 'cb',newFood
    newGhostStates = successorGameState.getGhostStates()
    #print 'c',newGhostStates
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    #print 'd',newScaredTimes
    "*** YOUR CODE HERE ***"
    newFoodList = newFood.asList()
    #print 'a',newFoodList
    #newGhostPos = newGhostStates.getPosition()
    #closestGhostDist = min([util.manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates])
    dist = min([util.manhattanDistance(newPos,a.getPosition()) for a in newGhostStates])
    if dist>3:
      #dist = 4
      x=1
      y=1
      dist = 3
      dist2 = ([util.manhattanDistance(newPos,a )for a in newFoodList])
      if len(dist2) > 0:
        
        dist1 = 10-min(dist2)
        #if dist1 >= dist:
         # dist1 = dist1 -2;
      else:
        dist1 = 10
      
    else :
      dist1 = 0
      x=1
      y=0
     
    #for a in newGhostStates :
     # print 'aaa',a.getPosition()
    #print 'd',newFoodList
    #print 'a',dist
    #print 'b',  dist1
    return successorGameState.getScore()+dist*x+dist1*y 

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

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
  
  def TerminalTest(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or self.depth == depth:
          #print 'depth',depth
          return True
        return False
  def MinVal(self,gameState,depth,ghosts):
    
    if self.TerminalTest(gameState, depth):
      #print 'd'
      return self.evaluationFunction(gameState)
    
    val = float("inf")
    #print '1234',ghosts
    for i in range(1, ghosts):
     # print '123',i
      actions = gameState.getLegalActions(i)
      for action in actions:
        successor = gameState.generateSuccessor(i, action)
        scores = self.MaxVal(successor, depth+1)
        val = min(val,scores)
        #print '1234',val
    return val
  def MaxVal(self, gameState,depth):
    d = depth
    #print 'sasad',d
    if self.TerminalTest(gameState, d):
      #print 'depth',d
      return self.evaluationFunction(gameState)
    val = float("-inf")
    actions = gameState.getLegalActions(0)
    #print actions
    for action in actions:
      successor = gameState.generateSuccessor(0, action)
      scores = self.MinVal(successor, depth+1, gameState.getNumAgents())
      if scores > val:
        maxAction=action
        #print 'aS',maxAction
        val = scores
    if depth == 0:
      #print maxAction
      return maxAction  
      
    #print 'val',val
    
    return val
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    
    
    #print 'aasas',gameState
    #print 'asd',gameState.generateSuccessor(0,'West')
    #print 'a',gameState.getNumAgents()
      #print 'b',self.MaxVal(gameState,1)
    return self.MaxVal(gameState, 0)
    #return 0
    #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  def TerminalTest(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or self.depth == depth:
          #print 'depth',depth
          return True
        return False
  def alphamax(self,gameState,depth,alpha,beta):
        if self.TerminalTest(gameState, depth):
          return self.evaluationFunction(gameState)
        val = float("-inf")
        actions = gameState.getLegalActions(0)
        #if Directions.STOP in actions:
          #actions.remove(Directions.STOP)
        #print actions
        for action in actions:
          successor = gameState.generateSuccessor(0, action)
          #print successor
          scores = self.betamin(successor, depth+1, gameState.getNumAgents(),alpha,beta)
          if scores > val:
            maxAction=action
            val = scores
          if val > beta :
            return val
          alpha = max(alpha, val)
        if depth == 0:
            return maxAction  
        return val
  def betamin(self,gameState,depth,ghosts,alpha,beta):
        if self.TerminalTest(gameState, depth):
          #print 'd'
          return self.evaluationFunction(gameState)
    
        val = float("inf")
        #print '1234',ghosts
        for i in range(1, ghosts):
          # print '123',i
          actions = gameState.getLegalActions(i)
          for action in actions:
            successor = gameState.generateSuccessor(i, action)
            scores = self.alphamax(successor, depth+1,alpha,beta)
            val = min(val,scores)
            if val < alpha:
              return val
            beta = min (beta,val)
           
        #print '1234',val
        return val
  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    return self.alphamax(gameState, 0, float("-inf"), float("inf"))

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """
  def TerminalTest(self, gameState, depth):
        if gameState.isWin() or gameState.isLose() or self.depth == depth:
          #print 'depth',depth
          return True
        return False
  def MaxVal(self, gameState,depth):
        d = depth
        #print 'sasad',d
        if self.TerminalTest(gameState, d):
        #print 'depth',d
          return self.evaluationFunction(gameState)
        val = float("-inf")
        actions = gameState.getLegalActions(0)
    #print actions
        for action in actions:
          successor = gameState.generateSuccessor(0, action)
          scores = self.EVal(successor, depth+1, gameState.getNumAgents())
          if scores > val:
            maxAction=action
            #print 'aS',maxAction
            val = scores
        if depth == 0:
          #print maxAction
          return maxAction  
      
        #print 'val',val
    
        return val
  def EVal(self,gameState,depth,agentnum):
      if self.TerminalTest(gameState,depth):
        return self.evaluationFunction(gameState);
      val = 0
      n=0
    #print '1234',ghosts
      for i in range(1, agentnum):
     # print '123',i
        actions = gameState.getLegalActions(i)
        for action in actions:
          successor = gameState.generateSuccessor(i, action)
          scores = self.MaxVal(successor, depth+1)
          val = val+scores
          n=n+1
        #print '1234',val
      return val/n
  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    return self.MaxVal(gameState, 0)

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    When pacman is near the ghost,the only thing we care is which action can lead pacman to get away from ghost.
    When pacman is far from the ghost,the most two important thing is to eat the nearest food and capsule.Also we should take the number of capsule left into account
    to make pacman move quickly,the less number of capsule left,the higher vaule will this point be.
    
  """
  "*** YOUR CODE HERE ***"
  newPos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  #print newFood.height
  newGhostStates = currentGameState.getGhostStates()
  #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
  #print newScaredTimes
  newFoodList = newFood.asList()
  #print newFoodList
  ghostdist = 99
  threshhold = newFood.height*newFood.width
  powerdist = threshhold
  fooddist = threshhold
  newCapsules = currentGameState.getCapsules()
  #print newCapsules,'asd'
  for food in newFoodList :
    fooddist = min(fooddist,(util.manhattanDistance(newPos,food)))
  for capsule in newCapsules:
    powerdist = min (powerdist,manhattanDistance(newPos, capsule))
  #print newCapsules
  for ghost in newGhostStates:
        if ghost.scaredTimer <= 0:
          #print ghost
          ghostdist = min(util.manhattanDistance(newPos,ghost.getPosition()),ghostdist)
  #fooddists = ([util.manhattanDistance(newPos,a )for a in newFoodList])
  
  if ghostdist>1.5:
    #dist = 4
    x=0
    y=1
    z=1.5
    m=1000
    #print ghostdist
    #ghostdist = 3
    
      
  else :
    #fooddist = 0
    x=1
    y=0.5
    z=0.5
    m=0
  ghostval = x * ghostdist
  foodval = y * fooddist
  #print fooddist
  fooddist = threshhold -fooddist
  powerdist = threshhold - powerdist
  #print powerdist
    #for a in newGhostStates :
     # print 'aaa',a.getPosition()
    #print 'd',newFoodList
    #print 'a',dist
    #print 'b',  dist1
  #print ghostdist," ",fooddist
  val = ghostdist*x+fooddist*y+z*powerdist
  #print val,x,ghostdist,fooddist,y
  return currentGameState.getScore()+ghostdist*x+fooddist*y+z*powerdist+m*(20-len(newCapsules))
# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

