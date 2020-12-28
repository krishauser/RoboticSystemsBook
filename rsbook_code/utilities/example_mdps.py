from __future__ import print_function,division

from .mdp import DiscreteMDP
from .example_graphs import grid_graph

class GraphMDP(DiscreteMDP):
    """Converts a graph search problem into a MDP.  The action space at
    a state is the set of neighbors of the state.  With probability pError,
    a random neighbor is chosen, and with probability 1 - pError, the
    desired neighbor is chosen.  (If pError=0, it's a standard graph search
    problem).
    
    Args:
        G (AdjListGraph): the graph to search on.
        reward_dict (dict): a dict mapping graph vertices to rewards
        terminal (list): a list of terminal states
    """
    def __init__(self,G,reward_dict=None,terminal=None,pError=0):
        self.G = G
        self.reward_dict = reward_dict
        if reward_dict is None:
            self.reward_dict = dict()
        if terminal is None:
            self.terminal_set = set()
        else:
            self.terminal_set = set(terminal)
        self.pError = pError
        DiscreteMDP.__init__(self)
    
    def states(self):
        return self.G.nodes()
    
    def actions(self):
        return self.G.nodes()
    
    def applicable_actions(self,s):
        return self.G.neighbors(s)
    
    def successors(self,s,a):
        return self.G.neighbors(s)
    
    def transition_probability(self,s,a,sn):
        Nneighbors = len(self.G.neighbors(s))
        if sn == a:
            return 1.0 - self.pError + self.pError/Nneighbors
        else:
            return self.pError/Nneighbors
    
    def transition_distribution(self,s,a):
        neighbors = self.G.neighbors(s)
        Nneighbors = len(neighbors)
        perror = self.pError/Nneighbors
        res = dict((sn,perror) for sn in neighbors)
        res[a] += 1.0 - self.pError
        return res


    def terminal(self,s):
        return s in self.terminal_set
    
    def num_reward_args(self):
        return 1
    
    def reward(self,s,a=None,sn=None):
        return self.reward_dict.get(s,0)


class GridMDP(DiscreteMDP):
    """Defines an MDP on a grid, with a length-proportional movement cost
    and some directional biases in the transition function.
    
    The reward function is reward_fn(s) evaluated at the arriving state,
    minus |s-s'|*movement_cost.
    
    The transition distribution has a probability of 1-pSelf-pAdjacent-pOther
    of going to the desired neighbor, a probability of pSelf of staying at
    the given node, pAdjacent of arriving at one of the two adjacent
    directions, and pOther of arriving at any other neighbor
    """
    def __init__(self,M,N,diagonals=False,
        movement_cost=0,reward_fn=None,terminal=None,
        pSelf=0,pAdjacent=0,pOther=0):
        self.G = grid_graph(M,N,diagonals)
        self.diagonals = diagonals
        self.movement_cost = movement_cost
        self.reward_fn = reward_fn
        if terminal is None:
            self.terminal_set = set()
        else:
            self.terminal_set = set(terminal)
        self.pSelf = pSelf
        self.pOther = pOther
        self.pAdjacent = pAdjacent
        DiscreteMDP.__init__(self)
    
    def states(self):
        return self.G.nodes()
    
    def actions(self):
        return self.G.nodes()
    
    def applicable_actions(self,s):
        return self.G.neighbors(s)
    
    def successors(self,s,a):
        succ = [a]
        if self.pSelf > 0:
            succ.append(s)
        if self.pOther > 0:
            succ = succ[1:] + list(self.G.neighbors(s))
        if self.pAdjacent > 0:
            for n in self.G.neighbors(s):
                if n == a:
                    continue
                if self.diagonals:
                    if sum(abs(v-w) for (v,w) in zip(n,a)) == 1:
                        succ.append(n)
                else:
                    if sum((1 if v!=w else 0) for (v,w) in zip(n,a)) == 2:
                        succ.append(n)
        return succ
    
    def transition_probability(self,s,a,sn):
        if sn == a:
            return 1.0 - self.pSelf - self.pAdjacent - self.pOther
        if sn == s:
            return self.pSelf
        if self.pAdjacent > 0:
            neighbors = self.G.neighbors(s)
            numOthers = len(neighbors) - 3
            for n in neighbors:
                if self.diagonals:
                    if sum(abs(v-w) for (v,w) in zip(n,a)) == 1:
                        return 0.5*self.pAdjacent
                else:
                    if sum((1 if v!=w else 0) for (v,w) in zip(n,a)) == 2:
                        return 0.5*self.pAdjacent
                if n == sn:
                    return self.pOther/numOthers
        elif self.pOther > 0:
            neighbors = self.G.neighbors(s)
            numOthers = len(neighbors) - 1
            if sn in neighbors:
                return self.pOther/numOthers
        return 0.0
    
    def transition_distribution(self,s,a):
        T = dict()
        T[a] = 1.0 - self.pSelf - self.pAdjacent - self.pOther
        if self.pSelf > 0:
            T[s] = self.pSelf
        if self.pAdjacent > 0:
            neighbors = self.G.neighbors(s)
            numOthers = len(neighbors) - 3
            for n in neighbors:
                if self.diagonals:
                    if sum(abs(v-w) for (v,w) in zip(n,a)) == 1:
                        T[n] = 0.5*self.pAdjacent
                    elif self.pOther > 0:
                        T[n] = self.pOther/numOthers
                else:
                    if sum((1 if v!=w else 0) for (v,w) in zip(n,a)) == 2:
                        T[n] = 0.5*self.pAdjacent
                    elif self.pOther > 0:
                        T[n] = self.pOther/numOthers
        elif self.pOther > 0:
            neighbors = self.G.neighbors(s)
            numOthers = len(neighbors) - 1
            for n in self.G.neighbors(s):
                T[n] = self.pOther/numOthers
        return T


    def terminal(self,s):
        return s in self.terminal_set
    
    def num_reward_args(self):
        if self.movement_cost > 0:
            return 2
        return 1
    
    def reward(self,s,a=None,sn=None):
        import numpy as np
        if a is None or self.movement_cost == 0:
            return self.reward_fn(s)
        return self.reward_fn(s) - self.movement_cost*np.linalg.norm(np.array(s)-np.array(a))
