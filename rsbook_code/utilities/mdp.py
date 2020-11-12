from __future__ import print_function,division
import numpy as np
import random
from scipy import sparse
import scipy.sparse.linalg
from builtins import range

class DiscreteMDP:
    """Base class for a Markov Decision Problem.  Subclasses should 
    overload, at a minimum,
    - states()
    - actions()
    - transition_probability(s,a,sn)
    - num_reward_args()
    - reward(s,a,sn)
    
    The reward function can be in the form R(s), R(s,a), or R(s,a,snext). 
    Which form is determined by num_reward_args().
    """
    def __init__(self,discount=1.0):
        self.discount = discount

    def states(self):
        raise NotImplementedError()
    
    def actions(self):
        raise NotImplementedError()
    
    def applicable_actions(self,s):
        """Returns the list of possible actions taken in s.  By default this
        returns the set of all actions."""
        return self.actions()
    
    def successors(self,s,a):
        """Returns the list of possible successors after applying action a in
        state s.  By default this returns list of all states, but for better
        speed this should return a smaller subset of states."""
        return self.states()
    
    def transition_probability(self,s,a,sn):
        raise NotImplementedError()
    
    def transition_distribution(self,s,a):
        return dict((sn,self.transition_probability(s,a,sn)) for sn in self.successors(s,a))
    
    def terminal(self,s):
        """Returns True if s is a terminal state"""
        return False
    
    def num_reward_args(self):
        """Must return 1, 2, or 3 depending on whether the reward function
        accepts s only (1), s and a (2), or s, a, and sn (3)"""
        raise NotImplementedError()
    
    def reward(self,s,a=None,sn=None):
        """Returns the reward function value R(s), R(s,a), or R(s,a,sn).
        Which option to use is defined by num_reward_args.
        
        Even if num_reward_args() > 1, this may still be called in the
        form R(s) if s is a terminal state.
        """
        raise NotImplementedError()

    
class DiscreteMDPSolver:
    """A solver for DiscreteMDPs.  Provides Value Iteration and Policy Iteration,
    as well as some helper functions e.g., for simulating rollouts.
    
    Attributes:
        mdp: the DiscreteMDP
        states (list): list of states
        state_index (dict): maps states to indices
        actions (list): list of actions
        action_index (dict): maps actions to indices
        transition_array (list of dict of (array,array) pairs): indexes the transition
            probability distributions by [state_index][action_index]. The value is a pair of
            np.ndarrays containing the next_state_index and the probability of transition. 
            Only entries in the dict are the applicable actions.
        transition_sparse_array (list of sparse arrays): indexes the transition probability,
            one per state. The matrix for state s has size applicable_actions(s) x |S|.
    """
    def __init__(self,mdp):
        """Initializes the solver with an MDP.
        
        Args:
            mdp (DiscreteMDP):
        """
        self.mdp = mdp
        self.states = mdp.states()
        self.state_index = dict((s,i) for (i,s) in enumerate(self.states))
        self.actions = mdp.actions()
        self.action_index = dict((s,i) for (i,s) in enumerate(self.actions))
        
        self.terminal_array = np.array([mdp.terminal(s) for s in self.states],dtype=bool)
        self.applicable_array = []
        self.transition_array = []
        self.transition_sparse_array = []
        for i,s in enumerate(self.states):
            self.transition_array.append(dict())
            if self.terminal_array[i]:
                self.applicable_array.append([0])
                self.transition_sparse_array.append(sparse.dok_matrix((1,len(self.states))))
            else:
                actions = mdp.applicable_actions(s)
                self.transition_sparse_array.append(sparse.dok_matrix((len(actions),len(self.states))))
                appl = [self.action_index[a] for a in actions]
                self.applicable_array.append(appl)
                for ia,a in enumerate(actions):
                    aindex = appl[ia]
                    T = mdp.transition_distribution(s,a)
                    sn_indices = []
                    sn_probabilities = []
                    for (sn,pr) in T.items():
                        assert (pr >= 0 and pr <= 1),"Invalid transition probability, must be in range [0,1]"
                        isn = self.state_index[sn]
                        sn_indices.append(isn)
                        sn_probabilities.append(pr)
                        self.transition_sparse_array[-1][ia,isn] = pr
                    self.transition_array[-1][aindex] = (np.array(sn_indices,dtype=int),np.array(sn_probabilities))
            self.transition_sparse_array[-1] = self.transition_sparse_array[-1].tocsr()
        if mdp.num_reward_args() == 1:
            self.reward_array = np.array([mdp.reward(s) for s in self.states])
        elif mdp.num_reward_args() == 2:
            self.reward_array = sparse.dok_matrix((len(self.states),len(self.actions)))
            self.reward_compressed_array = []
            for (i,s) in enumerate(self.states):
                if self.terminal_array[i]:
                    try:
                        r = self.mdp.reward(s)
                    except TypeError:
                        r = 0
                    self.reward_compressed_array.append(np.array([r]))
                    self.reward_array[i,0] = r
                else:
                    action_indices = self.applicable_array[i]
                    rewards = []
                    for j in action_indices:
                        r = mdp.reward(s,self.actions[j])
                        rewards.append(r)
                        self.reward_array[i,j] = r
                    self.reward_compressed_array.append(np.array(rewards))
            self.reward_array = self.reward_array.tocsc()
        elif mdp.num_reward_args() == 3:
            self.reward_compressed_array = []
            self.reward_array = sparse.dok_matrix((len(self.states),len(self.actions)))
            for i,s in enumerate(self.states):
                if self.terminal_array[i]:
                    try:
                        r = self.mdp.reward(s)
                    except TypeError:
                        r = 0
                    self.reward_compressed_array.append(np.array([r]))
                    self.reward_array[i,0] = r
                else:
                    action_indices = self.applicable_array[i]
                    rewards = []
                    for j in action_indices:
                        successor_indices,successor_values = self.transition_array[i][j]
                        ravg = 0.0
                        for (isn,pr) in zip(successor_indices,successor_values):
                            r = mdp.reward(s,a,self.states[isn])
                            ravg += r*pr
                        self.reward_array[i,j] = ravg
                        rewards.append(ravg)
                    self.reward_compressed_array.append(np.array(rewards))
            self.reward_array = self.reward_array.tocsc()
    
    def sample_rollout(self,s0,policy=None,action_sequence=None,max_steps=1000):
        """Samples a rollout of a policy or sequence of actions.
        
        Args:
            s0 (state): the initial state
            policy (callable, optional): if given, the policy to follow.
            action_sequence (list of actions): if given, the sequence of actions
                to follow.
            max_steps (int, optional): limits the maximum steps rolled out.
        
        Returns:
            list: a sequence (s0,a0,r0,s1,a1,r1,...,sT) where sT is a terminal
            state or T=max_steps.
        """
        if policy is None and action_sequence is None:
            raise ValueError("Need to provide at least one of policy or action_sequence")
        if action_sequence is not None:
            max_steps = min(len(action_sequence),max_steps)
        res = [s0]
        s = s0
        sindex = self.state_index[s0]
        for t in range(max_steps):
            if self.terminal_array[sindex]:
                return res
            if policy is None:
                a = action_sequence[t]
            else:
                a = policy(s)
            aindex = self.action_index[a]
            res.append(a)
            Ts = self.transition_array[sindex][aindex]
            snindex = np.random.choice(Ts[0],p=Ts[1])
            sn = self.states[snindex]
            r = self.mdp.reward(s,a,sn)
            res.append(r)
            res.append(sn)
            s = sn
            sindex = snindex
        return res
    
    def rollout_states(self,rollout):
        return rollout[0::3]
    
    def rollout_actions(self,rollout):
        return rollout[1::3]
    
    def rollout_rewards(self,rollout):
        return rollout[2::3]
    
    def rollout_return(self,rollout,discount=True):
        rterm = 0
        if len(self.reward_array.shape) == 1:
            slast = self.state_index[rollout[-1]]
            if self.terminal_array[slast]:
                rterm = self.reward_array[slast]
        if not discount or self.mdp.discount == 1:
            return rterm + sum(self.rollout_rewards(rollout))
        else:
            ret = 0.0
            rew = self.rollout_rewards(rollout)
            scale = 1.0
            for r in rew:
                ret += scale*r
                scale *= self.mdp.discount
            return ret + scale*rterm
            
    def value(self,policy,raw=False):
        """Solves for the value of a policy using inversion. 
        
        Args:
            policy (callable or array):  The function pi(s) -> a, either a
                function or an array.
            raw (bool, optional): if True, the policy array is assumed to be
                 an array of action indices.
        
        Returns:
            np.ndarray: the value function vector, whose entries are matched 
            to the self.states array.
        """
        if callable(policy):
            policy_array = np.array([(self.action_index[policy(s)] if not g else 0) for s,g in zip(self.states,self.terminal_array)])
        else:
            if not hasattr(policy,'__iter__'):
                raise TypeError("Invalid policy value, must be a function or an array")
            if len(policy) != len(self.states):
                raise ValueError("Invalid policy value, must be an array of length |S|")
            if raw:
                policy_array = policy
            else:
                policy_array = np.array([self.action_index[p] for p in policy])
        assert np.all(policy_array >= 0) and np.all(policy_array < len(self.actions)),"Invalid policy value"
        T = sparse.dok_matrix((len(self.states),len(self.states)))
        for i,Ts in enumerate(self.transition_array):
            if self.terminal_array[i]:
                pass
            else:
                Tsa = Ts[policy_array[i]]
                for j,p in zip(Tsa[0],Tsa[1]):
                    T[i,j] = p
        T = T.tocsc()
        if len(self.reward_array.shape) == 1:
            r = self.reward_array
        else:
            #reward is action dependent, form vector r by selecting the policy's value
            #TODO: is there a nice way to do this quickly with scipy sparse matrix?
            r = np.array([self.reward_array[i,j] for (i,j) in enumerate(policy_array)])
        assert r.shape == (len(self.states),)
        
        #solve for (I-gamma T) v = r
        A = sparse.eye(len(self.states)) - self.mdp.discount*T
        v = scipy.sparse.linalg.spsolve(A,r)
        return v
    
    def greedy_policy(self,values,raw=False):
        """Given a vector of values, returns the greedy policy.  Runs in
        O(|S||A|k) time, where |A| is the max # of actions per state, and
        k is the # of successors in a transition."""
        p = []
        for i,s in enumerate(self.states):
            if self.terminal_array[i]:
                p.append(None)
                continue
            #calculate optimal action for state s
            if len(self.reward_array.shape) == 1:
                avals = self.transition_sparse_array[i].dot(values)
            else:
                avals = self.reward_compressed_array[i] + self.transition_sparse_array[i].dot(values)
            abest = np.argmax(avals)
            p.append(abest)
        if raw:
            return p
        else:
            return [None if a is None else self.mdp.applicable_actions(s)[a] for (s,a) in zip(self.states,p)]
    
    def bellman_backup(self,values):
        """Given a value function array, performs one step of a bellman backup."""
        values = np.asarray(values)
        if len(self.reward_array.shape) == 1:
            temp = self.reward_array.astype(float)
            for i,s in enumerate(self.states):
                if self.terminal_array[i]: continue
                temp[i] += np.max(self.transition_sparse_array[i].dot(values))
        else:
            temp = np.zeros(len(self.states))
            for i,s in enumerate(self.states):
                if self.terminal_array[i]: 
                    temp[i] = values[i]
                else:
                    temp[i] = np.max(self.reward_compressed_array[i] + self.transition_sparse_array[i].dot(values))
        return temp
    
    def value_iteration(self,initial_values=None,N=100,epsilon=1e-3,verbose=0):
        """Performs the value iteration algorithm
        
        Args:
            initial_values (array, optional): a guess for the initial values
            N (int, optional): the number of iterations to run
            epsilon (float, optional): convergence tolerance
        """
        if initial_values is None:
            if len(self.reward_array.shape) == 1:
                initial_values = self.reward_array
            else:
                avg_reward = []
                for i in range(len(self.states)):
                    ri = self.reward_array[i,self.applicable_array[i]]
                    avg_reward.append(np.average(ri.todense()))
                initial_values = np.array(avg_reward)
        else:
            initial_values = np.asarray(initial_values)
        if len(initial_values) != len(self.states):
            raise ValueError("Invalid length of initial values, must have length |S|")
        values = initial_values.astype(float)
        for i in range(N):
            if verbose >= 2:
                print("Value iteration",i,", value function average",np.average(values))
                if verbose >= 3:
                    print("  Values",values)
            new_values = self.bellman_backup(values)
            if epsilon > 0 and np.allclose(values,new_values,rtol=0,atol=epsilon):
                if verbose:
                    print("Value iteration converged after",i,"iterations")
                return new_values
            if verbose >= 2:
                print("Value function difference",np.max(np.abs(values - new_values)))
            values = new_values
        if verbose:
            print("Value iteration did not converge after",N,"iterations")
        return values

    def policy_iteration(self,initial_policy=None,N=100,epsilon=1e-5,verbose=0):
        """Performs the policy iteration algorithm
        
        Args:
            initial_policy (list, optional): a guess for the initial policy
            N (int, optional): the number of iterations to run
            epsilon (float, optional): convergence tolerance
        
        Returns:
            list: a list of actions, aligned with the entries of self.states.
        """
        if initial_policy is None:
            initial_policy = [random.choice(appl) for appl in self.applicable_array]
        elif callable(initial_policy):
            initial_policy = [(self.action_index[initial_policy(s)] if not g else 0) for s,g in zip(self.states,self.terminal_array)]
        else:
            if len(initial_policy) != len(self.states):
                raise ValueError("Invalid length of initial_policy, must have length |S|")
            initial_policy = [self.action_index[p] for p in initial_policy]
        policy = np.array(initial_policy,dtype=int)
        values = self.value(policy,raw=True)
        for iters in range(N):
            if verbose:
                print("Policy iteration",iters,", value function average",np.average(values))
                if verbose >= 2:
                    print("  Values",values)
            for i,(s,p) in enumerate(zip(self.states,policy)):
                if self.terminal_array[i]:
                    continue
                #calculate optimal action for state s
                if len(self.reward_array.shape) == 1:
                    avals = self.transition_sparse_array[i].dot(values)
                else:
                    avals = self.reward_compressed_array[i] + self.transition_sparse_array[i].dot(values)
                abest = np.argmax(avals)
                abest = self.applicable_array[i][abest]
                if abest != p:
                    if verbose >= 2:
                        print("State",s,"action changed from",self.actions[p],"to",self.actions[abest])
                    policy[i] = abest
                else:
                    if verbose >= 2:
                        print("State",s,"action stayed at",self.actions[p])
            if iters + 1 == N:
                continue
            newvalues = self.value(policy,raw=True)
            if np.allclose(values,newvalues,rtol=0,atol=epsilon):
                if verbose >= 1:
                    print("Policy iteration converged in",iters+1,"iterations")
                return [self.actions[a] for a in policy]
            values = newvalues
        if verbose >= 1:
            print("Policy iteration did not converge in",N,"iterations")
        return [self.actions[a] for a in policy]

