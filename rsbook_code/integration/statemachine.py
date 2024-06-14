import time
from typing import List,Tuple,Callable,Optional

class StateMachine:
    """A simple state machine class.  Add your states, transitions, and auxiliary data.
    Then, repeatedly call step(). """
    def __init__(self,states : List[str]):
        self.states = states
        self.state_logic = [[] for s in states]
        self.state_transitions = [[] for s in states]
        self.current_state = 0
        self.current_entry_time = None
        self.auxiliary_data = {'start_time':None,'time':None,'state':None,'duration_in_state':None}
    
    def reset(self, init_state : Optional[str] = None):
        """Sets the state machine back to an initial state."""
        if init_state is None:
            self.current_state = 0
        else:
            s = self.states.index(init_state)
            if s < 0:
                raise ValueError("Invalid state")
            self.current_state = s
        self.current_entry_time = None
        self.auxiliary_data['state']=None
        self.auxiliary_data['start_time']=None
        self.auxiliary_data['time']=None
        self.auxiliary_data['duration_in_state']=None

    def add_data(self, name : str, value) -> None:
        """Adds a new item of data to be passed to callbacks."""
        self.auxiliary_data[name] = value

    def add_transition(self,source : str, target : str, test : Callable) -> None:
        """Test is a function f(data) where data is a dict of names
        (added by add_data()) mapped to values.  'time', 'start_time',
        and 'duration_in_state' are also available.

        If trigger is not None, then it is a function f(data) that gets
        called if test(data) returns True.

        Neither should modify data.
        """
        s = self.states.index(source)
        if s < 0:
            raise ValueError("Invalid source state")
        t = self.states.index(target)
        if t < 0:
            raise ValueError("Invalid target state")
        self.state_transitions[s].append((t,test))
    
    def add_logic(self, state : str,
                  enter : Optional[Callable]=None,
                  loop : Optional[Callable]=None,
                  exit : Optional[Callable]=None) -> None:
        """Adds enter, loop, or exit callbacks to a state.  The format of a callback
        f(data) is the same as in add_transition.  However, these function are allowed
        to modify the data dictionary. 
        
        Note: Changes to the defaults 'state', 'time', etc do not actually change
        these quantities in the state machine.

        Note: each callback should not block or take longer than your desired
        time step, so it is not appropriate for long-running tasks.
        """
        s = self.states.index(state)
        if s < 0:
            raise ValueError("Invalid state")
        self.state_logic[s].append((enter,loop,exit))
    
    def all_logic_all(self, enter : Optional[Callable]=None,
                      loop : Optional[Callable]=None,
                      exit : Optional[Callable]=None) -> None:
        """Adds a callback to every state. See all_logic()"""
        for l in self.state_logic:
            l.append((enter,loop,exit))

    def step(self, current_time = None) -> None:
        """Steps forward the state machine logic.  If current_time is given, you
        can control the internal timer.  Otherwise, it uses time.time().
        """
        if current_time is None:
            current_time = time.time()
        if self.current_entry_time is None:
            self.current_entry_time = current_time
            self.auxiliary_data['start_time'] = current_time
        self.auxiliary_data['state'] = self.states[self.current_state]
        self.auxiliary_data['time'] = current_time
        self.auxiliary_data['duration_in_state'] = current_time - self.current_entry_time
        for en,l,ex in self.state_logic[self.current_state]:
            if l is not None:
                l(self.auxiliary_data)
        for (t,test) in self.state_transitions[self.current_state]:
            if test(self.auxiliary_data):
                #transition from s to t
                for en,l,ex in self.state_logic[self.current_state]:
                    if ex is not None:
                        ex(self.auxiliary_data) 
                for en,l,ex in self.state_logic[t]:
                    if en is not None:
                        en(self.auxiliary_data) 
                self.current_entry_time = current_time
                self.current_state = t
                break
    
    def absorbing_states(self) -> List[str]:
        """Returns the list of absorbing (terminal) states."""
        return [self.states[s] for s in range(len(self.states)) if len(self.state_transitions[s])==0]

    def transitions(self) -> List[Tuple[str,str]]:
        """Gathers all transitions in a list of state pairs"""
        res = [set() for s in self.states]
        for s in range(len(self.states)):
            for t,test in self.state_transitions[s]:
                res[s].add(t)
        reslist = []
        for s,ts in enumerate(res):
            for t in ts:
                reslist.append((self.states[s],self.states[t]))
        return reslist

    def duration(self):
        """Returns how long the state machine has been running"""
        if self.auxiliary_data['start_time'] is None:
            return 0
        return self.auxiliary_data['time'] - self.auxiliary_data['start_time']

    def duration_in_state(self):
        """Returns how long the state machine has been in the current state."""
        if self.auxiliary_data['duration_in_state'] is None:
            return 0
        return self.auxiliary_data['duration_in_state']
