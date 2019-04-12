from copy import deepcopy, copy
import pygraphviz as pgv

class States:

    @staticmethod
    def recursive(problem, state, depth, visited_states):
        results = problem.succ(state)
        for result in results:
            print(depth * '\t' + str(result))
            States.recursive(problem, result, depth+1, visited_states)

    @staticmethod
    def iterative(problem, state):
        result = {problem.hash_state(state): {'state': state, 'children': set([])}}
        if problem.logfile!= None:
            print("Initial state "+problem.state2printstring(state))

        processing_list = [state]
        while processing_list:
            processing_state = processing_list.pop(0)
            successors = problem.succ(processing_state)
            for succ in successors:
                if problem.hash_state(succ) != problem.hash_state(processing_state):
                    hashed_state = problem.hash_state(succ)
                    result[problem.hash_state(processing_state)]['children'].add(hashed_state)
                    if hashed_state not in result:
                        if problem.logfile!= None:
                            print("Adding state "+problem.state2printstring(state))
                        result[hashed_state] = {'state': succ, 'children': set([])}
                        processing_list.append(succ)
                    elif problem.logfile!= None:
                        print("State "+problem.state2printstring(state)+" already explored")

        if problem.logfile!= None:
                        print("All possible states explored")
        return result

def base_prob():
    inflow = Quantity("I", [("0", False, 0), ("+", True, 1)])
    volume = Quantity("V", [("0", False, 0), ("+", True, 1), ("Max", False, 1)])
    outflow = Quantity("O", [("0", False, 0), ("+", True, 1), ("Max", False, 1)])

    volume.influence(inflow, positive=True)
    volume.influence(outflow, positive=False)
    outflow.proportional(volume, positive=True)

    volume.value_constraint(outflow, "Max", "Max")
    outflow.value_constraint(volume, "Max", "Max")
    volume.value_constraint(outflow, "0", "0")
    outflow.value_constraint(volume, "0", "0")

    return inflow, volume, outflow

def frog_prob():
    population = Quantity("Population", [("0", False, 0), ("Small", True, 1), ("Medium", False, 1), ("Large", True, 1)])
    birth = Quantity("Birth", [("0", False, 0), ("+", True, 1)])
    death = Quantity("Death", [("0", False, 0), ("+", True, 1)])

    population.influence(birth, positive=True)
    population.influence(death, positive=False)
    birth.proportional(population, positive=True)
    death.proportional(population, positive=True)

    return population, birth, death

def extra_prob():
    inflow = Quantity("I", [("0", False, 0), ("+", True, 1)])
    volume = Quantity("V", [("0", False, 0), ("+", True, 1), ("Max", False, 1)])
    height = Quantity("H", [("0", False, 0), ("+", True, 1), ("Max", False, 1)])
    pressure = Quantity("P", [("0", False, 0), ("+", True, 1), ("Max", False, 1)])
    outflow = Quantity("O", [("0", False, 0), ("+", True, 1), ("Max", False, 1)])

    volume.influence(inflow, positive=True)
    volume.influence(outflow, positive=False)
    outflow.proportional(pressure, positive=True)
    pressure.proportional(height, positive=True)
    height.proportional(volume, positive=True)

    pressure.value_constraint(outflow, "Max", "Max")
    outflow.value_constraint(pressure, "Max", "Max")
    pressure.value_constraint(outflow, "0", "0")
    outflow.value_constraint(pressure, "0", "0")

    pressure.value_constraint(height, "Max", "Max")
    height.value_constraint(pressure, "Max", "Max")
    pressure.value_constraint(height, "0", "0")
    height.value_constraint(pressure, "0", "0")

    volume.value_constraint(height, "Max", "Max")
    height.value_constraint(volume, "Max", "Max")
    volume.value_constraint(height, "0", "0")
    height.value_constraint(volume, "0", "0")

    return inflow, volume, height, pressure, outflow


class Quantity:

    def __init__(self, name, values):
        self.name = name
        self.values, self.intervals, self.signs = zip(*values)
        self.derivatives = [-1, 0, 1]
        self.influences = []
        self.proportionals = []
        self.value_constraints = []
        self.child = False

    def influence(self, quantity, positive=True):
        quantity.child = True
        self.influences.append((quantity, positive))

    def proportional(self, quantity, positive=True):
        quantity.child = True
        self.proportionals.append((quantity, positive))

    def value_constraint(self, quantity, own_val, other_val):
        quantity.child = True
        if own_val in self.values and other_val in quantity.values:
            self.value_constraints.append((quantity, own_val, other_val))
        else:
            raise ValueError('Incorrect value specified')

    def hasProportionals(self):
        return len(self.proportionals) > 0

    def hasInfluences(self):
        return len(self.influences) > 0

    def value2index(self, value):
        return self.values.index(value)

    def isMax(self, value):
        value_idx = self.value2index(value)
        return (value_idx == len(self.values)-1 and not self.intervals[value_idx])

    def isMin(self, value):
        value_idx = self.value2index(value)
        return (value_idx == 0 and not self.intervals[value_idx])

    def next_value(self, value, derivative):
        result = []
        index = self.value2index(value)
        if derivative == 0 or self.intervals[index]:
            result.append(value)

        # Check if there is a next value
        if derivative > 0 and index+1 < len(self.values):
            result.append(self.values[index+1])
        elif derivative < 0 and index-1 >= 0:
            result.append(self.values[index-1])

        return result

class Method:

    @staticmethod
    def dev2str(devv):
        if devv is None:
            return '#'
        elif hasattr(devv, '__iter__'):
            r = []
            for dev in devv:
                if dev > 0:
                    r.append('+')
                elif dev < 0:
                    r.append('-')
                else:
                    r.append('0')
            return r
        else:
            if devv > 0:
                return '+'
            elif devv < 0:
                return '-'
            else:
                return '0'

    def __init__(self, quantities, fixed=False,logfile=None):
        self.quantities = quantities
        self.fixed = fixed
        self.logfile=logfile

    def hash_state(self, state):
        sorted_keys = [q.name for q in self.quantities]
        result_hash = ""

        for key in sorted_keys:
            result_hash += str(key) + ": " + state[key][0] + ", " + Method.dev2str(state[key][1]) + "\n"

        return result_hash

    def state2printstring(self,state):
        s=self.hash_state(state)
        return "("+s.replace("\n", "; ").replace(", None","")+")"

    def succ(self, state):
        new_states = [{}]

        for quantity in self.quantities:
            value, derivative = state[quantity.name]
            quantity_new_states = []
            while new_states:
                new_dict = new_states.pop(0)
                for v in quantity.next_value(value, derivative):
                    value_new_dict = deepcopy(new_dict)
                    value_new_dict[quantity.name] = [v, None]
                    quantity_new_states.append(value_new_dict)
            new_states = quantity_new_states

        if self.logfile!=None:
            if new_states:

                for quantity in self.quantities:
                    value, derivative = state[quantity.name]

            else:
                print("\t\t Terminal state\n")

        vc_new_states = []
        for s in new_states:
            if not self.check_constraints(s):
                vc_new_states.append(s)
        new_states = vc_new_states

        for quantity in self.quantities:
            if quantity.hasInfluences():
                quantity_new_states = []
                while new_states:
                    new_dict = new_states.pop(0)
                    xx = self.propagate_influences(quantity, state, new_dict)

                    for deri in xx:
                        value_new_dict = deepcopy(new_dict)
                        value_new_dict[quantity.name][1] = deri
                        quantity_new_states.append(value_new_dict)
                new_states = quantity_new_states


        final_new_states = []

        for new_state in new_states:
            final_new_states += self.propagate_proportionals(state, new_state)
        return final_new_states

    def propagate_influences(self, quantity, old_state, new_state):
        new_der = set()
        change = False
        for infl_quantity, infl_positive in quantity.influences:
            _, der = old_state[infl_quantity.name]
            value, _ = new_state[infl_quantity.name]
            value_idx = infl_quantity.value2index(value)
            sign = infl_quantity.signs[value_idx]

            if sign != 0:
                if infl_positive:
                    new_der.add(sign)
                else:
                    new_der.add(-sign)
            if (der != 0):
                change = True

        if not change:
            new_der = set([old_state[quantity.name][1]])
        elif -1 in new_der and 1 in new_der:
            new_der.add(0)
        elif not new_der:
            new_der.add(0)
        new_der = new_der & set([old_state[quantity.name][1], min(1, max(-1, old_state[quantity.name][1] + 1)), min(1, max(-1, old_state[quantity.name][1] - 1))])

        return new_der

    def propagate_proportionals(self, old_state, new_state):
        new_states = [deepcopy(new_state)]

        visited_quantities = {}
        for quantity in self.quantities:
            derivatives = set()


            if quantity.name not in visited_quantities:
                visited_quantities, derivatives = self.propagate_proportional(quantity, visited_quantities, derivatives, old_state, new_state)

            quantity_new_states = []
            while new_states:
                new_dict = new_states.pop(0)
                for deri in visited_quantities[quantity.name]:
                    deri_new_dict = deepcopy(new_dict)
                    new_value = deri_new_dict[quantity.name][0]
                    value_idx = quantity.value2index(new_value)
                    interval = quantity.intervals[value_idx]
                    if (not ((deri < 0 and quantity.isMin(new_value)) or (deri > 0 and quantity.isMax(new_value))) \
                            and not (deri_new_dict[quantity.name][0] != old_state[quantity.name][0] \
                                     and interval and deri != old_state[quantity.name][1])):
                        deri_new_dict[quantity.name][1] = deri
                        quantity_new_states.append(deri_new_dict)
                    elif (deri_new_dict[quantity.name][0] != old_state[quantity.name][0] and interval and deri != old_state[quantity.name][1]) and self.logfile!=None:
                        deri_new_dict[quantity.name][1] = deri
            new_states = quantity_new_states

        if self.logfile!=None:
            for i in new_states:
                print("\t\t\t\t adding "+self.state2printstring(i)+" to the list of future states")
        return new_states

    def propagate_proportional(self, quantity, visited_quantities, derivatives, old_state, new_state):
        visited_quantities[quantity.name] = set()
        if not quantity.hasProportionals() and not quantity.hasInfluences():
            _, der = old_state[quantity.name]
            if self.fixed:
                visited_quantities[quantity.name] = set({der})
                derivatives = set({der})
            else:

                visited_quantities[quantity.name] = set({der, min(1, der+1), max(-1, der-1)})
                derivatives = set({der, min(1, der+1), max(-1, der-1)})
        elif not quantity.hasProportionals():
            _, der = new_state[quantity.name]
            visited_quantities[quantity.name].add(der)
            derivatives.add(der)
        else:
            _, der = new_state[quantity.name]
            if der is not None:

                derivatives.add(der)
                visited_quantities[quantity.name].add(der)
            if self.logfile!=None :
                pass
            for prop_quantity, prop_positive in quantity.proportionals:
                if prop_quantity.name not in visited_quantities:

                    visited_quantities, derivatives = self.propagate_proportional(prop_quantity, visited_quantities, derivatives, old_state, new_state)

                derivatives |= set([(1 if prop_positive else -1)*i for i in visited_quantities[prop_quantity.name]])
                if -1 in derivatives and 1 in derivatives:
                    derivatives.add(0)

            visited_quantities[quantity.name] = copy(derivatives)

        return visited_quantities, derivatives

    def check_constraints(self, state):
        for quantity in self.quantities:
            for other_quantity, own_val, other_val in quantity.value_constraints:
                if state[quantity.name][0] == own_val and state[other_quantity.name][0] != other_val:
                    if self.logfile!=None:
                        print("\t\t\t State "+self.state2printstring(state)+" not aligned with constraints. Dropping..")
                    return True
        return False


class Plot:

    @staticmethod
    def _status_to_str(status):
        def _dev_to_str(dev):
            if dev > 0:
                return '+'
            elif dev < 0:
                return '-'
            else:
                return '0'

        return '\n'.join((k + ':(' + str(v[0]) + ',' + _dev_to_str(v[1]) + ')' for k, v in sorted(status.items()) if k != 'child'))

    @staticmethod
    def draw(graph, filename):
        G = pgv.AGraph(directed=True, color='red')

        G.add_nodes_from((k for k, v in graph.items() if v['children']), color='red')
        G.add_nodes_from((k for k, v in graph.items() if not v['children']), color='blue')
        G.add_edges_from(((k, c) for k, v in graph.items() if 'children' in v for c in v['children']))
        G.layout(prog='dot')
        G.draw(filename)



if __name__ == "__main__":
    inflow, volume, outflow, height, pressure = extra_prob()

    prob1 = Method([inflow, volume, outflow, height, pressure], fixed=False, logfile=True)

    start_state = {"I": ["0", 1], "V": ["0", 0], "O": ["0", 0], "H": ["0", 0], "P": ["0", 0]}

    result = States.iterative(prob1, start_state)
    print(len(result))

    """ Base problem """
    # inflow, volume, outflow = base_prob()
    # base_problem = Method([inflow, volume, outflow], fixed=False, logfile=True)
    # base_start_state = {"I": ["0", 0], "V": ["0", 0], "O": ["0", 0]}
    # state_graph = States.iterative(base_problem, base_start_state)
    # Plot.draw(state_graph, 'state_graph.png')



    """ Extra problem """
    inflow, volume, height, pressure, outflow = extra_prob()
    extra_problem = Method([inflow, volume, height, pressure, outflow], fixed=False, logfile=True)
    extra_start_state = {"I": ["0", 0], "V": ["0", 0], "H": ["0", 0], "P": ["0", 0], "O": ["0", 0]}
    result_extra_problem = States.iterative(extra_problem, extra_start_state)
    Plot.draw(result_extra_problem, 'extra_states.png')

    population, birth, death = frog_prob()
    prob1 = Method([population, birth, death], fixed=True, logfile=True)
    start_state = {"Population": ["Small", 1], "Birth": ["Plus", 1], "Death": ["Plus", 1]}


