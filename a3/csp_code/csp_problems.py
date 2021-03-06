from csp import Constraint, Variable, CSP
from constraints import *
from backtracking import bt_search
import util


##################################################################
### NQUEENS
##################################################################

def nQueens(n, model):
    '''Return an n-queens CSP, optionally use tableContraints'''
    #your implementation for Question 4 changes this function
    #implement handling of model == 'alldiff'
    if not model in ['table', 'alldiff', 'row']:
        print("Error wrong sudoku model specified {}. Must be one of {}").format(
            model, ['table', 'alldiff', 'row'])

    i = 0
    dom = []
    for i in range(n):
        dom.append(i+1)

    vars = []
    for i in dom:
        vars.append(Variable('Q{}'.format(i), dom))

    cons = []

    if model == 'alldiff':
        cons.append(AllDiffConstraint("nQueens_alldiff", vars))
        for i in range(n):
          for j in range(i+1, n):
            cons.append(NeqConstraint("nQueens_neq", [vars[i], vars[j]], i+1, j+1))
    else:
        constructor = QueensTableConstraint if model == 'table' else QueensConstraint
        for qi in range(len(dom)):
            for qj in range(qi+1, len(dom)):
                con = constructor("C(Q{},Q{})".format(qi+1,qj+1),
                                            vars[qi], vars[qj], qi+1, qj+1)
                cons.append(con)

    csp = CSP("{}-Queens".format(n), vars, cons)
    return csp

def solve_nQueens(n, algo, allsolns, model='row', variableHeuristic='fixed', trace=False):
    '''Create and solve an nQueens CSP problem. The first
       parameer is 'n' the number of queens in the problem,
       The second specifies the search algorithm to use (one
       of 'BT', 'FC', or 'GAC'), the third specifies if
       all solutions are to be found or just one, variableHeuristic
       specfies how the next variable is to be selected
       'random' at random, 'fixed' in a fixed order, 'mrv'
       minimum remaining values. Finally 'trace' if specified to be
       'True' will generate some output as the search progresses.
    '''
    csp = nQueens(n, model)
    solutions, num_nodes = bt_search(algo, csp, variableHeuristic, allsolns, trace)
    print("Explored {} nodes".format(num_nodes))
    if len(solutions) == 0:
        print("No solutions to {} found".format(csp.name()))
    else:
       print("Solutions to {}:".format(csp.name()))
       i = 0
       for s in solutions:
           i += 1
           print("Solution #{}: ".format(i)),
           for (var,val) in s:
               print("{} = {}, ".format(var.name(),val), end='')
           print("")


##################################################################
### Class Scheduling
##################################################################

NOCLASS='NOCLASS'
LEC='LEC'
TUT='TUT'
class ScheduleProblem:
    '''Class to hold an instance of the class scheduling problem.
       defined by the following data items
       a) A list of courses to take

       b) A list of classes with their course codes, buildings, time slots, class types, 
          and sections. It is specified as a string with the following pattern:
          <course_code>-<building>-<time_slot>-<class_type>-<section>

          An example of a class would be: CSC384-BA-10-LEC-01
          Note: Time slot starts from 1. Ensure you don't make off by one error!

       c) A list of buildings

       d) A positive integer N indicating number of time slots

       e) A list of pairs of buildings (b1, b2) such that b1 and b2 are close 
          enough for two consecutive classes.

       f) A positive integer K specifying the minimum rest frequency. That is, 
          if K = 4, then at least one out of every contiguous sequence of 4 
          time slots must be a NOCLASS.

        See class_scheduling.py for examples of the use of this class.
    '''

    def __init__(self, courses, classes, buildings, num_time_slots, connected_buildings, 
        min_rest_frequency):
        #do some data checks
        for class_info in classes:
            info = class_info.split('-')
            if info[0] not in courses:
                print("ScheduleProblem Error, classes list contains a non-course", info[0])
            if info[3] not in [LEC, TUT]:
                print("ScheduleProblem Error, classes list contains a non-lecture and non-tutorial", info[1])
            if int(info[2]) > num_time_slots or int(info[2]) <= 0:
                print("ScheduleProblem Error, classes list  contains an invalid class time", info[2])
            if info[1] not in buildings:
                print("ScheduleProblem Error, classes list  contains a non-building", info[3])

        for (b1, b2) in connected_buildings:
            if b1 not in buildings or b2 not in buildings:
                print("ScheduleProblem Error, connected_buildings contains pair with non-building (", b1, ",", b2, ")")

        if num_time_slots <= 0:
            print("ScheduleProblem Error, num_time_slots must be greater than 0")

        if min_rest_frequency <= 0:
            print("ScheduleProblem Error, min_rest_frequency must be greater than 0")

        #assign variables
        self.courses = courses
        self.classes = classes
        self.buildings = buildings
        self.num_time_slots = num_time_slots
        self._connected_buildings = dict()
        self.min_rest_frequency = min_rest_frequency

        #now convert connected_buildings to a dictionary that can be index by building.
        for b in buildings:
            self._connected_buildings.setdefault(b, [b])

        for (b1, b2) in connected_buildings:
            self._connected_buildings[b1].append(b2)
            self._connected_buildings[b2].append(b1)

    #some useful access functions
    def connected_buildings(self, building):
        '''Return list of buildings that are connected from specified building'''
        return self._connected_buildings[building]


def solve_schedules(schedule_problem, algo, allsolns,
                 variableHeuristic='mrv', silent=False, trace=False):
    #Your implementation for Question 6 goes here.
    #
    #Do not but do not change the functions signature
    #(the autograder will twig out if you do).

    #If the silent parameter is set to True
    #you must ensure that you do not execute any print statements
    #in this function.
    #(else the output of the autograder will become confusing).
    #So if you have any debugging print statements make sure you
    #only execute them "if not silent". (The autograder will call
    #this function with silent=True, class_scheduling.py will call
    #this function with silent=False)

    #You can optionally ignore the trace parameter
    #If you implemented tracing in your FC and GAC implementations
    #you can set this argument to True for debugging.
    #
    #Once you have implemented this function you should be able to
    #run class_scheduling.py to solve the test problems (or the autograder).
    #
    #
    '''This function takes a schedule_problem (an instance of ScheduleProblem
       class) as input. It constructs a CSP, solves the CSP with bt_search
       (using the options passed to it), and then from the set of CSP
       solution(s) it constructs a list (of lists) specifying possible schedule(s)
       for the student and returns that list (of lists)

       The required format of the list is:
       L[0], ..., L[N] is the sequence of class (or NOCLASS) assigned to the student.

       In the case of all solutions, we will have a list of lists, where the inner
       element (a possible schedule) follows the format above.
    '''

    #BUILD your CSP here and store it in the varable csp
    slot_num = schedule_problem.num_time_slots
    course_num = len(schedule_problem.courses)
    min_rest_frequency = schedule_problem.min_rest_frequency
    connected_buildings = schedule_problem._connected_buildings

    # domain of each slot
    dom = [NOCLASS]
    for class_ in schedule_problem.classes:
        found = False
        for course in schedule_problem.courses:
            if course in class_:
                found = True
                break
        if found:
            dom.append(class_)

    # variables in csp
    vars = []
    for i in range(slot_num):
        vars.append(Variable('Slot{}'.format(i), dom))

    cons = []
    cons.append(NValuesConstraint("name", vars, [NOCLASS], max(0, slot_num-course_num*2), max(0, slot_num-course_num*2)))

    # for each course, one can only take one lec and one tut
    lec_candidates = {}
    tut_candidates = {}
    for i in range(1, len(dom)):
        c_info = dom[i].split("-")
        if c_info[3] == "LEC":
            if c_info[0] not in lec_candidates:
                lec_candidates[c_info[0]] = []
            lec_candidates[c_info[0]].append(dom[i])
        else:
            if c_info[0] not in tut_candidates:
                tut_candidates[c_info[0]] = []
            tut_candidates[c_info[0]].append(dom[i])

    for l in lec_candidates.values():
        cons.append(NValuesConstraint("name", vars, l, 1, 1))
    for l in tut_candidates.values():
        cons.append(NValuesConstraint("name", vars, l, 1, 1))

    # CSC108-SF-1-LEC-01 
    # 0 -> courses, 1 -> location, 2 -> timeslot, 3 -> LEC/TUT, 4 -> section num
    for i in range(slot_num):
        for j in range(i+1, slot_num):
            # for each pair of time slots, create all possible combinations of candidates
            # store them in a table constaint
            satisfying_assignments = []
            for c1 in dom:
                c1_info = c1.split("-")
                for c2 in dom:
                    c2_info = c2.split("-")
                    if c1 == NOCLASS or c2 == NOCLASS:
                        if c1 == NOCLASS and c2 != NOCLASS and (j+1) == int(c2_info[2]) or \
                           c1 != NOCLASS and c2 == NOCLASS and (i+1) == int(c1_info[2]) or \
                           c1 == NOCLASS and c2 == NOCLASS:
                            satisfying_assignments.append([c1, c2])
                        continue
                    
                    # 1. candidate needs to be at the right time slot
                    # 2. for same course, lec must comes before tut
                    # 3. for consective time slot, two buildings need to be connected
                    if c1 == c2 or \
                       (i+1) != int(c1_info[2]) or (j+1) != int(c2_info[2]) or \
                       c1_info[0] == c2_info[0] and c1_info[3] == "TUT" and c2_info[3] == "LEC" or \
                       j-i == 1 and c2_info[1] not in connected_buildings[c1_info[1]]:
                        continue
                    
                    # same course and same type, take either one of them and make the other as NOCLASS
                    if c1_info[0] == c2_info[0] and c1_info[3] == c2_info[3]:
                        satisfying_assignments.append([NOCLASS, c2])
                        satisfying_assignments.append([c1, NOCLASS])
                        continue

                    satisfying_assignments.append([c1, c2])
            # new table constraint
            cons.append(TableConstraint("name", [vars[i], vars[j]], satisfying_assignments))

    scope = []
    for i in range(min_rest_frequency-1):
        scope.append(vars[i])
    for i in range(min_rest_frequency, len(vars)):
        scope.append(vars[i])
        cons.append(NValuesConstraint("name", scope, [NOCLASS], 1, float('inf')))
        scope.pop(0)

    csp = CSP("solve_schedules", vars, cons)
    #invoke search with the passed parameters
    solutions, num_nodes = bt_search(algo, csp, variableHeuristic, allsolns, trace)

    #Convert each solution into a list of lists specifying a schedule
    #for each student in the format described above.

    #then return a list containing all converted solutions
    ans = []
    for sol in solutions:
        l = []
        for (var, val) in sol:
            l.append(val)
        ans.append(l)
    return ans


