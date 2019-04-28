#   Look for #IMPLEMENT tags in this file. These tags indicate what has
#   to be implemented to complete the warehouse domain.

#   You may add only standard python imports---i.e., ones that are automatically
#   available on TEACH.CS
#   You may not remove any imports.
#   You may not import or otherwise source any of your own files

import os  # for time functions
from search import *  # for search engines
from sokoban import SokobanState, Direction, PROBLEMS  # for Sokoban specific classes and problems


def sokoban_goal_state(state):
    '''
  @return: Whether all boxes are stored.
  '''
    for box in state.boxes:
        if box not in state.storage:
            return False
    return True


def heur_manhattan_distance(state):
    # IMPLEMENT
    '''admissible sokoban puzzle heuristic: manhattan distance'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # We want an admissible heuristic, which is an optimistic heuristic.
    # It must never overestimate the cost to get from the current state to the goal.
    # The sum of the Manhattan distances between each box that has yet to be stored and the storage point nearest to it is such a heuristic.
    # When calculating distances, assume there are no obstacles on the grid.
    # You should implement this heuristic function exactly, even if it is tempting to improve it.
    # Your function should return a numeric value; this is the estimate of the distance to the goal.
    ans = 0
    for box in state.boxes:
        min_dis = float("inf")
        for storage in state.storage:
            min_dis = min(min_dis, abs(box[0] - storage[0]) + abs(box[1] - storage[1]))
        ans += min_dis
    return ans


# SOKOBAN HEURISTICS
def trivial_heuristic(state):
    '''trivial admissible sokoban heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state (# of moves required to get) to the goal.'''
    count = 0
    for box in state.boxes:
        if box not in state.storage:
            count += 1
    return count


# def box2storage_helper(storages, boxes):
#     if len(boxes) == 0:
#         return 0
#
#     ans = float("inf")
#     box = boxes[0]
#     for i, storage in enumerate(storages):
#         cost = abs(box[0] - storage[0]) + abs(box[1] - storage[1])
#         cost += box2storage_helper(storages[:i] + storages[i + 1:], boxes[1:])
#         ans = min(ans, cost)
#     return ans
#
#
# def robot2box_helper(robots, boxes):
#     if len(boxes) == 0 or len(robots) == 0:
#         return 0
#
#     ans = float("inf")
#     box = boxes[0]
#     for i, robot in enumerate(robots):
#         cost = abs(box[0] - robot[0]) + abs(box[1] - robot[1])
#         cost += robot2box_helper(robots, boxes[1:])
#         ans = min(ans, cost)
#     return ans


def scipy_sol(robots, boxes, storages):
    if len(boxes) == 0:
        return 0

    matrix1 = []
    for robot in robots:
        row = []
        for box in boxes:
            row.append(abs(robot[0] - box[0]) + abs(robot[1] - box[1]))
        matrix1.append(row)

    matrix2 = []
    for box in boxes:
        row = []
        for storage in storages:
            row.append(abs(box[0] - storage[0]) + abs(box[1] - storage[1]))
        matrix2.append(row)

    import numpy as np
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)

    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(matrix1)
    robot2box = matrix1[row_ind, col_ind].sum()
    row_ind, col_ind = linear_sum_assignment(matrix2)
    box2storage = matrix2[row_ind, col_ind].sum()

    return robot2box + box2storage


def heur_alternate(state):
    # IMPLEMENT
    '''a better heuristic'''
    '''INPUT: a sokoban state'''
    '''OUTPUT: a numeric value that serves as an estimate of the distance of the state to the goal.'''
    # heur_manhattan_distance has flaws.
    # Write a heuristic function that improves upon heur_manhattan_distance to estimate distance between the current state and the goal.
    # Your function should return a numeric value for the estimate of the distance to the goal.

    corners = []
    width = state.width
    height = state.height
    corners.append((0, 0))
    corners.append((0, width - 1))
    corners.append((height - 1, 0))
    corners.append((height - 1, width - 1))

    for box in state.boxes:
        if box in state.storage:
            continue
        if box in corners:
            return float("inf")

    for box1 in state.boxes:
        for box2 in state.boxes:
            if box1 == box2:
                continue
            if box1 in state.storage and box2 in state.storage:
                continue
            if box1[0] == box2[0] and abs(box1[1]-box2[1]) == 1:
                if box1[0] == 0 or box1[0] == height-1:
                    return float("inf")
            if abs(box1[0]-box2[0]) == 1 and box1[1] == box2[1]:
                if box1[1] == 0 or box1[1] == width-1:
                    return float("inf")

    # available_box = []
    # for box in state.boxes:
    #     if box not in state.storage:
    #         available_box.append(box)
    #
    # available_storage = []
    # for storage in state.storage:
    #     if storage not in state.boxes:
    #         available_storage.append(storage)
    #
    # box2storage = box2storage_helper(available_storage, available_box)
    # robot2box = robot2box_helper(state.robots, available_box)
    # return box2storage + robot2box
    return scipy_sol(state.robots, state.boxes, state.storage)


def heur_zero(state):
    '''Zero Heuristic can be used to make A* search perform uniform cost search'''
    return 0


def fval_function(sN, weight):
    # IMPLEMENT
    """
    Provide a custom formula for f-value computation for Anytime Weighted A star.
    Returns the fval of the state contained in the sNode.

    @param sNode sN: A search node (containing a SokobanState)
    @param float weight: Weight given by Anytime Weighted A star
    @rtype: float
    """

    # Many searches will explore nodes (or states) that are ordered by their f-value.
    # For UCS, the fvalue is the same as the gval of the state. For best-first search, the fvalue is the hval of the state.
    # You can use this function to create an alternate f-value for states; this must be a function of the state and the weight.
    # The function must return a numeric f-value.
    # The value will determine your state's position on the Frontier list during a 'custom' search.
    # You must initialize your search engine object as a 'custom' search engine if you supply a custom fval function.
    return sN.gval + weight * sN.hval


def anytime_weighted_astar(initial_state, heur_fn, weight=1., timebound=10):
    # IMPLEMENT
    '''Provides an implementation of anytime weighted a-star, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of weighted astar algorithm'''
    start_time = os.times()[0]
    se = SearchEngine('custom', 'full')
    wrapped_fval_function = (lambda sN: fval_function(sN, weight))
    se.init_search(initial_state, sokoban_goal_state, heur_fn, wrapped_fval_function)

    ans = se.search(timebound, (float("inf"), float("inf"), float("inf")))
    if not ans:
        return ans

    while os.times()[0] - start_time < timebound:
        weight = max(1, 0.8 * weight)
        candidate = se.search(start_time + timebound - os.times()[0], (float("inf"), float("inf"), ans.gval))
        if not candidate:
            return ans
        ans = candidate

    return ans


def anytime_gbfs(initial_state, heur_fn, timebound=10):
    # IMPLEMENT
    '''Provides an implementation of anytime greedy best-first search, as described in the HW1 handout'''
    '''INPUT: a sokoban state that represents the start state and a timebound (number of seconds)'''
    '''OUTPUT: A goal state (if a goal is found), else False'''
    '''implementation of weighted astar algorithm'''
    start_time = os.times()[0]
    se = SearchEngine('best_first', 'full')
    se.init_search(initial_state, sokoban_goal_state, heur_fn)

    ans = se.search(timebound, (float("inf"), float("inf"), float("inf")))
    if not ans:
        return ans

    while os.times()[0] - start_time < timebound:
        candidate = se.search(start_time + timebound - os.times()[0], (ans.gval, float("inf"), float("inf")))
        if not candidate:
            return ans
        ans = candidate

    return ans
