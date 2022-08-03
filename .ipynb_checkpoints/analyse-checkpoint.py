#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:01:34 2020

@author: jacobb
"""
import numpy as np
import torch
import pdb
import copy
from itertools import permutations
from scipy.sparse.csgraph import shortest_path

# Track prediction accuracy over walk, and calculate fraction of locations visited and actions taken to assess performance
def performance(forward, model, environments):
    # Keep track of whether model prediction were correct, as well as the fraction of nodes/edges visited, across environments
    all_correct, all_location_frac, all_action_frac = [], [], []
    # Run through environments and monitor performance in each
    for env_i, env in enumerate(environments):
        # Keep track for each location whether it has been visited
        location_visited = np.full(env.n_locations, False)
        # And for each action in each location whether it has been taken
        action_taken = np.full((env.n_locations,model.hyper['n_actions']), False)
        # Not all actions are available at every location (e.g. edges of grid world). Find how many actions can be taken
        # Impossible actions should be all 0s
        action_available = np.full((env.n_locations,model.hyper['n_actions']), False)
        for currLocation in env.locations:
            for currAction in currLocation['actions']:
                if np.sum(currAction['transition']) > 0:
                    if model.hyper['has_static_action']:
                        if currAction['id'] > 0:
                            action_available[currLocation['id'], currAction['id'] - 1] = True
                    else:
                        action_available[currLocation['id'], currAction['id']] = True                    
        # Make array to list whether the observation was predicted correctly or not
        correct = []  
        # Make array that stores for each step the fraction of locations visited
        location_frac = []
        # And an array that stores for each step the fraction of actions taken
        action_frac = []
        # Run through iterations of forward pass to check when an action is taken for the first time
        for step in forward:
            # Update the states that have now been visited
            location_visited[step.g[env_i]['id']] = True
            # ... And the actions that now have been taken
            if model.hyper['has_static_action']:
                if step.a[env_i] > 0:
                    action_taken[step.g[env_i]['id'], step.a[env_i] - 1] = True
            else:
                action_taken[step.g[env_i]['id'], step.a[env_i]] = True                    
            # Mark the location of the previous iteration as visited
            correct.append((torch.argmax(step.x_gen[2][env_i]) == torch.argmax(step.x[env_i])).numpy())
            # Add the fraction of locations visited for this step
            location_frac.append(np.sum(location_visited) / location_visited.size)
            # ... And also add the fraction of actions taken for this step
            action_frac.append(np.sum(action_taken) / np.sum(action_available))
        # Add performance and visitation fractions of this environment to performance list across environments
        all_correct.append(correct)
        all_location_frac.append(location_frac)
        all_action_frac.append(action_frac)
    # Return 
    return all_correct, all_location_frac, all_action_frac

# Track prediction accuracy per location, after a transition towards the location 
def location_accuracy(forward, model, environments):
    # Keep track of whether model prediction were correct for each environment, separated by arrival and departure location
    accuracy_from, accuracy_to = [], []
    # Run through environments and monitor performance in each
    for env_i, env in enumerate(environments):
        # Make array to list whether the observation was predicted correctly or not
        correct_from = [[] for _ in range(env.n_locations)]  
        correct_to = [[] for _ in range(env.n_locations)]          
        # Run through iterations of forward pass to check when an action is taken for the first time
        for step_i, step in enumerate(forward[1:]):
            # Prediction on arrival: sensory prediction when arriving at given node
            correct_to[step.g[env_i]['id']].append((torch.argmax(step.x_gen[2][env_i]) == torch.argmax(step.x[env_i])).numpy().tolist())
            # Prediction on depature: sensory prediction after leaving given node - i.e. store whether the current prediction is correct for the previous location
            correct_from[forward[step_i].g[env_i]['id']].append((torch.argmax(step.x_gen[2][env_i]) == torch.argmax(step.x[env_i])).numpy().tolist())
        # Add performance and visitation fractions of this environment to performance list across environments
        accuracy_from.append([sum(correct_from_location) / (len(correct_from_location) if len(correct_from_location) > 0 else 1) for correct_from_location in correct_from])
        accuracy_to.append([sum(correct_to_location) / (len(correct_to_location) if len(correct_to_location) > 0 else 1) for correct_to_location in correct_to])
    # Return 
    return accuracy_from, accuracy_to

# Track occupation per location
def location_occupation(forward, model, environments):
    # Keep track of how many times each location was visited
    occupation = []
    # Run through environments and monitor performance in each
    for env_i, env in enumerate(environments):
        # Make array to list whether the observation was predicted correctly or not
        visits = [0 for _ in range(env.n_locations)]  
        # Run through iterations of forward pass to check when an action is taken for the first time
        for step in forward:
            # Prediction on arrival: sensory prediction when arriving at given node
            visits[step.g[env_i]['id']] += 1
        # Add performance and visitation fractions of this environment to performance list across environments
        occupation.append(visits)
    # Return occupation of states during walk across environments
    return occupation

# Measure zero-shot inference for this model: see if it can predict an observation following a new action to a know location
def zero_shot(forward, model, environments, include_stay_still=True):
    # Get the number of actions in this model
    n_actions = model.hyper['n_actions'] + model.hyper['has_static_action']
    # Track for all opportunities for zero-shot inference if the predictions were correct across environments
    all_correct_zero_shot = []
    # Run through environments and check for zero-shot inference in each of them
    for env_i, env in enumerate(environments):
        # Keep track for each location whether it has been visited
        location_visited = np.full(env.n_locations, False)
        # And for each action in each location whether it has been taken
        action_taken = np.full((env.n_locations, n_actions), False)
        # Get the very first iteration
        prev_iter = forward[0]
        # Make list that for all opportunities for zero-shot inference tracks if the predictions were correct
        correct_zero_shot = []
        # Run through iterations of forward pass to check when an action is taken for the first time
        for step in forward[1:]:
            # Get the previous action and previous location location
            prev_a, prev_g = prev_iter.a[env_i], prev_iter.g[env_i]['id']
            # If the previous action was standing still: only count as valid transition standing still actions are included as zero-shot inference
            if model.hyper['has_static_action'] and prev_a == 0 and not include_stay_still:
                prev_a = None
            # Mark the location of the previous iteration as visited
            location_visited[prev_g] = True
            # Zero shot inference occurs when the current location was visited, but the previous action wasn't taken before
            if location_visited[step.g[env_i]['id']] and prev_a is not None and not action_taken[prev_g, prev_a]:
                # Find whether the prediction was correct
                correct_zero_shot.append((torch.argmax(step.x_gen[2][env_i]) == torch.argmax(step.x[env_i])).numpy())
            # Update the previous action as taken
            if prev_a is not None:
                action_taken[prev_g, prev_a] = True
            # And update the previous iteration to the current iteration
            prev_iter = step
        # Having gone through the full forward pass for one environment, add the zero-shot performance to the list of all 
        all_correct_zero_shot.append(correct_zero_shot)
    # Return lists of success of zero-shot inference for all environments
    return all_correct_zero_shot

# Compare TEM performance to a 'node' and an 'edge' agent, that remember previous observations and guess others
def compare_to_agents(forward, model, environments, include_stay_still=True):
    # Get the number of actions in this model
    n_actions = model.hyper['n_actions'] + model.hyper['has_static_action']
    # Store for each environment for each step whether is was predicted correctly by the model, and by a perfect node and perfect edge agent
    all_correct_model, all_correct_node, all_correct_edge = [], [], []
    # Run through environments and check for correct or incorrect prediction
    for env_i, env in enumerate(environments):
        # Keep track for each location whether it has been visited
        location_visited = np.full(env.n_locations, False)
        # And for each action in each location whether it has been taken
        action_taken = np.full((env.n_locations, n_actions), False)
        # Make array to list whether the observation was predicted correctly or not for the model
        correct_model = []
        # And the same for a node agent, that picks a random observation on first encounter of a node, and the correct one every next time
        correct_node = []  
        # And the same for an edge agent, that picks a random observation on first encounter of an edge, and the correct one every next time
        correct_edge = []          
        # Get the very first iteration
        prev_iter = forward[0]
        # Run through iterations of forward pass to check when an action is taken for the first time
        for step in forward[1:]:
            # Get the previous action and previous location location
            prev_a, prev_g = prev_iter.a[env_i], prev_iter.g[env_i]['id']
            # If the previous action was standing still: only count as valid transition if standing still actions are included as zero-shot inference
            if model.hyper['has_static_action'] and prev_a == 0 and not include_stay_still:
                prev_a = None
            # Mark the location of the previous iteration as visited
            location_visited[prev_g] = True
            # Update model prediction for this step
            correct_model.append((torch.argmax(step.x_gen[2][env_i]) == torch.argmax(step.x[env_i])).numpy())
            # Update node agent prediction for this step: correct when this state was visited beofre, otherwise chance
            correct_node.append(True if location_visited[step.g[env_i]['id']] else np.random.randint(model.hyper['n_x']) == torch.argmax(step.x[env_i]).numpy())
            # Update edge agent prediction for this step: always correct if no action taken, correct when action leading to this state was taken before, otherwise chance
            correct_edge.append(True if prev_a is None else True if action_taken[prev_g, prev_a] else np.random.randint(model.hyper['n_x']) == torch.argmax(step.x[env_i]).numpy())
            # Update the previous action as taken
            if prev_a is not None:
                action_taken[prev_g, prev_a] = True
            # And update the previous iteration to the current iteration
            prev_iter = step
        # Add the performance of model, node agent, and edge agent for this environment to list across environments
        all_correct_model.append(correct_model)
        all_correct_node.append(correct_node)
        all_correct_edge.append(correct_edge)
    # Return list of prediction success for all three agents across environments
    return all_correct_model, all_correct_node, all_correct_edge

# Calculate rate maps for this model: what is the firing pattern for each cell at all locations?
def rate_map(forward, model, environments):
    #print(type(forward))
    #print(forward)
    # Store location x cell firing rate matrix for abstract and grounded location representation across environments    
    all_g, all_p = [], []
    # Go through environments and collect firing rates in each
    for env_i, env in enumerate(environments):
        # Collect grounded location/hippocampal/place cell representation during walk: separate into frequency modules, then locations
        p = [[[] for loc in range(env.n_locations)] for f in range(model.hyper['n_f'])]
        # Collect abstract location/entorhinal/grid cell representation during walk: separate into frequency modules, then locations
        g = [[[] for loc in range(env.n_locations)] for f in range(model.hyper['n_f'])]
        # In each step, concatenate the representations to the appropriate list
        for step in forward:
            # Run through frequency modules and append the firing rates to the correct location list
            for f in range(model.hyper['n_f']):
                #print('step.g_inf: {0}'.format(step.g_inf[f][env_i].numpy()))
                #print('step.p_inf: {0}'.format(step.p_inf[f][env_i].numpy()))
                g[f][step.g[env_i]['id']].append(step.g_inf[f][env_i].numpy())
                p[f][step.g[env_i]['id']].append(step.p_inf[f][env_i].numpy())
        # Now average across location visits to get a single representation vector for each location for each frequency
        for cells, n_cells in zip([p, g], [model.hyper['n_p'], model.hyper['n_g']]):
            #print('len(cells): {0}'.format(len(cells)))
            for f, frequency in enumerate(cells):
                #print('len(frequency): {0}'.format(len(frequency)))
                # Average across visits of the each location, but only the second half of the visits so model roughly knows the environment
                for l, location in enumerate(frequency):
                    #print('len(location): {0}'.format(len(location)))
                    frequency[l] = sum(location[int(len(location)/2):]) / len(location[int(len(location)/2):]) if len(location[int(len(location)/2):]) > 0 else np.zeros(n_cells[f])
                # Then concatenate the locations to get a [locations x cells for this frequency] matrix
                cells[f] = np.stack(frequency, axis=0)
                #print('len(cells[f]): {0}'.format(len(cells[f])))
        # Append the final average representations of this environment to the list of representations across environments
        all_g.append(g)
        all_p.append(p)
    #print('{0}'.format(len(all_p))) 4 environments
    #print('{0}'.format(len(all_p[0]))) 5 frequencies
    #print('{0}'.format(len(all_p[0][0]))) 29 locations
    #print('{0}'.format(len(all_p[0][0][0]))) 100 cells
    #print('{0}'.format(len(all_p[0][0][0][0]))) int
    # Return list of locations x cells matrix of firing rates for each frequency module for each environment
    #print('all_g[0][0][0]: {0}\n'.format(all_g[0][0][0]))
    #print('all_p: {0}]\n'.format(all_p))::-1]
    
    return all_g, all_p

def reverse_trajectory(trace):
    
    return trace[::-1]

def path_dependence(three_arm_trajectories):
    
    # left -> center, center -> left, center -> right, right -> center
    LC, CL, CR, RC = three_arm_trajectories
    
    # The center (shared) portion of every trajectory
    LC_C = LC[-4:]
    RC_C = RC[-4:]
    CL_C = CL[1:4]
    CR_C = CR[1:4]
    
    # both inbound compare destinations
    inbound_overlap = normalized_overlap(LC_C, RC_C)
    
    # both outbound compare destinations
    outbound_overlap = normalized_overlap(CL_C, CR_C)
    
    # both left trajectories compare direction
    left_trajectories_overlap = normalized_overlap(LC, CL[::-1])
    
    # both right trajectories compare direction
    right_trajectories_overlap = normalized_overlap(RC, CR[::-1])
    
    return inbound_overlap, outbound_overlap, left_trajectories_overlap, right_trajectories_overlap
    
def plot_path_dependence(three_arm_trajectories):
    
    fig, axs = plt.subplots()
    
    axs[0].plot()
    
def make_all_p_nonnegative(p_env, normalize_only_neighbor=False):
    all_p_nonnegative = copy.deepcopy(p_env)
    for frequency in range(len(p_env)):
        for trajectory in range(len(p_env[frequency])):
            for cell_num in range(len(p_env[frequency][trajectory][0])):
                traj_by_loc_by_cell = np.array(p_env[frequency])
                all_traj_min = np.min(traj_by_loc_by_cell[:, :, cell_num][traj_by_loc_by_cell[:, :, cell_num] != -999])
                all_traj_max = np.max(traj_by_loc_by_cell[:, :, cell_num][traj_by_loc_by_cell[:, :, cell_num] != -999] - all_traj_min)
                if all_traj_min != 0 and all_traj_max != 0:
                    trajectory_loc_by_cell = all_p_nonnegative[frequency][trajectory][:, cell_num]
                    # Shift and scale all non -999 values
                    trajectory_loc_by_cell[trajectory_loc_by_cell != -999] = trajectory_loc_by_cell[trajectory_loc_by_cell != -999] - all_traj_min
                    trajectory_loc_by_cell[trajectory_loc_by_cell != -999] = trajectory_loc_by_cell[trajectory_loc_by_cell != -999] / all_traj_max
                    all_p_nonnegative[frequency][trajectory][:, cell_num] = trajectory_loc_by_cell
                    #print(trajectory)
    return all_p_nonnegative

def get_path(pred, start, goal):
    path = [goal]
    k = goal
    while pred[start, k] != -9999:
        path.append(pred[start, k])
        k = pred[start, k]
    return path[::-1]

def get_trajectory_states(trajectory, environment):
    
    trajectory_dict = make_trajectory_dict()
    # Calculate graph properties for shortest path calculation between arms
    adjacency = np.array(environment.adjacency) 
    dists, pred = shortest_path(adjacency, directed=False, method='FW', return_predecessors=True)

    curr_arm_end_state = trajectory[0] * 5
    new_arm_end_state = trajectory[1] * 5
    # Using the adjacency matrix, create a trajectory using a shortest path traversal from end of curr_arm to end of new_arm
    #trajectory = self.get_path(pred, curr_arm_end_state, new_arm_end_state)[1:]
    #print(trajectory)
    
    return [environment.locations[state_id]['id'] for state_id in get_path(pred, curr_arm_end_state, new_arm_end_state)][1:]

def normalized_overlap_path_equivalence(traj_a_ident, traj_b_ident, trace_a, trace_b, environment, zero_trajectory_threshold=0, impute_val=-1):

    traj_a_state_seq = np.array(get_trajectory_states(traj_a_ident, environment))
    traj_b_state_seq = np.array(get_trajectory_states(traj_b_ident, environment))
    
    unambiguous_trajectory_locations = traj_a_state_seq != traj_b_state_seq
    
    unambiguous_trace_a = trace_a[unambiguous_trajectory_locations]
    unambiguous_trace_b = trace_b[unambiguous_trajectory_locations]
    
    return normalized_overlap_traces(unambiguous_trace_a, unambiguous_trace_b, zero_trajectory_threshold, impute_val)

def normalized_overlap_path_dependence(traj_a_ident, traj_b_ident, trace_a, trace_b, environment, zero_trajectory_threshold=0, impute_val=1):

    traj_a_state_seq = np.array(get_trajectory_states(traj_a_ident, environment))
    traj_b_state_seq = np.array(get_trajectory_states(traj_b_ident, environment))
    
    # For opposite_direction trajectories
    if traj_a_ident == traj_b_ident[::-1]:
        
        # Flip traj_b before comparing
        traj_b_state_seq = traj_b_state_seq[::-1]
        trace_b = trace_b[::-1]
        # Remove states that don't line up for trajectories a and b
        traj_a_state_seq = traj_a_state_seq[:-1]
        shared_trace_a = trace_a[:-1]
        traj_b_state_seq = traj_b_state_seq[1:]
        shared_trace_b = trace_b[1:]
        
    else:
        
        shared_trajectory_locations = traj_a_state_seq == traj_b_state_seq

        shared_trace_a = trace_a[shared_trajectory_locations]
        shared_trace_b = trace_b[shared_trajectory_locations]

    return normalized_overlap_traces(shared_trace_a, shared_trace_b, zero_trajectory_threshold, impute_val)

# Allow negative and positive traces
def normalized_overlap_traces(trace_a, trace_b, zero_trajectory_threshold=0, impute_val=1):
    
    # If trace maximum is below threshold, set this trajectory trace to 0
    max_a = np.max(np.abs(trace_a))
    max_b = np.max(np.abs(trace_b))
    
    if max_a < zero_trajectory_threshold:
        trace_a = np.zeros(trace_a.shape)
    if max_b < zero_trajectory_threshold:
        trace_b = np.zeros(trace_b.shape)
    
    # Find regions where traces are on the same side vs the opposite side
    same_side = np.sign(trace_a * trace_b)
    min_trace = np.min(np.abs(np.vstack([trace_a, trace_b])), axis=0)
    min_trace_area = np.sum(same_side * min_trace)
    
    #print('min_trace: {0}'.format(min_trace))
    #print('sum: {0}'.format(np.sum(np.vstack([trace_a, trace_b]), axis=0)))
    #elementwise_overlap = 2 * min_trace / np.sum(np.abs(np.vstack([trace_a, trace_b])), axis=0)
    #print('elementwise_overlap0: {0}'.format(elementwise_overlap))
    #elementwise_overlap[np.isnan(elementwise_overlap)] = 0
    #print('elementwise_overlap1: {0}'.format(elementwise_overlap))
    #total_overlap = np.sum(same_side * elementwise_overlap)
    area_a, area_b = np.sum(np.abs(trace_a)), np.sum(np.abs(trace_b))
    total_overlap = 2 * min_trace_area / (area_a + area_b) if (area_a + area_b != 0) else impute_val
    return total_overlap

def trajectory_len(trajectory):
    return abs(trajectory[1] - trajectory[0]) * 2 + 6

def make_trajectory_dict():
    trajectories = permutations(np.arange(6), r=2)
    trajectory_dict = {}
    for traj_idx, traj in enumerate(trajectories):
        trajectory_dict[traj] = traj_idx
    return trajectory_dict

# Calculate rate maps for this model: what is the firing pattern for each cell at all locations?
def trajectories_rate_maps(forward, model, environments, trajectories=list(permutations(np.arange(6), r=2))):
    #print(type(forward))
    #print(forward)
    # Make dictionary to easily convert from trajectory tuple to idx in trajectories list
    trajectory_dict = {}
    for traj_idx, traj in enumerate(trajectories):
        trajectory_dict[traj] = traj_idx
    # Make an array where every element is the current trajectory for step_i in the test walk
    trajectory_labels = np.ones(shape=(len(environments), len(forward))) * -1
    arm_visit_times = [[] for env in environments]
    for env_i, env in enumerate(environments):
        for step_i, step in enumerate(forward):
            if not step.g[env_i]['id'] % 5:
                arm_visit_times[env_i].append((step_i, int(step.g[env_i]['id'] / 5)))
        #print(arm_visit_times[env_i])
        for visit_idx in np.arange(len(arm_visit_times[env_i][:-1])):
            arm_visit_1, arm_visit_2 = arm_visit_times[env_i][visit_idx], arm_visit_times[env_i][visit_idx+1]
            # +1 is so that we count reward visits as being at the end of trajectories
            #print([trajectory_dict[(arm_visit_1[1], arm_visit_2[1])]])
            trajectory_labels[env_i][arm_visit_1[0]+1:arm_visit_2[0]+1] = [trajectory_dict[(arm_visit_1[1], arm_visit_2[1])]] * (arm_visit_2[0] - arm_visit_1[0])
    # Store location x cell firing rate matrix for abstract and grounded location representation across environments
    all_g, all_p = [], []
    # Go through environments and collect firing rates in each
    for env_i, env in enumerate(environments):
        # Collect grounded location/hippocampal/place cell representation during walk: separate into frequency modules, then locations
        p = [[[[] for loc in range(env.n_locations)] for trajectory_i in range(len(trajectories))] for f in range(model.hyper['n_f'])]
        # Collect abstract location/entorhinal/grid cell representation during walk: separate into frequency modules, then locations
        g = [[[[] for loc in range(env.n_locations)] for trajectory_i in range(len(trajectories))] for f in range(model.hyper['n_f'])]
        #print(len(g))
        #print(len(g[0]))
        #print(len(g[0][0]))
        #print(len(g[0][0][0]))
        #print('g[0][1][1]: {0}'.format(g[0][1][1]))
        #print(g)
        #print(trajectory_labels)
        for trajectory_i, trajectory in enumerate(trajectories):

            # In each step, concatenate the representations to the appropriate list
            for step_i, step in enumerate(forward[1:]):
                # Skip first location since it doesn't technically have a trajectory
                # assigned to it. But make sure that step number is corrected for this
                step_i = step_i + 1
                # Run through frequency modules and append the firing rates to the correct location list
                for f in range(model.hyper['n_f']):
                    #print('step.g_inf: {0}'.format(step.g_inf[f][env_i].numpy()))
                    #print('step.p_inf: {0}'.format(step.p_inf[f][env_i].numpy()))
                    # If the step is on this particular trajectory
                    # trajectory_labels is of len(forward). Each item is the trajectory the animal was on at
                    # during step_i
                    if trajectory_labels[env_i][step_i] == trajectory_i:
                        #print('trajectory: {0}\ntrajectory_i: {1}\nf: {2}\nenv_i: {3}\nstep_i: {4}\ntrajectory_labels[env_i][step_i]: {5}\nlocation: {6}\n'.format(trajectory, trajectory_i, f, env_i, step_i, trajectory_labels[env_i][step_i], step.g[env_i]['id']))
                        #print(g[f])
                        #print('g[0][1][1]: {0}'.format(g[0][1][1]))
                        #print(g[f][trajectory_i])
                        #print('g[f][trajectory_i][step.g[env_i][\'id\']]: {0}'.format(g[f][trajectory_i][step.g[env_i]['id']]))
                        g[f][trajectory_i][step.g[env_i]['id']].append(step.g_inf[f][env_i].numpy())
                        p[f][trajectory_i][step.g[env_i]['id']].append(step.p_inf[f][env_i].numpy())
                        #print(g[f][trajectory_i][step.g[env_i]['id']])
        # Now average across location visits to get a single representation vector for each location for each frequency
        for cells, n_cells in zip([p, g], [model.hyper['n_p'], model.hyper['n_g']]):
            #print('len(cells): {0}'.format(len(cells)))
            for f, frequency in enumerate(cells):
                #print('len(frequency): {0}'.format(len(frequency)))
                # Average across visits of the each location, but only the second half of the visits so model roughly know the environment
                # Do this separately for each trajectory
                for trajectory_i, trajectory in enumerate(frequency):
                    for l, location in enumerate(trajectory):
                        #print('len(location): {0}'.format(len(location)))
                        frequency[trajectory_i][l] = sum(location[int(len(location)/2):]) / len(location[int(len(location)/2):]) if (len(location[int(len(location)/2):]) > 0) else np.ones(n_cells[f]) * -999
# Then concatenate the locations to get a [locations x trajectories x cells for this frequency] matrix
                    cells[f][trajectory_i] = np.stack(trajectory, axis=0)
                #print('len(cells[f]): {0}'.format(len(cells[f])))
            # Append the final average representations of this environment to the list of representations across environments
            all_g.append(g)
            all_p.append(p)
        #print('{0}'.format(len(all_p))) 4 environments
        #print('{0}'.format(len(all_p[0]))) 5 frequencies
        #print('{0}'.format(len(all_p[0][0]))) 30 trajectories
        #print('{0}'.format(len(all_p[0][0][0]))) variable n locations
        #print('{0}'.format(len(all_p[0][0][0][0]))) 100 cells
        #print('{0}'.format(len(all_p[0][0][0][0][0]))) int
        # Return list of locations x cells matrix of firing rates for each frequency module for each environment
        #print('all_g[0][0][0]: {0}\n'.format(all_g[0][0][0]))
        #print('all_p: {0}]\n'.format(all_p))
    return all_g, all_p

# Helper function to generate input for the model
def generate_input(environment, walk):
    # If no walk was provided: use the environment to generate one
    if walk is None:
        # Generate a single walk from environment with length depending on number of locations (so you're likely to visit each location)
        walk = environment.generate_walks(environment.graph['n_locations']*100, 1)[0]
        # Now this walk needs to be adjusted so that it looks like a batch with batch size 1
        for step in walk:
            # Make single location into list
            step[0] = [step[0]]
            # Make single 1D observation vector into 2D row vector
            step[1] = step[1].unsqueeze(dim=0)
            # Make single action into list
            step[2] = [step[2]]
    return walk

# Smoothing function (originally written by James)
def smooth(a, wsz):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    out0 = np.convolve(a, np.ones(wsz, dtype=int), 'valid') / wsz
    r = np.arange(1, wsz - 1, 2)
    start = np.cumsum(a[:wsz - 1])[::2] / r
    stop = (np.cumsum(a[:-wsz:-1])[::2] / r)[::-1]
    return np.concatenate((start, out0, stop))