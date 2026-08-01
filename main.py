# 0. Initialization

import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter


# DEFINE
MINIMUM_MINUTES = 100 # remove reserve players, +100 to save time
POPULATION_COUNT = 300
MINUTE_THRESHOLD = 2000 # lower than x minutes, players will get a linear penalty aplpied to thier score
POINT_BIAS = 0.60
ICT_BIAS = 0.40
STARTERS_AND_SUBS = True # set True to penalize the subs, to more accuaretly reflect starting XI importance
SUB_PENALTY = 0.20 # a sub player is worth x times less a starter
BUDGET = 1000  # 1000 base
COST_PENALTY = 0.002 # 2% penalty per million (10 units) -> 0.1% penalty per unit
MAX_TEAM_COUNT = 3 # standard fantasy draft limit
TEAM_OVER_PENALTY = 0.10 # 10% penalty per violation
PROPORTIONAL_METHOD  = True # set True to use proportional, False to use tournamenet
TOURNAMENT_SIZE = 2 # default 2. only used if above is False
INTERVAL_TO_IMPROVE = 100 # how many generations to find an improvement?
GENERATIONS = 500
CROSSOVER_RATE = 0.60 # 60% base crossovers
MUTATION_RATE = 0.10 # 10% base mutations
ELITSM = True # use elitism
# hyperparameter guide: https://www.mdpi.com/2078-2489/10/12/390

# algorthim source: https://www.datacamp.com/tutorial/genetic-algorithm-python

# download dataset. originally from:
# https://github.com/vaastav/Fantasy-Premier-League/blob/master/data/2025-26/players_raw.csv#L6
if not os.path.exists('players_raw.csv'):
    !gdown "1_KsLNuVyEKQSE4aPGODLZF2KNh898aSq" -O players_raw.csv

# relevant fields in file
fields = [
    'id', # PLayer ID
    'first_name',
    'second_name',
    'team', # Players team
    'element_type', # Position (1=GK, 2=DEF, 3=MID, 4=ATT)
    'now_cost', # Player value (1000 budget)
    'total_points', # Points earned
    'minutes', # Minutes played
    'ict_index', # ICT_index (custom metric to measure players influence, creativity and threat)
    'selected_by_percent' # Selected percent
]
init_data = pd.read_csv('players_raw.csv', usecols=fields)

# consdier: https://www.fotmob.com/leagues/47/stats/season/27110/players/mins_played/premier-league-players
cleaned_data = init_data[init_data['minutes'] >= MINIMUM_MINUTES] # apply minimum minutes needed


# combine first and last name to handle edge case names
first_names = cleaned_data['first_name'].fillna('')
second_names = cleaned_data['second_name'].fillna('')
full_names = (first_names + ' ' + second_names).str.strip()

id_name = pd.Series(full_names.values, index=cleaned_data['id']).to_dict()

# force the column order and index for the final data element, removing name fields
ordered_columns = [
    'id', # 0
    'team', # 1
    'element_type', # 2
    'now_cost', # 3
    'total_points', # 4
    'minutes', # 5
    'ict_index', # 6
    'selected_by_percent' # 7
    # 8 final score - see below
]
final_data = cleaned_data[ordered_columns]

# dictionary where key = ID, value = array of stats following above index
player_dict = {}
for x in final_data.to_numpy():
    player_id = int(x[0])
    player_dict[player_id] = list(x) # IMPORTANT: Convert numpy array to list so we can append!

# create stats (points and ict) normalzied from 0 to 100 for a more compareable and understandble scoring
max_points_per_90 = max(((p[4] / p[5]) * 90) for p in player_dict.values() if p[5] >= MINUTE_THRESHOLD)
max_ICT_per_90 = max(((p[6] / p[5]) * 90) for p in player_dict.values() if p[5] >= MINUTE_THRESHOLD)
#max_points_per_90 = max((p[4]) for p in player_dict.values())
#max_ICT_per_90 = max((p[6]) for p in player_dict.values())
for player_id, data in player_dict.items():
    player_minutes = data[5]

    # calculate this players per 90 stat
    points_per_90 = (data[4] / player_minutes) * 90
    ICT_per_90 = (data[6] / player_minutes) * 90
    #points_per_90 = data[4]
    #ICT_per_90 = data[6]

    # normalize in relation to max
    normalzied_points = points_per_90 / max_points_per_90
    normalized_ICT = ICT_per_90 / max_ICT_per_90

    # apply bias
    raw_score = (POINT_BIAS * normalzied_points) + (ICT_BIAS * normalized_ICT)

    # use a saturation function to penalize low minutes players who may have unnatural high points or ict
    if player_minutes >= MINUTE_THRESHOLD:
        saturation_multiplier = 1.0
    else:
        saturation_multiplier = (player_minutes / MINUTE_THRESHOLD) # essnetialy divies out player minutes and instead used threshold value

    final_score = (raw_score * saturation_multiplier) * 100

    # append to the list (inside the final index: 8)
    data.append(final_score)


# split along positions
final_data_gk = final_data[final_data['element_type'] == 1]
final_data_def = final_data[final_data['element_type'] == 2]
final_data_mid = final_data[final_data['element_type'] == 3]
final_data_att = final_data[final_data['element_type'] == 4]

# create player numpy array from pandas
player_pool_gk = final_data_gk.to_numpy()
player_pool_def = final_data_def.to_numpy()
player_pool_mid = final_data_mid.to_numpy()
player_pool_att = final_data_att.to_numpy()
#rint(f"Gk count: {player_pool_gk.shape[0]}")
#print(f"Def count: {player_pool_def.shape[0]}")
#print(f"Mid count: {player_pool_mid.shape[0]}")
#print(f"Att count: {player_pool_att.shape[0]}")
#print(f"Total remaining count: {final_data.shape[0]}")


# Ranking step
all_players_sorted = sorted(player_dict.values(), key=lambda p: p[-1], reverse=True)

# Format: {player_id: rank_integer}
player_rank_lookup = {}
for rank, player_data in enumerate(all_players_sorted):
    player_rank_lookup[int(player_data[0])] = rank + 1

# Print the Top X players
print(f"{'Rank':<6} | {'Player Name':<34} | {'Score':<8} | {'Pts':<5} | {'ICT':<6} | {'Min':<6}")
print(f"{'-'*85}")

for i in range(20):
    p_data = all_players_sorted[i]
    p_id = int(p_data[0])
    p_name = id_name[p_id]
    p_pts = p_data[4]
    p_min = p_data[5]
    p_ict = p_data[6]
    p_score = p_data[-1]

    print(f"#{i+1:<5} | {p_name:<34} | {p_score:>8.2f} | {p_pts:>5} | {p_ict:>6.1f} | {p_min:>6.1f}")

# Helper Functions

def fitness_function(chromosome, player_dict):
    chromosome_cost = 0
    chromosome_score = 0
    team_list = np.empty(15, dtype=int)
    itr = 0

    gk_line = []
    def_line = []
    mid_line = []
    att_line = []
    flat_squad = np.concatenate(chromosome) # flatten array
    for player_id in flat_squad:
        player_data = player_dict[int(player_id)]

        chromosome_cost += player_data[3] # track cost fo this team

        if STARTERS_AND_SUBS:
            # divide into each position line
            if player_data[2] == 1:
                gk_line.append(player_data[-1])
            elif player_data[2] == 2:
                def_line.append(player_data[-1])
            elif player_data[2] == 3:
                mid_line.append(player_data[-1])
            elif player_data[2] == 4:
                att_line.append(player_data[-1])

        else:
            chromosome_score += player_data[-1] # track score of this team. use -1 since this will guarnatee we look at the last columnm, which was just apened

        team_list[itr] = player_data[1] # track each team entry
        itr+=1

    # with the filled out list, calcaute the adjusted scores
    if STARTERS_AND_SUBS:
            # sort from best to worst
            gk_line.sort(reverse=True)
            def_line.sort(reverse=True)
            mid_line.sort(reverse=True)
            att_line.sort(reverse=True)

            # apply the penalty to subs (the last player, who here is the worst player)
            gk_score = gk_line[0] + (gk_line[1] * (SUB_PENALTY / 2)) # worth even less
            def_score = sum(def_line[0:4]) + (def_line[4] * SUB_PENALTY)
            mid_score = sum(mid_line[0:4]) + (mid_line[4] * SUB_PENALTY)
            att_score = sum(att_line[0:2]) + (att_line[2] * SUB_PENALTY)

            # make the total
            chromosome_score = gk_score + def_score + mid_score + att_score

    # check constraints - currently a hard cutoff, apply gradual penalties
    penalty_multiplier = 1.0 # multipler value that will be lowered per each foul

    # cost less than 1000
    if chromosome_cost > BUDGET:
        #chromosome_score = 0

        cost_overage = chromosome_cost - BUDGET
        budget_penalty_pct = cost_overage * COST_PENALTY
        #penalty_multiplier -= budget_penalty_pct
        penalty_multiplier *= max(0, (1.0 - budget_penalty_pct))

    # roster does not include more than x players from a team
    max_single_team_count = np.max(np.bincount(team_list))
    if max_single_team_count > MAX_TEAM_COUNT:
        #chromosome_score = 0

        team_violations = max_single_team_count - MAX_TEAM_COUNT
        team_penalty_pct = team_violations * TEAM_OVER_PENALTY
        #penalty_multiplier -= team_penalty_pct
        penalty_multiplier *= max(0, (1.0 - team_penalty_pct))

    if penalty_multiplier <= 0.005:
        #penalty_multiplier = 0.0 # kill it
        penalty_multiplier = 0.005 # instead of killing it, give it a small chance to breed

    chromosome_score = chromosome_score * penalty_multiplier # if no fouls, score remains unchanged


    return chromosome_score, chromosome_cost


# 1. Population / 2. Encoding / 3. Fitness Function

population = []

for _ in range(POPULATION_COUNT): # create x random teams
    # 2 goalkeepers, 5 defenders, 5 midfileres, 3 foward to fill in a 442 formation
    # Issue 1: no differnce in the dataset between player roles and sides, and midfielder itself is a vague term
    # Isseu 2: admittetdly this an abstraction. we simply gave 1 subs for each role
    # Issue 3 (FIXED): howevers subs are valued the same as starters, in fact there is no disction.
    gk_list = np.random.choice(player_pool_gk[:, 0], 2 , replace=False)
    def_list = np.random.choice(player_pool_def[:, 0], 5 , replace=False)
    mid_list = np.random.choice(player_pool_mid[:, 0], 5 , replace=False)
    att_list = np.random.choice(player_pool_att[:, 0], 3 , replace=False)

    chromosome = [gk_list, def_list, mid_list, att_list] # value encoding with player id, which uses player_dict to find any needed info

    score, cost = fitness_function(chromosome, player_dict)

    chromosome_team = {
        "chromosome": chromosome,
        "fitness": score,
        "cost": cost
    }

    population.append(chromosome_team)

#print(f"generated {len(population)} teams")


# 4. Selection  / 5. Crossover / 6. Mutation / 7. Ending Condition

print(f"{'Gen 0':<9} Score: {population[0]['fitness']:.2f} | Cost: {population[0]['cost']:>5.1f}")

# trackers for graph output
history_best_fitness = []
history_avg_fitness = []
history_worst_fitness = []
history_best_cost = []
history_diversity = []
final_costs = [ind["cost"] for ind in population]
final_points = []

# end condtiona trackers
best_fitness = -1
stagnation_count = 0
end_threshold = max(INTERVAL_TO_IMPROVE, int(GENERATIONS * 0.10)) # stop at either x gen or 10% of total gen, whichever is bigger
end_early = False
end_value = -1


position_pool = [player_pool_gk, player_pool_def, player_pool_mid, player_pool_att]
position_count = [2, 5, 5, 3]

for i in range(GENERATIONS):
    # selection
    # proportional method
    if PROPORTIONAL_METHOD == True:
        parent_pool = []
        total_fitness = sum(ind["fitness"] for ind in population)

        if total_fitness == 0: # edge case - unlikely
            for _ in range(POPULATION_COUNT):
                parent_pool.append(random.choice(population))
        else:
            for _ in range(POPULATION_COUNT): # make x amounnt of entries in parent_pool
                # apply fitness poprortioanl method
                pick = random.uniform(0, total_fitness)
                current = 0
                for x in population: # poportional random choose
                    current += x["fitness"]
                    if current >= pick:
                        parent_pool.append(x)
                        break
    # tournament method
    else:
        for _ in range(POPULATION_COUNT):
            # pick x random teams, the best wins
            competitors = random.sample(population, TOURNAMENT_SIZE)
            winner = max(competitors, key=lambda ind: ind["fitness"])
            parent_pool.append(winner)

    # crossover
    new_population = []
    for j in range(0, POPULATION_COUNT, 2): # extract the xth chromosome (array of players)

        # this holds the array of players [[gk], [def], etc...]
        # so parent1[0][0] would be the first goalkeeper
        # gk_data = player_dict[int(first_gk_id)] to search the data of goalkeeper
        # gk_data[4] following the index would give points
        parent1 = parent_pool[j]["chromosome"]
        parent2 = parent_pool[j+1]["chromosome"]

        child1 = [] # need to repopulate with same number of parents
        child2 = []

        if random.random() < CROSSOVER_RATE: # are we doing crossover?
            for k in range(4): # create child
                p1_position = list(parent1[k])
                p2_position = list(parent2[k])

                combined_parent = list(set(p1_position + p2_position)) # remove duplicates

                # random crossover, rather then splitting along a line, since order doenst matter
                child1.append( np.random.choice(combined_parent, position_count[k], replace=False) )
                child2.append( np.random.choice(combined_parent, position_count[k], replace=False) )
        else:
            child1 = [np.array(pos).copy() for pos in parent1]
            child2 = [np.array(pos).copy() for pos in parent2]

        # mutation
        for child_combined in [child1, child2]: # consider both childs
            if random.random() < MUTATION_RATE: # are we mutating?
                random_position = random.randint(0, 3) # pick a random position
                random_index = random.randint(0, len(child_combined[random_position]) - 1) # pick a random valid index for the postiion
                random_player = np.random.choice(position_pool[random_position][:, 0]) # pick a random player for the position

                while random_player in child_combined[random_position]: # check if the random player is already in the child
                    random_player = np.random.choice(position_pool[random_position][:, 0]) # if so, keep generating a new random player
                child_combined[random_position][random_index] = random_player # add the new unique random player

        # update the new population
        child1_score, child1_cost = fitness_function(child1, player_dict)
        child1_chromosome = {
            "chromosome": child1,
            "fitness": child1_score,
            "cost": child1_cost
        }
        new_population.append(child1_chromosome)

        child2_score, child2_cost = fitness_function(child2, player_dict)
        child2_chromosome = {
            "chromosome": child2,
            "fitness": child2_score,
            "cost": child2_cost
        }
        new_population.append(child2_chromosome)

    if ELITSM == True:
        # elitism method - replace worst child with single best parent
        best_parent = max(population, key=lambda ind: ind["fitness"]) # find the best parent in this generation
        new_population.sort(key=lambda ind: ind["fitness"]) # sort child by worst
        new_population[0] = best_parent # replace the worst child with a copy of the best parent

    population = new_population # begin the new generaiton

    # print out results every 10 genrations
    if (i + 1) % 10 == 0:
        best_squad = max(population, key=lambda ind: ind["fitness"])

        # output raw points as a universal comparison
        # note, we are not optimzing for raw points.
        # since we are using this database in lieu of offical stats,
        # points and cost are balanced around an artifical economy for fantasy draft.
        # instead, aim to look at how this might apply to the real world,
        # so efficeicny (points  per minute) and expected impact (ict) matter too
        raw_points = 0
        raw_ict = 0
        for sub_array in best_squad["chromosome"]:
            for player_id in sub_array:
                raw_points += player_dict[int(player_id)][4]
                raw_ict += player_dict[int(player_id)][6]

        print(f"Gen {i + 1:<5} Score: {best_squad['fitness']:.2f} | Cost: {best_squad['cost']:>6.1f} | Points: {raw_points:>6} | ICT: {raw_ict:>6.1f}")


    # check end condition - turned off for now
    #new_population.sort(key=lambda ind: ind["fitness"], reverse=True) # run it aagin in case ELITSIM is off
    #if new_population[0]["fitness"] > best_fitness:
    #    best_fitness = new_population[0]["fitness"]
    #    stagnation_count = 0 # best found, reset
    #else:
    #    stagnation_count += 1 # count up

    #if stagnation_count >= end_threshold:
    #    print(f"Convergence reached at Generation {i}.")
    #    print(f"The AI could not find a better squad for {end_threshold} generations.")
    #    end_early = True
    #    end_value = i
    #    break


    # graph stuff
    best_ind = max(population, key=lambda ind: ind["fitness"])
    avg_fitness = sum(ind["fitness"] for ind in population) / len(population)
    worst_ind = min(population, key=lambda ind: ind["fitness"])

    history_best_fitness.append(best_ind["fitness"])
    history_avg_fitness.append(avg_fitness)
    history_worst_fitness.append(worst_ind["fitness"])

    history_best_cost.append(best_ind["cost"])
    unique_teams = set(tuple(sorted(map(int, np.concatenate(ind["chromosome"])))) for ind in population)
    history_diversity.append(len(unique_teams))


# more graph stuff
final_costs = [ind["cost"] for ind in population]
final_points = []
for ind in population:
    pts = sum(player_dict[int(p_id)][4] for sub in ind["chromosome"] for p_id in sub)
    final_points.append(pts)

position_costs = {0: [], 1: [], 2: [], 3: []} # GK, DEF, MID, ATT
for ind in population:
    for pos_idx in range(4):
        pos_cost = sum(player_dict[int(p_id)][3] for p_id in ind["chromosome"][pos_idx])
        position_costs[pos_idx].append(pos_cost)


# Final Results 1: See the squad

def print_squad(individual):
    position_labels = ["Goalkeepers", "Defenders", "Midfielders", "Forwards"]
    chromosome = individual["chromosome"]

    total_raw_points = 0
    total_raw_ict = 0
    picked_percent = 0

    # loop through the 4 positional sub arrays
    for i in range(4):
        print(f"\n{position_labels[i]}:")

        for player_id in chromosome[i]:
            pid_int = int(player_id)
            player_data = player_dict[pid_int]

            player_points = player_data[4]
            total_raw_points += player_points

            player_ict = player_data[6]
            total_raw_ict += player_ict

            player_cost = player_data[3]
            picked_percent += player_data[7]

            local_heuristic_rank = player_rank_lookup.get(pid_int, "N/A")

            name_with_rank = f"{id_name[pid_int]} (#{local_heuristic_rank})"

            print(f" - {name_with_rank:<36} | Cost: {player_cost:>6.1f} | Pts: {player_points:>6} | ICT: {player_ict:>6.1f}")
            #print(f" - {id_name[pid_int]}")

    picked_percent = picked_percent / 15
    print("==============================")
    print(f"Team cost:     {individual['cost']} / {BUDGET}")
    print(f"Heuristic Fit: {individual['fitness']:.2f}")
    print(f"Total Points:  {total_raw_points}")
    print(f"Total ICT:  {total_raw_ict:.2f}")
    print(f"Avgerage Pick:  {picked_percent:.2f}%")
    print("==============================")


# Sort the final population from highest fitness to lowest
sorted_population = sorted(population, key=lambda ind: ind["fitness"], reverse=True)
# check if the best team is a legitamite team under the constraints set by the hyperparameters
for rank, individual in enumerate(sorted_population):
    chromosome = individual["chromosome"]

    total_cost = 0
    team_list = np.empty(15, dtype=int)
    itr = 0

    flat_squad = np.concatenate(chromosome)
    for player_id in flat_squad:
        player_data = player_dict[int(player_id)]

        total_cost += player_data[3]

        team_list[itr] = player_data[1]
        itr += 1

    max_single_team_count = np.max(np.bincount(team_list))

    is_cost_valid = total_cost <= BUDGET
    is_team_valid = max_single_team_count <= MAX_TEAM_COUNT

    if is_cost_valid and is_team_valid:
        print(f"Best Valid Squad (Ranked {rank + 1} in the final generation)")
        print_squad(individual)
        valid_squad_found = True
        break
    else:
        print(f"Skipping Rank {rank + 1} | Legal Cost: {is_cost_valid} | Legal Teams: {is_team_valid} \n")


# Fine most common surviving players
all_final_players = []
for ind in population:
    all_final_players.extend(np.concatenate(ind["chromosome"]))
player_counts = Counter(all_final_players)

print("\n==============================")
print("MOST COMMON SURVIVING PLAYERS")
print("==============================")
top_x = player_counts.most_common(5)
for rank, (p_id, count) in enumerate(top_x):
    name = id_name[int(p_id)]
    percentage = (count / POPULATION_COUNT) * 100
    print(f"{rank + 1}. {name:<34} | In {percentage:.1f}% of final squads")


# Final Results 2: See the results over time
plt.figure(figsize=(12, 10))

# Fitness Progression
plt.figure(figsize=(10, 6))
plt.plot(history_best_fitness, label="Best Fitness", color="blue", linewidth=2)
plt.plot(history_avg_fitness, label="Average Fitness", color="orange", linestyle="--")
plt.plot(history_worst_fitness, label="Worst Fitness", color="red", linestyle=":", alpha=0.7)
# Create a trendline for the average Fitness
if end_early == True:
    z = np.polyfit(range(end_value), history_avg_fitness, 1)
    p = np.poly1d(z)
    plt.plot(range(end_value), p(range(end_value)), color="black", linestyle="-.", label="Avg Trendline")
else:
    z = np.polyfit(range(GENERATIONS), history_avg_fitness, 1)
    p = np.poly1d(z)
    plt.plot(range(GENERATIONS), p(range(GENERATIONS)), color="black", linestyle="-.", label="Avg Trendline")
plt.title("Genetic Algorithm Convergence Stack")
plt.xlabel("Generation")
plt.ylabel("Heuristic Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Cost Management
plt.subplot(2, 1, 2)
plt.plot(history_best_cost, label="Cost of Best Squad", color="green", linewidth=2)
plt.axhline(y=BUDGET, color='red', linestyle=':', label=f"Budget Constraint ({BUDGET})")
plt.title("Constraint Learning: Squad Cost over Generations")
plt.xlabel("Generation")
plt.ylabel("Cost (Units)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Team Diversity
plt.figure(figsize=(10, 4))
plt.plot(history_diversity, color="purple", linewidth=2)
plt.title("Population Genetic Diversity Over Time")
plt.xlabel("Generation")
plt.ylabel(f"Number of Unique Squads (Max {POPULATION_COUNT})")
plt.grid(True, alpha=0.3)
plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(final_costs, final_points, alpha=0.6, c=final_points, cmap='viridis', label="Squads")
plt.axvline(x=BUDGET, color='red', linestyle='--', linewidth=2, label='Budget Ceiling')

plt.title("Final Generation Squad Distribution vs. Constraints")
plt.xlabel("Total Squad Cost")
plt.ylabel("Total Raw Points")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Postion allocation
labels = ["Goalkeepers", "Defenders", "Midfielders", "Forwards"]
data_to_plot = [position_costs[0], position_costs[1], position_costs[2], position_costs[3]]

plt.figure(figsize=(10, 5))
plt.boxplot(data_to_plot, labels=labels, patch_artist=True, boxprops=dict(facecolor='lightblue', color='blue'))
plt.title("Budget Distribution across Positions in the Final Generation")
plt.ylabel("Total Position Cost (Units)")
plt.grid(True, alpha=0.3)
plt.show()


def fitness_function_NO_PENALTY(chromosome, player_dict):
    chromosome_cost = 0

    gk_line = []
    def_line = []
    mid_line = []
    att_line = []

    for pos_index, pos_list in enumerate(chromosome):
        for player_id in pos_list:
            player_data = player_dict[int(player_id)]
            chromosome_cost += player_data[3] # track cost
            score = player_data[-1] # heuristic score

            if pos_index == 0:
                gk_line.append(score)
            elif pos_index == 1:
                def_line.append(score)
            elif pos_index == 2:
                mid_line.append(score)
            elif pos_index == 3:
                att_line.append(score)

    if STARTERS_AND_SUBS:
        gk_line.sort(reverse=True)
        def_line.sort(reverse=True)
        mid_line.sort(reverse=True)
        att_line.sort(reverse=True)

        gk_score = gk_line[0] + (gk_line[1] * (SUB_PENALTY / 2))
        def_score = sum(def_line[0:4]) + (def_line[4] * SUB_PENALTY)
        mid_score = sum(mid_line[0:4]) + (mid_line[4] * SUB_PENALTY)
        att_score = sum(att_line[0:2]) + (att_line[2] * SUB_PENALTY)

        chromosome_score = gk_score + def_score + mid_score + att_score
    else:
        chromosome_score = sum(gk_line) + sum(def_line) + sum(mid_line) + sum(att_line)

    return chromosome_score, chromosome_cost


best_team = True
if best_team == True: # last plauyer, commend out, is cheap sub option to mimic a more realstic team
    my_gks = [
        "David Raya Martín",
        "Caoimhín Kelleher"
        #"Mads Hermansen"
    ]

    my_defs = [
        "Gabriel dos Santos Magalhães",
        "Marc Guéhi",
        "Marcos Senesi Barón",
        "Virgil van Dijk",
        "James Tarkowski"
        #"Patrick Dorgu"
    ]

    my_mids = [
        "Bruno Borges Fernandes",
        "Antoine Semenyo",
        "Morgan Gibbs-White",
        "Declan Rice",
        "Elliot Anderson"
        #"Tyler Adams"
    ]
    my_atts = [
        "Erling Haaland",
        "Jarrod Bowen",
        "Igor Thiago Nascimento Rodrigues"
        #"Junior Kroupi"
    ]
else:

    my_gks = [
        "Gianluigi Donnarumma",
        "David Raya Martín"
    ]

    my_defs = [
        "Marc Cucurella Saseta",
        "Gabriel dos Santos Magalhães",
        "William Saliba",
        "Reece James",
        "Virgil van Dijk"
    ]

    my_mids = [
        "Mohamed Salah",
        "Bukayo Saka",
        "Bruno Borges Fernandes",
        "Cole Palmer",
        "Declan Rice"
    ]

    my_atts = [
        "Erling Haaland",
        "Igor Thiago Nascimento Rodrigues",
        "Antoine Semenyo"
    ]

# 2. Reverse lookup to find their IDs
name_to_id = {v: k for k, v in id_name.items()}

try:
    human_chromosome = [
        [name_to_id[name] for name in my_gks],
        [name_to_id[name] for name in my_defs],
        [name_to_id[name] for name in my_mids],
        [name_to_id[name] for name in my_atts]
    ]

    print("The 'Best' Squad")

    position_labels = ["Goalkeepers", "Defenders", "Midfielders", "Forwards"]
    total_raw_points = 0
    total_raw_ict = 0
    picked_percent = 0

    # 2.5 Loop through the 4 positional sub-arrays
    for i in range(4):
        print(f"\n{position_labels[i]}:")
        for player_id in human_chromosome[i]:
            pid_int = int(player_id)
            player_data = player_dict[pid_int]

            player_points = player_data[4]
            total_raw_points += player_points

            player_ict = player_data[6]
            total_raw_ict += player_ict

            player_cost = player_data[3]
            player_min = player_data[5]
            picked_percent += player_data[7]

            local_heuristic_rank = player_rank_lookup.get(pid_int, "N/A")

            name_with_rank = f"{id_name[pid_int]} (#{local_heuristic_rank})"

            print(f" - {name_with_rank:<36} | Cost: {player_cost:>6.1f} | Pts: {player_points:>6} | ICT: {player_ict:>6.1f} | Min: {player_min}")

    human_score, human_cost = fitness_function_NO_PENALTY(human_chromosome, player_dict)
    human_points = sum(player_dict[pid][4] for sublist in human_chromosome for pid in sublist)

    picked_percent = picked_percent / 15
    print("==============================")
    print(f"Team cost:     {human_cost} / {BUDGET}")
    print(f"Heuristic Fit: {human_score:.2f}")
    print(f"Total Points:  {human_points}")
    print(f"Total ICT:  {total_raw_ict:.2f}")
    print(f"Avgerage Pick:  {picked_percent:.2f}%")
    print("==============================\n")

except KeyError as e:
    print(f"Name Error: Could not find {e} in the dataset")
