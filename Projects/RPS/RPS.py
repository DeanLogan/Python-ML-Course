# The example function below keeps track of the opponent's opponent_historyory and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

counter_move = {"R": "P", "P": "S", "S": "R"}

steps = {}

def player(prev_play, opponent_history=[]):
    if prev_play != "":
        opponent_history.append(prev_play)
    n = 3

    guess = "R"
    if len(opponent_history) > n:
        pattern = join(opponent_history[-n:])

        if join(opponent_history[-(n + 1):]) in steps.keys():
            steps[join(opponent_history[-(n + 1):])] += 1
        else:
            steps[join(opponent_history[-(n + 1):])] = 1

        possible = [pattern + "R", pattern + "P", pattern + "S"]

        for i in possible:
            if not i in steps.keys():
                steps[i] = 0

        predict = max(possible, key=lambda key: steps[key])

        if predict[-1] == "P":
            guess = "S"
        if predict[-1] == "R":
            guess = "P"
        if predict[-1] == "S":
            guess = "R"
        print(steps)

    return guess


def join(moves):
    return "".join(moves)