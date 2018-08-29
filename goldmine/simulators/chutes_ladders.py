import autograd as ag
import autograd.numpy as np
from goldmine.simulators.base import Simulator


# def random_numbers():
#     rng = np.random.rand(6)
#     return (np.array(rng))
#
#
# rng = random_numbers()
#
#
# def make_weights(theta):
#     p_roll = []
#     p_i = 1 / 6 * rng ** (theta)
#     p_roll += [p_i]
#     p_roll = np.array(p_roll)
#     p_roll = p_roll / np.sum(p_roll)
#     return (np.array(p_roll)[0])

def make_weights(theta):
    p_roll = np.exp(theta * np.linspace(-2., 2., 6))
    p_roll /= np.sum(p_roll)

    return p_roll


CHUTES_LADDERS = {1: 38, 4: 14, 9: 31, 16: 6, 21: 42, 28: 84, 36: 44,
                  47: 26, 49: 11, 51: 67, 56: 53, 62: 19, 64: 60,
                  71: 91, 80: 100, 87: 24, 93: 73, 95: 75, 98: 78}  # dictionary of chutes and ladders


class ChutesLaddersSimulator(Simulator):
    n = 1

    ### initialization
    def __init__(self):
        self.d_simulate = ag.grad_and_aux(self.grad_simulate)

    ### creates samples x given theta (weights); p(x|theta)
    def simulate(self, theta=0, random_state=None):
        p_i = make_weights(theta)

        turns_list = []  # number of turns per game
        winner = []  # which player wins each game
        rolls_list = []
        log_list = []

        position1_list = []  # player 1 position list
        position2_list = []  # player 2 position list
        final_position = []

        counter = 0
        prob_side = []

        log_p_xz = 1.0
        roll_list = []
        position = []
        position1 = 0
        position2 = 0
        turns = 0
        counter += 1
        while position1 < 100 and position2 < 100:
            turns += 1
            random_number1 = np.random.rand()
            random_number2 = np.random.rand()
            ### player_1's turn
            if random_number1 <= p_i[0]:
                roll1 = 1
                log_p_xz *= p_i[0]
            elif random_number1 <= p_i[0] + p_i[1]:
                roll1 = 2
                log_p_xz *= p_i[1]
            elif random_number1 <= p_i[0] + p_i[1] + p_i[2]:
                roll1 = 3
                log_p_xz *= p_i[2]
            elif random_number1 <= p_i[0] + p_i[1] + p_i[2] + p_i[3]:
                roll1 = 4
                log_p_xz *= p_i[3]
            elif random_number1 <= p_i[0] + p_i[1] + p_i[2] + p_i[3] + p_i[4]:
                roll1 = 5
                log_p_xz *= p_i[4]
            else:
                roll1 = 6
                log_p_xz *= p_i[5]

            position1 += roll1
            position1 = CHUTES_LADDERS.get(position1, position1)

            ### player_2's turn
            if random_number2 <= p_i[0]:
                roll2 = 1
                log_p_xz *= p_i[0]
            elif random_number2 <= p_i[0] + p_i[1]:
                roll2 = 2
                log_p_xz *= p_i[1]
            elif random_number2 <= p_i[0] + p_i[1] + p_i[2]:
                roll2 = 3
                log_p_xz *= p_i[2]
            elif random_number2 <= p_i[0] + p_i[1] + p_i[2] + p_i[3]:
                roll2 = 4
                log_p_xz *= p_i[3]
            elif random_number2 <= p_i[0] + p_i[1] + p_i[2] + p_i[3] + p_i[4]:
                roll2 = 5
                log_p_xz *= p_i[4]
            else:
                roll2 = 6
                log_p_xz *= p_i[5]

            position2 += roll2
            position2 = CHUTES_LADDERS.get(position2, position2)

            ### logging positions/roll data
            position1_list += [position1]
            position2_list += [position2]
            position += [[position1, position2]]
            roll_list += [[roll1, roll2]]

            ### ends game
            if position1 >= 100:
                final_position += [position]
                winner += [0.0]
                turns_list += [turns]
                rolls_list += [roll_list]
                log_p_xz = (np.log(log_p_xz))
                prob_side.append(log_p_xz)

                position1_list += ["Player 1 Wins"]
                position2_list += ["Player 1 Wins"]
                continue

            elif position2 >= 100:
                final_position += [position]
                rolls_list += [roll_list]
                winner += [1.0]
                turns_list += [turns]
                log_p_xz = (np.log(log_p_xz))
                prob_side.append(log_p_xz)

                position1_list += ["Player 2 Wins"]
                position2_list += ["Player 2 Wins"]
                continue
            log_list.append(prob_side)
        return (log_p_xz, turns, rolls_list)

    def grad_simulate(self, theta=0, n=1, random_state=None):
        p_i = make_weights(theta)

        turns_list = []  # number of turns per game
        winner = []  # which player wins each game
        rolls_list = []
        log_list = []

        position1_list = []  # player 1 position list
        position2_list = []  # player 2 position list
        final_position = []

        counter = 0
        prob_side = []

        log_p_xz = 1.0
        roll_list = []
        position = []
        position1 = 0
        position2 = 0
        turns = 0
        counter += 1
        while position1 < 100 and position2 < 100:
            turns += 1
            random_number1 = np.random.rand()
            random_number2 = np.random.rand()
            ### player_1's turn
            if random_number1 <= p_i[0]:
                roll1 = 1
                log_p_xz *= p_i[0]
            elif random_number1 <= p_i[0] + p_i[1]:
                roll1 = 2
                log_p_xz *= p_i[1]
            elif random_number1 <= p_i[0] + p_i[1] + p_i[2]:
                roll1 = 3
                log_p_xz *= p_i[2]
            elif random_number1 <= p_i[0] + p_i[1] + p_i[2] + p_i[3]:
                roll1 = 4
                log_p_xz *= p_i[3]
            elif random_number1 <= p_i[0] + p_i[1] + p_i[2] + p_i[3] + p_i[4]:
                roll1 = 5
                log_p_xz *= p_i[4]
            else:
                roll1 = 6
                log_p_xz *= p_i[5]

            position1 += roll1
            position1 = CHUTES_LADDERS.get(position1, position1)

            ### player_2's turn
            if random_number2 <= p_i[0]:
                roll2 = 1
                log_p_xz *= p_i[0]
            elif random_number2 <= p_i[0] + p_i[1]:
                roll2 = 2
                log_p_xz *= p_i[1]
            elif random_number2 <= p_i[0] + p_i[1] + p_i[2]:
                roll2 = 3
                log_p_xz *= p_i[2]
            elif random_number2 <= p_i[0] + p_i[1] + p_i[2] + p_i[3]:
                roll2 = 4
                log_p_xz *= p_i[3]
            elif random_number2 <= p_i[0] + p_i[1] + p_i[2] + p_i[3] + p_i[4]:
                roll2 = 5
                log_p_xz *= p_i[4]
            else:
                roll2 = 6
                log_p_xz *= p_i[5]

            position2 += roll2
            position2 = CHUTES_LADDERS.get(position2, position2)

            ### logging positions/roll data
            position1_list += [position1]
            position2_list += [position2]
            position += [[position1, position2]]
            roll_list += [[roll1, roll2]]

            ### ends game
            if position1 >= 100:
                final_position += [position]
                winner += [0.0]
                turns_list += [turns]
                rolls_list += [roll_list]
                log_p_xz = (np.log(log_p_xz))
                prob_side.append(log_p_xz)

                position1_list += ["Player 1 Wins"]
                position2_list += ["Player 1 Wins"]
                continue

            elif position2 >= 100:
                final_position += [position]
                rolls_list += [roll_list]
                winner += [1.0]
                turns_list += [turns]
                log_p_xz = (np.log(log_p_xz))
                prob_side.append(log_p_xz)

                position1_list += ["Player 2 Wins"]
                position2_list += ["Player 2 Wins"]
                continue
            log_list.append(prob_side)
        return (log_p_xz, turns)

    def get_theta_n(self, n_theta):  # used to get theta values for rvs_ratio, if theta is not specified
        all_theta = []
        for i in range(n_theta):
            x = np.random.dirichlet(np.ones(6), size=1)
            all_theta += [x.tolist()[0]]
        return (all_theta)

    def rvs(self, theta, n, random_state=None):
        turns_list = []
        log_list = []
        for i in range(n):
            log_p_xz, turns, rolls_list = self.simulate(theta)
            turns_list.append(turns)
            log_list.append(log_p_xz)
        log_list = np.array(log_list)
        return (turns_list)

    def rvs_score(self, theta, n, random_state=None):
        all_x = []
        all_t_xz = []

        for i in range(n):
            _, x = self.grad_simulate(theta)
            t_xz, _ = self.d_simulate(theta)

            all_x.append(x)
            all_t_xz.append(t_xz)

        all_x = np.array(all_x)
        all_t_xz = np.array(all_t_xz)

        return (all_x, all_t_xz)

    def rvs_ratio(self, theta, theta0, theta1, n=1, random_state=None):
        all_x = []
        all_r_xz = []

        for i in range(n):
            log_p0_xz, turns_0, rolls_0 = self.simulate(theta)
            total_rolls = [roll for pair in rolls_0[0] for roll in pair]

            weight0 = make_weights(theta0)
            weight1 = make_weights(theta1)

            log_r_xz = 0.
            for roll in total_rolls:
                log_r_xz += np.log(weight0[roll - 1] / weight1[roll - 1])

            all_x.append(turns_0)
            all_r_xz.append(np.exp(log_r_xz))

        all_x = np.array(all_x)
        all_r_xz = np.array(all_r_xz)

        return (all_x, all_r_xz)

    def rvs_ratio_score(self, theta0, theta1, n, random_state=None):
        ratio, _ = self.rvs_ratio(theta0=theta0, theta1=theta1)
        t_xz, turns = self.rvs_score(theta=theta0, n=n)

        turns = np.array(turns)
        ratio = np.array(ratio)
        t_xz = np.array(t_xz)

        return (turns, ratio, t_xz)
