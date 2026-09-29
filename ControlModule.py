# Import required dependencies
import numpy as np
import mdptoolbox

class ControlModule:
    def __init__(self):
        """ Dummy constructor to use the Python Class as a namespace """
        pass


    @staticmethod
    def generate_P(probs: np.ndarray, n_states: np.int32) -> np.ndarray:
        """
        This method generates the probabilities of the transition matrix
        This matrix will consist of 30,000 numbers that will determine the probability
        to go from one state to another
        """
        #We initialize the matrix with zeros
        #P: Probability matrix
        P = np.zeros((3, n_states, n_states), dtype=np.float64)

        for s in range(n_states):

            #DECREASE (d)
            #First we determine the special cases, when the state is o.
            if s == 0:
                P[0, s, 0] = 1.0
            #When the state is 1
            elif s == 1:
                P[0, s, 0] = probs[0, 0] + probs[0, 1]
                P[0, s, 1] = probs[0, 2]
            #Every other case
            else:
                P[0, s, s - 2] = probs[0, 0]
                P[0, s, s - 1] = probs[0, 1]
                P[0, s, s] = probs[0, 2]

            # MAINTAIN (m)
            #First we determine the special cases, when s is 0.
            if s == 0:
                P[1, s, 0] = probs[1, 0] + probs[1, 1]
                P[1, s, 1] = probs[1, 2]
            #In the maximum (in this case 99)
            elif s == n_states - 1:
                P[1, s, s - 1] = probs[1, 0]
                P[1, s, s] = probs[1, 1] + probs[1, 2]
            #General case
            else:
                P[1, s, s - 1] = probs[1, 0]
                P[1, s, s] = probs[1, 1]
                P[1, s, s + 1] = probs[1, 2]

            # INCREASE (i)
            # Special cases, when we are at the last level (In this case 99)
            if s == n_states - 1:
                P[2, s, s] = 1.0
            #Second largest (In this case 98)
            elif s == n_states - 2:
                P[2, s, s] = probs[2, 0]
                P[2, s, s + 1] = probs[2, 1] + probs[2, 2]
            #general case
            else:
                P[2, s, s] = probs[2, 0]
                P[2, s, s + 1] = probs[2, 1]
                P[2, s, s + 2] = probs[2, 2]

        return P

    @staticmethod
    def generate_R(demand: np.float64, n_states: np.int32, n_actions: np.int32) -> np.ndarray:
        """
        This method generates the cost matrix C for the current demand point d_t.

        The cost of transitioning to state s' is the absolute distance between
        the current demand and the power level of s'. If the action moves the
        reactor away from the demand (decrease going below, increase going above),
        the cost is doubled as a penalty.

        Returns C shaped (n_actions, n_states, n_states).
        """
        C = np.zeros((n_actions, n_states, n_states), dtype=np.float64)

        for a in range(n_actions):
            for s in range(n_states):
                for s_next in range(n_states):

                    # lower bound of the destination interval, e.g. state 33 → 0.33
                    level_next = s_next / n_states

                    # base cost: how far are we from what the grid needs right now
                    cost = abs(demand - level_next)

                    current_level = s /n_states

                    # double the cost if we are actively moving away from the target:
                    # decreasing when already below demand, or increasing when above it
                    if a == 0 and current_level < demand:
                        cost *= 2.0
                    elif a == 2 and current_level > demand:
                        cost *= 2.0

                    C[a, s, s_next] = -cost

        return C

    @staticmethod
    def control_iteration(demand: np.float64, current_state: np.int32, n_states: np.int32,
                          n_actions: np.int32, gamma: np.float64, P: np.ndarray = None, probs: np.ndarray = None) -> np.int32:
        """
        This method is the responsible to determine what is the best action to take
        considering the demand, the current state, the number of states and actions.
        In this specific implementation it should normally receive the matrix P.
        """

        """
        Since the guidelines said that we should calculate matrix P only once, when executing the code
        It is going to be calculated outside the loop in control loop,
        nevertheless since it says that control iteration should do it, in case it is not made,
        we generate it here.
        """
        if P is None:
            P = ControlModule.generate_P(probs, n_states)

        #Here matrix C is calculated for each iteration for a specific demand
        R = ControlModule.generate_R(demand, n_states, n_actions)


        vi = mdptoolbox.mdp.ValueIteration(P, R, gamma)

        # 3. Ejecutar y obtener acción
        vi.run()
        return vi.policy[current_state]

    @staticmethod
    def control_loop(demand: np.ndarray, probs: np.ndarray, n_states: np.int32,
                     n_actions: np.int32, gamma: np.float64) -> np.ndarray:

        history = []
        current_state = 0

        # --- LA CLAVE ESTÁ AQUÍ ---
        # Calculamos P UNA SOLA VEZ antes de empezar el bucle
        P_matrix = ControlModule.generate_P(probs, n_states)
        # --------------------------

        offsets = {0: [-2, -1, 0], 1: [-1, 0, 1], 2: [0, 1, 2]}

        for d_t in demand:
            # Ahora le pasamos la P_matrix ya calculada
            action = ControlModule.control_iteration(d_t, current_state, n_states, n_actions, gamma, P_matrix)

            # Resto del código igual...
            jump = np.random.choice(offsets[action], p=probs[action])
            current_state = int(np.clip(current_state + jump, 0, n_states - 1))
            history.append(current_state / float(n_states))

        return np.array(history, dtype=np.float64)