class HMM:
    def __init__(self, hidden_states, observable_states, steady_state_probabilities, transition_probabilities,
                 emission_probabilities):
        self.hidden_states = hidden_states
        self.observable_states = observable_states
        self.steady_state_probabilities = steady_state_probabilities
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities

    def viterbi(self, observed_sequence):
        path = {s: [] for s in self.hidden_states}
        curr_probs = {s: self.steady_state_probabilities[s] * self.emission_probabilities[s][observed_sequence[0]] for s
                      in self.hidden_states}
        for i in range(1, len(observed_sequence)):
            previous_state_probs = curr_probs
            curr_probs = {}

            for s in self.hidden_states:
                max_prob, max_state = max(
                    (previous_state_probs[last_state] * self.transition_probabilities[last_state][s] *
                     self.emission_probabilities[s][observed_sequence[i]], last_state) for last_state in
                    self.hidden_states
                )

                curr_probs[s] = max_prob
                path[s].append(max_state)

        max_path = None
        max_path_prob = -1

        for s in self.hidden_states:
            path[s].append(s)
            if max_path_prob < curr_probs[s]:
                max_path_prob = curr_probs[s]
                max_path = path[s]
        return max_path, max_path_prob

    def forward(self, observed_sequence):
        m = len(observed_sequence)
        n = len(self.hidden_states)
        dp = [[0 for _ in range(len(self.hidden_states))] for __ in range(len(observed_sequence))]

        for i, state in enumerate(self.hidden_states):
            dp[0][i] = self.steady_state_probabilities[state] * self.emission_probabilities[state][observed_sequence[0]]

        for t in range(1, m):
            for curr_idx, curr_state in enumerate(self.hidden_states):
                for last_idx, last_state in enumerate(self.hidden_states):
                    dp[t][curr_idx] += dp[t - 1][last_idx] * self.transition_probabilities[last_state][curr_state] * \
                                       self.emission_probabilities[curr_state][observed_sequence[curr_idx]]

        return sum(dp[-1], 0)


def main():
    obs = ['normal', 'cold', 'dizzy']
    states = ('Healthy', 'Fever')

    observations = ('normal', 'cold', 'dizzy')

    start_probability = {'Healthy': 0.6, 'Fever': 0.4}

    transition_probability = {
        'Healthy': {'Healthy': 0.7, 'Fever': 0.3},
        'Fever': {'Healthy': 0.4, 'Fever': 0.6},
    }

    emission_probability = {
        'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
        'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
    }

    hmm = HMM(states, obs, start_probability, transition_probability, emission_probability)
    print(hmm.viterbi(obs))
    print(hmm.forward(obs))


if __name__ == '__main__':
    main()
