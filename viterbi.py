def viterbi(obs, states, start_probability, transition_probability, emission_probability):
    path = {s: [] for s in states}
    current_state_probs = {}
    for s in states:
        current_state_probs[s] = start_probability[s] * emission_probability[s][obs[0]]

    for i in range(1, len(obs)):
        previous_state_probs = current_state_probs
        current_state_probs = {}

        for curr_state in states:
            max_prob, max_state = max(
                (previous_state_probs[last_state] * transition_probability[last_state][curr_state] *
                 emission_probability[curr_state][obs[i]], last_state) for last_state in states
            )

            current_state_probs[curr_state] = max_prob
            path[curr_state].append(max_state)

    max_prob = -1
    max_prob_path = None
    for s in states:
        path[s].append(s)
        if current_state_probs[s] > max_prob:
            max_prob = current_state_probs[s]
            max_prob_path = path[s]

    return max_prob_path, max_prob


def main():
    obs = ['a', 'bat']
    states = ('DET', 'NOUN')

    # observations = ('normal', 'cold', 'dizzy')

    start_probability = {'DET': 0.6, 'NOUN': 0.4}

    transition_probability = {
        'DET': {'NOUN': 0.7, 'DET': 0.3},
        'NOUN': {'NOUN': 0.4, 'DET': 0.6},
    }

    emission_probability = {
        'Healthy': {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
        'Fever': {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
    }
    emission_probability = {
        'DET': {'a': 0.9, 'bat': 0.1},
        'NOUN': {'a': 0.1, 'bat': 0.9}
    }
    print(f'input sentence: {" ".join(obs)}')
    print(viterbi(obs, states, start_probability, transition_probability, emission_probability))


if __name__ == '__main__':
    main()
