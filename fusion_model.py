def build_markov(data, order=2, alpha=1.0):
    transitions = {}
    sequences = [list(x) for x in data]

    for seq in sequences:
        for i in range(len(seq) - order):
            key = tuple(seq[i:i+order])
            next_digit = seq[i+order]
            if key not in transitions:
                transitions[key] = {}
            transitions[key][next_digit] = transitions[key].get(next_digit, 0) + 1

    for k in transitions:
        total = sum(transitions[k].values()) + 10 * alpha
        for d in map(str, range(10)):
            transitions[k][d] = (transitions[k].get(d, 0) + alpha) / total

    return transitions


def markov_predict(data, order=2, top_k=5, alpha=1.0):
    if not data or len(data) < order + 1:
        return []

    transitions = build_markov(data, order=order, alpha=alpha)
    last = list(data[-1])
    state = tuple(last[-order:])
    preds = []

    for _ in range(top_k * 2):
        seq = last[-order:]
        for _ in range(2):
            next_probs = transitions.get(state)
            if not next_probs:
                break
            next_digit = max(next_probs, key=next_probs.get)
            seq.append(next_digit)
            state = tuple(seq[-order:])
        preds.append("".join(seq[-4:]))

    return list(dict.fromkeys(preds))[:top_k]
