
def dist(pred_of_query,pred_of_Dl,debug=False):
    # Norm Function
    norm = lambda a, b: np.sqrt(np.subtract(a, b) ** 2)
    distance = np.array([(norm(row, pred_of_Dl)) for row in pred_of_query])
    if debug: print(f"Distance Shape: {distance.shape}")
    return distance

def weight(distance,epsilon=0.1,debug=False):
    # Weight Function
    if debug: print(f"Epsilon: {epsilon}")
    dist = distance < epsilon
    weight = np.sum(dist, axis=2)
    if debug: print(f"Weight Shape: {weight.shape}")
    return weight
