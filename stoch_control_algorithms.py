import numpy as np
from itertools import product

# Transition kernal
P = np.array([[[.1, .9],[.5, .5]], [[.8, .2],[.2, .8]]])
# States - indexed by zero becuase python
states = np.array([0,1])
# Actions - indexed by zero
actions = np.array([0,1])
beta = 0.8

def policy_iteration():
    print('Policy Iteration')
    # Policy initalization to be u(0)=0, u(1)=0
    policy = np.zeros(2, dtype=int)
    policy[1] = 0
    policy[0] = 1
    # Value initialization
    W = np.array([np.nan, np.nan])

    policy_changed = False

    # Iterat through policies while the policy is not changing
    while not policy_changed:
        # Get value of policy
        W = get_policy_value(policy)
        # Set policy changed to false
        policy_changed = False
        for s in states:
            # Initialize variable to check value of alternative actions for state
            s_value = np.zeros(2)
            for a in actions:
                s_value[a] = cost(s, a) + beta*sum([P[s, a, s_]*W[s_] for s_ in states])
            # Best action for current state
            a_ = np.argmin(s_value)
            # If best action for state less than policy value for state
            if s_value[a_] < W[s]:
                # Update policy for state to best action
                policy[s] = a_
                # Set policy changed flag
                policy_changed = True

    print('Policy obtained:\nWhen in state x_t = 0, u_t =', policy[0], 'and when x_t = 1, u_t =', policy[1])
    return policy

def value_iteration():
    print('\nValue Iteration')
    # Value for each state
    value = np.zeros(2)
    # Temp value array for comparison
    value_ = np.zeros(2)
    while True:
        for s in states:
            # Assign previous value array to current value
            value_[s] = value[s]
            # Compute minimum value for state and update value array
            value[s] = min([sum(P[s, a, s_]*(cost(s, a)+beta*value_[s_]) for s_ in states) for a in actions])
        # If the value is within convergence threshold
        if sum(abs(value-value_)) < .1:
            # Exit loop
            break
    # Obtain policy for the optimal aproximently optimal value array
    policy = np.zeros(2)
    for s in states:
        policy[s] = np.argmin([sum(P[s, a, s_]*(cost(s, a) + beta*value_[s_]) for s_ in states) for a in actions])

    print('policy', policy)
    # print('Policy obtained:\nWhen in state x_t = 0, u_t =', policy[0], 'and when x_t = 1, u_t =', policy[1])
    return policy

def q_learning():
    print('\nQ Learning')
    # Initialize matrix to be zeros for all state-action pairs
    Q = np.zeros((2,2))
    # Initialize starting state to be 0
    s = 0
    # Initialize first action to be 0
    a = 0
    # Count how many times each state-action pair has been entered
    s_a_count = np.zeros((2,2))
    # IDK how many iterations to do
    for _ in range(100):
        # Increment counter for current state action pair
        s_a_count[s, a] += 1
        # Randomly enter next state according to current state and action
        s_ = np.random.choice(a=states, p=P[s, a])
        # Update value for state action pair in Q table
        Q[s,a] = Q[s,a] + (1/(1+s_a_count[s,a]))*(cost(s, a) + beta*min(Q[s_]) - Q[s,a])
        # Choose next action randomly ~90 percent of the time
        if np.random.uniform() <.9:
            a = np.random.choice(actions)
        # Here choose the current best action for the current state
        else:
            a = np.argmin(Q[s_])
        # Update state to next state
        s = s_
    policy = np.argmin(Q, axis=1)
    print('Policy obtained:\nWhen in state x_t = 0, u_t =', policy[0], 'and when x_t = 1, u_t =', policy[1])
    return policy

def get_policy_value(policy):
    beta = 0.8
    # Identity matrix
    I = np.eye(2)
    # Transition matrix under current policy
    P_ = np.array([[P[0, policy[0], 0], P[1, policy[1], 0]],
                   [P[0, policy[0], 1], P[1, policy[1], 1]]])
    # Cost vector
    C = np.array([[cost(0, policy[0])],[cost(1, policy[1])]])

    # Solve for W=((I-beta*P_)^-1) * C
    W = np.matmul(np.linalg.inv(I-beta*P_),C)
    return W


def cost(x, u):
    c = 0
    if x and u == 1:
        c = -1
    c += .5*(u)
    return c

if __name__ == "__main__":
    policy = policy_iteration()
    policy = value_iteration()
    policy = q_learning()
