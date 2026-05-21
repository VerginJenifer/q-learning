# Q Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
To develop a Python program that determines the optimal policy for the Frozen Lake environment using the Q-Learning algorithm. The program should train an agent through interactions with the environment to learn the optimal action-value function and derive the best policy for reaching the goal state. Additionally, the program should implement the Monte Carlo method to estimate the state-value function based on sampled episodes. Finally, a comparison should be made between the state values obtained from Q-Learning and Monte Carlo methods in terms of convergence, accuracy, and learning behavior, to understand the differences between these reinforcement learning approaches.

## Q LEARNING ALGORITHM
### Step 1:
Initialize Q-table and hyperparameters.

### Step 2 :
Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.

### Step 3 :
After training, derive the optimal policy from the Q-table.

### Step 4 :
Implement the Monte Carlo method to estimate state values.

### Step 5 :
Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.

## Q LEARNING FUNCTION
### Name: D Vergin Jenifer
### Register Number: 212223240174

```
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
      state,done = env.reset(), False
      while not done:
        action = select_action(state, Q, epsilons[e])
        next_state,reward,done,_=env.step(action)
        td_target = reward + gamma * Q[next_state].max() * (not done)
        td_error = td_target - Q[state][action]
        Q[state][action] = Q[state][action] + alphas[e] * td_error
        state = next_state
      Q_track[e] = Q
      pi_track.append(np.argmax(Q, axis=1))
    V=np.max(Q, axis=1)
    pi=lambda s: {s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    

    return Q, V, pi, Q_track, pi_track
```




## OUTPUT:
### Optimal value function
<img width="472" height="387" alt="image" src="https://github.com/user-attachments/assets/7e0b8107-be46-4044-8af6-c5da9c0f72b3" />
<img width="902" height="642" alt="image" src="https://github.com/user-attachments/assets/93d3ec63-2285-41b6-8b33-f00cfca4b280" />

### Optimal policy
<img width="607" height="111" alt="image" src="https://github.com/user-attachments/assets/d8165410-22ae-481b-b00d-d3a55c81b11a" />



### Plot comparing the state value functions of Monte Carlo method and Qlearning
<img width="1725" height="777" alt="image" src="https://github.com/user-attachments/assets/f947fec8-8552-4601-a1f0-00197553c4ea" />
<img width="1725" height="777" alt="image" src="https://github.com/user-attachments/assets/97775398-f82e-4e73-97cc-fb63987d8e14" />


## RESULT:

Write your result here
