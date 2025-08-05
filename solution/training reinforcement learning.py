import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools
import json
import datetime
from tensorflow.keras.callbacks import TensorBoard

# ==========================
# ITEM CLASS
# ==========================
class Item:
    def __init__(self, dx, dy, dz, weight, name=None, can_rotate=True):
        self.dx, self.dy, self.dz = dx, dy, dz
        self.weight = weight
        self.volume = dx * dy * dz
        self.name = name or f"Item_{dx}x{dy}x{dz}_{weight}kg"
        self.can_rotate = can_rotate

    def get_all_orientations(self):
        return list(set(itertools.permutations((self.dx, self.dy, self.dz)))) if self.can_rotate else [(self.dx, self.dy, self.dz)]

# ==========================
# ENVIRONMENT (TRAINING - SINGLE CONTAINER)
# ==========================
class ContainerPackingEnv:
    def __init__(self, container_size=(243, 606, 259), items=[]):
        self.width, self.depth, self.height = container_size
        self.initial_items = sorted(items, key=lambda x: (-x.volume, -x.weight))
        self._build_action_map()
        self.reset()

    def _build_action_map(self):
        self.action_map = {0: 'new_container'}
        self.reverse_action_map = {'new_container': 0}
        idx = 1
        for fs in range(50):
            for o in range(6):
                self.action_map[idx] = (fs, o)
                self.reverse_action_map[(fs, o)] = idx
                idx += 1
        self.total_actions = idx

    def reset(self):
        self.container = self._new_container()
        self.unplaced = self.initial_items.copy()
        return self._get_state()

    def _new_container(self):
        return {'free_spaces': [(0, 0, 0)], 'placed_items': [], 'volume_used': 0, 
                'grid': np.zeros((self.width, self.depth, self.height), dtype=np.int16)}

    def _get_state(self):
        # Basic info
        state = [
            len(self.unplaced),
            self.container['volume_used'] / (self.width * self.depth * self.height)
        ]
        
        # Weight distribution
        left = sum(i.weight for i, (x,_,_), _ in self.container['placed_items'] if x < self.width/2)
        right = sum(i.weight for i, (x,_,_), _ in self.container['placed_items'] if x >= self.width/2)
        front = sum(i.weight for i, (_,y,_), _ in self.container['placed_items'] if y < self.depth/2)
        rear = sum(i.weight for i, (_,y,_), _ in self.container['placed_items'] if y >= self.depth/2)
        
        state.extend([left, right, front, rear])
        
        # Add current item info
        if self.unplaced:
            next_item = self.unplaced[0]
            state.extend([next_item.dx, next_item.dy, next_item.dz, next_item.weight])
        else:
            state.extend([0, 0, 0, 0])
        
        return np.array(state)

    def _valid(self, item, pos, orient):
        x, y, z = pos
        dx, dy, dz = orient
        
        # 1. Check physical boundaries
        if x + dx > self.width or y + dy > self.depth or z + dz > self.height:
            return False
        
        # 2. Check overlap
        if np.any(self.container['grid'][x:x+dx, y:y+dy, z:z+dz]):
            return False
        
        return True

    def _update_spaces(self, pos, dims):
        x, y, z = pos
        dx, dy, dz = dims
        self.container['free_spaces'] = [fs for fs in self.container['free_spaces'] 
                                      if not (x <= fs[0] < x+dx and y <= fs[1] < y+dy and z <= fs[2] < z+dz)]
        for s in [(x+dx, y, z), (x, y+dy, z), (x, y, z+dz)]:
            if all(i < lim for i, lim in zip(s, (self.width, self.depth, self.height))):
                if self.container['grid'][s] == 0 and s not in self.container['free_spaces']:
                    self.container['free_spaces'].append(s)

    def _valid_actions(self):
        if not self.unplaced:
            return []
        actions = []
        item = self.unplaced[0]
        for fs_idx, pos in enumerate(self.container['free_spaces'][:50]):
            for o_idx, orient in enumerate(item.get_all_orientations()[:6]):
                if self._valid(item, pos, orient):
                    actions.append(self.reverse_action_map[(fs_idx, o_idx)])
        return actions

    def step(self, action):
        reward = 0
        done = False
        key = self.action_map[action]

        if key == 'new_container':
            return self._get_state(), -10, True, {}

        if not self.unplaced:
            return self._get_state(), 0, True, {}

        item = self.unplaced.pop(0)
        fs_idx, o_idx = key
        pos = self.container['free_spaces'][fs_idx]
        orient = item.get_all_orientations()[o_idx]

        if not self._valid(item, pos, orient):
            self.unplaced.insert(0, item)
            return self._get_state(), -5, False, {}

        x, y, z = pos
        dx, dy, dz = orient
        
        # Calculate support factor
        support_area = 0
        if z > 0:  # Not on floor
            for i in range(x, x+dx):
                for j in range(y, y+dy):
                    if self.container['grid'][i,j,z-1] == 1:
                        support_area += 1
            support_factor = support_area / (dx*dy)
        else:
            support_factor = 1.0

        # Update container state
        self.container['grid'][x:x+dx, y:y+dy, z:z+dz] = 1
        self.container['placed_items'].append((item, pos, orient))
        self.container['volume_used'] += item.volume
        self._update_spaces(pos, orient)

        # Calculate rewards
        util = self.container['volume_used'] / (self.width * self.depth * self.height)
        
        # Base rewards
        reward = item.volume * 0.1 + 50 * util
        
        # Support rewards
        if z > 0:
            reward += 10 * support_factor
            if support_factor < 0.3:
                reward -= 5
        
        # Balance rewards
        left_weight = sum(i.weight for i, (px,_,_), _ in self.container['placed_items'] if px < self.width/2)
        right_weight = sum(i.weight for i, (px,_,_), _ in self.container['placed_items'] if px >= self.width/2)
        front_weight = sum(i.weight for i, (_,py,_), _ in self.container['placed_items'] if py < self.depth/2)
        rear_weight = sum(i.weight for i, (_,py,_), _ in self.container['placed_items'] if py >= self.depth/2)
        
        total_weight = max(1, left_weight + right_weight)
        imbalance_lr = abs(left_weight - right_weight) / total_weight
        imbalance_fb = abs(front_weight - rear_weight) / total_weight
        
        reward += 10 * (1 - imbalance_lr) + 10 * (1 - imbalance_fb)
        
        # Completion bonus
        if not self.unplaced:
            reward += 100 + (200 if util > 0.85 else 0)
            done = True
        
        return self._get_state(), reward, done, {}

# ==========================
# AGENT
# ==========================
class PackingAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.model = self._build_model()
        self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = TensorBoard(log_dir=self.log_dir, histogram_freq=1)

    def _build_model(self):
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(256, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(self.action_size)
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001), 
                     loss=Huber())
        return model

    def act(self, state, valid_actions):
        if not valid_actions:
            return random.choice(range(self.action_size))
            
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        return max(valid_actions, key=lambda a: q_values[a])

    def remember(self, s, a, r, s_, done):
        self.memory.append((s, a, r, s_, done))

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([x[0] for x in minibatch])
        actions = np.array([x[1] for x in minibatch])
        rewards = np.array([x[2] for x in minibatch])
        next_states = np.array([x[3] for x in minibatch])
        dones = np.array([x[4] for x in minibatch])
        
        targets = self.model.predict(states, verbose=0)
        next_q = self.model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        
        self.model.fit(states, targets, batch_size=batch_size, 
                      verbose=0, callbacks=[self.tensorboard_callback])

    def save(self, filename):
        self.model.save(filename)

    @staticmethod
    def load(filename, state_size, action_size):
        agent = PackingAgent(state_size, action_size)
        agent.model = load_model(filename)
        agent.epsilon = agent.epsilon_min
        return agent

# ==========================
# TRAINING FUNCTION
# ==========================
def train_agent(env, episodes=1500):
    initial_state = env.reset()
    agent = PackingAgent(state_size=initial_state.shape[0], action_size=env.total_actions)
    best_reward = -float('inf')
    
    for e in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            valid_actions = env._valid_actions() or [0]  # Default to new_container
            
            action = agent.act(state, valid_actions)
            next_state, reward, done, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            if len(agent.memory) >= 64 and len(agent.memory) % 10 == 0:
                agent.replay(batch_size=64)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save("best_model.keras")
        
        print(f"Episode {e+1}, Reward: {episode_reward:.1f}, Epsilon: {agent.epsilon:.3f}")
        
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
    
    return agent


# ==========================
# TESTING AND VISUALIZATION
# ==========================
def pack_items_into_containers(agent, items, container_size, max_containers):
    containers = []
    remaining_items = items.copy()
    
    while len(containers) < max_containers and remaining_items:
        env = ContainerPackingEnv(container_size=container_size, items=remaining_items)
        state = env.reset()
        done = False
        
        while not done and remaining_items:
            valid_actions = env._valid_actions()
            if not valid_actions:
                break
            action = agent.act(state, valid_actions)
            state, _, done, _ = env.step(action)
        
        packed_container = env.container
        containers.append(packed_container)
        packed_items = [item for item, _, _ in packed_container['placed_items']]
        remaining_items = [item for item in remaining_items if item not in packed_items]
    
    return containers

def visualize_all_containers(containers, container_size):
    for idx, container in enumerate(containers):
        # Calculate metrics
        total_weight = sum(item.weight for item, _, _ in container['placed_items'])
        cog_x = sum((x + dx/2) * item.weight for item, (x,y,z), (dx,dy,dz) in container['placed_items']) / total_weight
        cog_y = sum((y + dy/2) * item.weight for item, (x,y,z), (dx,dy,dz) in container['placed_items']) / total_weight
        
        left = sum(item.weight for item, (x,_,_), _ in container['placed_items'] if x < container_size[0]/2)
        right = sum(item.weight for item, (x,_,_), _ in container['placed_items'] if x >= container_size[0]/2)
        front = sum(item.weight for item, (_,y,_), _ in container['placed_items'] if y < container_size[1]/2)
        rear = sum(item.weight for item, (_,y,_), _ in container['placed_items'] if y >= container_size[1]/2)
        
        # Plot
        fig = plt.figure(figsize=(12, 5))
        
        ax1 = fig.add_subplot(121)
        bars1 = ax1.bar(['Left', 'Right'], [left, right], color=['blue', 'orange'])
        ax1.axhline((left+right)/2, color='red', linestyle='--', label='Balance Target')
        ax1.set_title('Left-Right Weight Distribution')
        ax1.legend()
        
        ax2 = fig.add_subplot(122)
        bars2 = ax2.bar(['Front', 'Rear'], [front, rear], color=['green', 'red'])
        ax2.axhline((front+rear)/2, color='red', linestyle='--', label='Balance Target')
        ax2.set_title('Front-Rear Weight Distribution')
        ax2.legend()
        
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.1f}kg', ha='center', va='bottom')
        
        plt.suptitle(f"Container {idx+1}\nCoG: ({cog_x:.1f}, {cog_y:.1f})")
        plt.tight_layout()
        plt.show()
# ==========================
# MAIN EXECUTION
# ==========================
if __name__ == "__main__":
    CONTAINER_SIZE = (243, 606, 259)
    ITEMS = [
        *[Item(15, 30, 20, 1, "HeavyBox1") for _ in range(200)],
        *[Item(150, 30, 30, 1, "HeavyBox2") for _ in range(100)],
        *[Item(40, 40, 100, 1, "SmallBox1") for _ in range(50)]
    ]
    MAX_CONTAINERS = 3
    EPISODES = 1500  # Reduced for testing

    print("=== TRAINING PHASE ===")
    env = ContainerPackingEnv(container_size=CONTAINER_SIZE, items=ITEMS)
    agent = train_agent(env, episodes=EPISODES)
    agent.save("packing_agent.keras")

    print("\n=== TESTING PHASE ===")
    containers = pack_items_into_containers(agent, ITEMS, CONTAINER_SIZE, MAX_CONTAINERS)
    
    total_packed = sum(len(c['placed_items']) for c in containers)
    print(f"\nResults:")
    print(f"- Containers used: {len(containers)}")
    print(f"- Items packed: {total_packed}/{len(ITEMS)}")
    
    visualize_all_containers(containers, CONTAINER_SIZE)