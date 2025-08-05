import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import itertools
from tensorflow.keras.models import load_model

class Item:
    def __init__(self, dx, dy, dz, weight, name=None, can_rotate=True):
        self.dx, self.dy, self.dz = dx, dy, dz
        self.weight = weight
        self.volume = dx * dy * dz
        self.name = name or f"Item_{dx}x{dy}x{dz}_{weight}kg"
        self.can_rotate = can_rotate

    def get_all_orientations(self):
        if self.can_rotate:
            return list(set(itertools.permutations((self.dx, self.dy, self.dz))))
        return [(self.dx, self.dy, self.dz)]

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
        for fs in range(50):  # First 50 free spaces
            for o in range(6):  # First 6 orientations
                self.action_map[idx] = (fs, o)
                self.reverse_action_map[(fs, o)] = idx
                idx += 1
        self.total_actions = idx

    def reset(self):
        self.container = {
            'free_spaces': [(0, 0, 0)],
            'placed_items': [],
            'volume_used': 0,
            'grid': np.zeros((self.width, self.depth, self.height), dtype=np.int16)
        }
        self.unplaced = self.initial_items.copy()
        return self._get_state()

    def _get_state(self):
        """Returns the 10-feature state vector"""
        # Feature 1: Remaining items count
        remaining_items = len(self.unplaced)
        
        # Feature 2: Volume utilization
        utilization = self.container['volume_used'] / (self.width * self.depth * self.height)
        
        # Features 3-6: Weight distribution
        left = right = front = rear = 0.0
        for item, (x, y, _), _ in self.container['placed_items']:
            if x < self.width/2:
                left += item.weight
            else:
                right += item.weight
            if y < self.depth/2:
                front += item.weight
            else:
                rear += item.weight
        
        # Features 7-10: Next item dimensions and weight
        if self.unplaced:
            next_item = self.unplaced[0]
            dx, dy, dz, weight = next_item.dx, next_item.dy, next_item.dz, next_item.weight
        else:
            dx = dy = dz = weight = 0
        
        return np.array([
            remaining_items,      # Feature 1
            utilization,         # Feature 2
            left,                # Feature 3
            right,               # Feature 4
            front,               # Feature 5
            rear,                # Feature 6
            dx,                  # Feature 7
            dy,                  # Feature 8
            dz,                  # Feature 9
            weight               # Feature 10
        ], dtype=np.float32)

    def _valid(self, item, pos, orient):
        x, y, z = pos
        dx, dy, dz = orient
        if (x + dx > self.width or y + dy > self.depth or z + dz > self.height):
            return False
        return not np.any(self.container['grid'][x:x+dx, y:y+dy, z:z+dz])

    def _update_spaces(self, pos, dims):
        x, y, z = pos
        dx, dy, dz = dims
        
        # Remove covered spaces
        self.container['free_spaces'] = [
            fs for fs in self.container['free_spaces']
            if not (x <= fs[0] < x+dx and y <= fs[1] < y+dy and z <= fs[2] < z+dz)
        ]
        
        # Add new potential spaces
        for new_pos in [(x+dx, y, z), (x, y+dy, z), (x, y, z+dz)]:
            if all(0 <= p < lim for p, lim in zip(new_pos, (self.width, self.depth, self.height))):
                if self.container['grid'][new_pos] == 0 and new_pos not in self.container['free_spaces']:
                    self.container['free_spaces'].append(new_pos)

    def _valid_actions(self):
        if not self.unplaced:
            return []
            
        item = self.unplaced[0]
        valid_actions = []
        
        for fs_idx, pos in enumerate(self.container['free_spaces'][:50]):
            for o_idx, orient in enumerate(item.get_all_orientations()[:6]):
                if self._valid(item, pos, orient):
                    valid_actions.append(self.reverse_action_map[(fs_idx, o_idx)])
        
        return valid_actions or [0]  # Default to new_container if no valid actions

    def step(self, action):
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

        # Place the item
        x, y, z = pos
        dx, dy, dz = orient
        self.container['grid'][x:x+dx, y:y+dy, z:z+dz] = 1
        self.container['placed_items'].append((item, pos, orient))
        self.container['volume_used'] += item.volume
        self._update_spaces(pos, orient)

        return self._get_state(), 0, len(self.unplaced) == 0, {}

class PackingPredictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.model.summary()  # Show model architecture
    
    def predict_action(self, state, valid_actions):
        if not valid_actions:
            return 0  # Default to new container
        
        q_values = self.model.predict(state[np.newaxis], verbose=0)[0]
        return valid_actions[np.argmax([q_values[a] for a in valid_actions])]

def pack_items(model_path, items, container_size, max_containers=3):
    predictor = PackingPredictor(model_path)
    containers = []
    remaining_items = items.copy()
    
    while len(containers) < max_containers and remaining_items:
        env = ContainerPackingEnv(container_size, remaining_items)
        state = env.reset()
        done = False
        
        while not done:
            valid_actions = env._valid_actions()
            if not valid_actions:
                break
                
            action = predictor.predict_action(state, valid_actions)
            state, _, done, _ = env.step(action)
        
        containers.append(env.container)
        remaining_items = [item for item in remaining_items 
                         if not any(item is placed[0] for placed in env.container['placed_items'])]
    
    return containers, remaining_items
def visualize_packing(containers, container_size):
    # Create a single figure with subplots
    fig = plt.figure(figsize=(10 * len(containers), 5))
    
    for i, container in enumerate(containers):
        ax = fig.add_subplot(1, len(containers), i+1, projection='3d')
        
        # Calculate balance metrics
        left = sum(item.weight for item, (x,_,_), _ in container['placed_items'] if x < container_size[0]/2)
        right = sum(item.weight for item, (x,_,_), _ in container['placed_items'] if x >= container_size[0]/2)
        front = sum(item.weight for item, (_,y,_), _ in container['placed_items'] if y < container_size[1]/2)
        rear = sum(item.weight for item, (_,y,_), _ in container['placed_items'] if y >= container_size[1]/2)
        
        # Draw items
        for item, (x, y, z), (dx, dy, dz) in container['placed_items']:
            # Create cube vertices
            vertices = [
                [x, y, z],
                [x+dx, y, z],
                [x+dx, y+dy, z],
                [x, y+dy, z],
                [x, y, z+dz],
                [x+dx, y, z+dz],
                [x+dx, y+dy, z+dz],
                [x, y+dy, z+dz]
            ]
            
            # Define cube faces
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]], 
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[0], vertices[3], vertices[7], vertices[4]]
            ]
            
            # Color by weight
            color = plt.cm.plasma(item.weight / 5.0)  # Assuming max weight is 5
            
            ax.add_collection3d(Poly3DCollection(
                faces, 
                facecolors=color,
                edgecolors='k',
                alpha=0.7
            ))
        
        ax.set_xlim(0, container_size[0])
        ax.set_ylim(0, container_size[1])
        ax.set_zlim(0, container_size[2])
        ax.set_xlabel('X (Left/Right)')
        ax.set_ylabel('Y (Front/Back)')
        ax.set_zlabel('Z (Height)')
        
        ax.set_title(f"Container {i+1}\n"
                    f"Left/Right: {left:.1f}/{right:.1f} kg\n"
                    f"Front/Rear: {front:.1f}/{rear:.1f} kg\n"
                    f"Utilization: {container['volume_used']/(container_size[0]*container_size[1]*container_size[2]):.1%}")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Configuration
    CONTAINER_SIZE = (243, 606, 259)
    ITEMS = [
        *[Item(15, 30, 20, 1) for _ in range(200)],
        *[Item(150, 30, 30, 1) for _ in range(100)],
        *[Item(40, 40, 100, 1) for _ in range(50)]
    ]
    
    # Load model and predict
    MODEL_PATH = "best_model.keras"  # Change to your model path
    packed_containers, remaining_items = pack_items(MODEL_PATH, ITEMS, CONTAINER_SIZE)
    
    # Results
    print(f"\nPacking Results:")
    print(f"- Containers used: {len(packed_containers)}")
    print(f"- Items packed: {sum(len(c['placed_items']) for c in packed_containers)}/{len(ITEMS)}")
    print(f"- Items remaining: {len(remaining_items)}")
    
    # Visualization
    visualize_packing(packed_containers, CONTAINER_SIZE)