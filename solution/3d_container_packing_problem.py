import math
from typing import List, Tuple, Dict
import numpy as np

class BoxType:
    def __init__(self, d, l, h):
        self.d = d
        self.l = l
        self.h = h

class Container:
    def __init__(self, D, L, H, W):
        self.D = D
        self.H = H
        self.L = L
        self.W = W

class Layer:
    def __init__(self, box_type_idx, orientation, num_boxes):
        self.box_type_idx = box_type_idx
        self.orientation = orientation
        self.num_boxes = num_boxes
        self.height = None  # Inisialisasi height
        self.weight = 0.0   # Inisialisasi weight

class SolutionCandidate:
    def __init__(self, layers):
        self.layers = layers
        self.total_height = sum(layer.height for layer in layers)
        self.volume_utilization = 0.0
        self.weight_utilization = 0.0

def generate_layers(container, box_type):
    layers = {}
    for i, box in enumerate(box_type):
        for p in range(1,7):
            if p == 1:
                depth, length, height = box.d, box.l, box.h
            elif p == 2:
                depth, length, height = box.d, box.h, box.l
            elif p == 3:
                depth, length, height = box.l, box.d, box.h
            elif p == 4:
                depth, length, height = box.l, box.h, box.d
            elif p == 5:
                depth, length, height = box.h, box.d, box.l
            elif p == 6:
                depth, length, height = box.h, box.l, box.d
            
            n_depth = math.floor(container.D / depth)
            n_length = math.floor(container.L / length)

            num_boxes = n_depth * n_length
            
            if num_boxes > 0:
                layer = Layer(i, orientation=p, num_boxes=num_boxes)
                layer.height = height
                layers[(i, p)] = layer
    return layers

def generate_solution_candidates(container, box_types, layers):
    solutions = []
    t = len(box_types)

    for (i, p), layer in layers.items():
        if layer.height <= container.H:
            sol = SolutionCandidate([layer])
            sol.volume_utilization = (layer.num_boxes * box_types[i].d * box_types[i].l * box_types[i].h) / (container.D * container.L * container.H)
            solutions.append(sol)

    for (i1, p1), layer1 in layers.items():
        for (i2, p2), layer2 in layers.items():
            if i1 == i2 and p1 == p2:
                continue

            total_height = layer1.height + layer2.height
            if total_height <= container.H:
                sol = SolutionCandidate([layer1, layer2])
                vol1 = layer1.num_boxes * box_types[i1].d * box_types[i1].l * box_types[i1].h
                vol2 = layer2.num_boxes * box_types[i2].d * box_types[i2].l * box_types[i2].h
                sol.volume_utilization = (vol1 + vol2) / (container.D * container.L * container.H)
                solutions.append(sol)
    solutions.sort(key=lambda x: x.volume_utilization, reverse=True)
    return solutions

def pack_boxes(bin: Container, box_types: List[BoxType], solutions: List[SolutionCandidate], 
               num_boxes: List[int], box_weights: List[List[float]], 
               layers: Dict[Tuple[int, int], Layer]) -> Tuple[int, List[List[Layer]]]:
    """
    Packs boxes into bins using the best solution candidates.
    Args:
        bin (Container): The container to pack boxes into.
        box_types (List[BoxType]): List of box types with their dimensions and max weights.
        solutions (List[SolutionCandidate]): List of precomputed solution candidates.
        num_boxes (List[int]): Number of boxes available for each type.
        box_weights (List[List[float]]): Weights of the boxes for each type.
        layers (Dict[Tuple[int, int], Layer]): Precomputed layers for box types and orientations.
    Returns:
        Tuple[int, List[List[Layer]]]: Number of bins used and the packed bins with their layers.
    """
    # Initialize remaining boxes and weights (create deep copies)
    remaining_boxes = num_boxes.copy()
    remaining_weights = [weights.copy() for weights in box_weights]
    bins = []
    m = 0
    
    # Verification variables
    initial_counts = num_boxes.copy()
    packed_counts = [0] * len(box_types)
    
    while sum(remaining_boxes) > 0:
        # Sort remaining weights for each box type
        sorted_weights = []
        for i in range(len(box_types)):
            sorted_weights.append(sorted(remaining_weights[i], reverse=True))
        
        best_solution = None
        best_solution_idx = -1
        
        # Find the best solution candidate that fits in the bin
        for r, solution in enumerate(solutions):
            total_weight = 0.0
            feasible = True
            
            # Temporary variables for this solution attempt
            temp_packed = [0] * len(box_types)
            
            for layer in solution.layers:
                i = layer.box_type_idx
                if remaining_boxes[i] - temp_packed[i] < layer.num_boxes:
                    feasible = False
                    break
                
                half = layer.num_boxes // 2
                if half > 0:
                    total_weight += sum(sorted_weights[i][:half]) + sum(sorted_weights[i][-half:])
                else:
                    total_weight += sum(sorted_weights[i][:layer.num_boxes])
                
                temp_packed[i] += layer.num_boxes
            
            if feasible and total_weight <= bin.W:
                best_solution = solution
                best_solution_idx = r
                break
        
        if best_solution is None:
            # Fallback packing for remaining boxes
            bin_content = []
            for i in range(len(box_types)):
                if remaining_boxes[i] > 0:
                    for p in range(1, 7):
                        if (i, p) in layers:
                            num_to_pack = remaining_boxes[i]
                            new_layer = Layer(i, p, num_to_pack)
                            new_layer.height = layers[(i, p)].height
                            new_layer.weight = sum(sorted_weights[i][:num_to_pack])
                            
                            bin_content.append(new_layer)
                            packed_counts[i] += num_to_pack
                            remaining_boxes[i] = 0
                            sorted_weights[i] = []
                            break
            
            if bin_content:
                bins.append(bin_content)
                m += 1
                remaining_weights = sorted_weights
            break
        
        # Pack the best solution found
        bin_content = []
        for layer in best_solution.layers:
            i = layer.box_type_idx
            num_to_pack = layer.num_boxes
            new_layer = Layer(i, layer.orientation, num_to_pack)
            new_layer.height = layer.height
            
            half = num_to_pack // 2
            if half > 0:
                new_layer.weight = sum(sorted_weights[i][:half]) + sum(sorted_weights[i][-half:])
                del sorted_weights[i][:half]
                del sorted_weights[i][-(num_to_pack - half):]
            else:
                new_layer.weight = sum(sorted_weights[i][:num_to_pack])
                del sorted_weights[i][:num_to_pack]
            
            bin_content.append(new_layer)
            remaining_boxes[i] -= num_to_pack
            packed_counts[i] += num_to_pack
        
        bins.append(bin_content)
        m += 1
        remaining_weights = sorted_weights
    
    # Final verification
    for i in range(len(box_types)):
        if initial_counts[i] != packed_counts[i]:
            print(f"Error: Box type {i} initial={initial_counts[i]}, packed={packed_counts[i]}")
            return 0, []
    
    return m, bins

# panjang, lebar, tinggi, kapasitas kontainer
D = 243.8
L = 609.6
H = 259
bin = Container(D=D, L=L, H=H, W=6804)
    
box_types = [
    BoxType(d=0.15, l=0.3, h=0.2),
    BoxType(d=1.5, l=0.3, h=0.3),
    BoxType(d=0.4, l=0.4, h=1)
]

# Example problem instance
num_boxes = [200, 100, 50]  # Number of boxes for each type
box_weights = [
    [0.5] * 200,    # Weights for type 1 boxes (simplified)
    [2.0] * 100,    # Weights for type 2 boxes
    [1.0] * 50    # Weights for type 3 boxes
]

print("Running Stage 1: Generating layers...")
layers = generate_layers(bin, box_types)
print(f"Generated {len(layers)} layers")

print("\nRunning Stage 2: Generating solution candidates...")
solutions = generate_solution_candidates(bin, box_types, layers)
print(f"Generated {len(solutions)} solution candidates")
print(f"Best solution candidate has {solutions[0].volume_utilization*100:.1f}% volume utilization")

print("\nRunning Stage 3: Packing boxes into bins...")
num_bins, packed_bins = pack_boxes(bin, box_types, solutions, num_boxes, box_weights, layers)

print("\nPacking results:")
print(f"Total bins used: {num_bins}")
for i, bin_content in enumerate(packed_bins, 1):
    bin_weight = sum(layer.weight for layer in bin_content)
    bin_volume = sum(layer.num_boxes * (box_types[layer.box_type_idx].d * box_types[layer.box_type_idx].l * box_types[layer.box_type_idx].h)for layer in bin_content)
    bin_utilization = bin_volume / (bin.D * bin.L * bin.H)
    
    print(f"\nBin {i}:")
    print(f"- Weight: {bin_weight:.1f}/{bin.W} ({bin_weight/bin.W*100:.1f}%)")
    print(f"- Volume utilization: {bin_utilization*100:.1f}%")
    print("Layers:")
    for layer in bin_content:
        box_type = box_types[layer.box_type_idx]
        print(f"  - Type {layer.box_type_idx+1}, Orientation {layer.orientation}: {layer.num_boxes} boxes")
        print(f"    - Dimensions: {box_type.d}x{box_type.l}x{box_type.h}")
        print(f"    - Layer weight: {layer.weight:.1f}")