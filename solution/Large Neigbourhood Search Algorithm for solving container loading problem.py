import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import time
import random
import math

# --- Global Configuration / Parameters ---
CPU_TIME_LIMIT = 300  # seconds
OPTIMISE_BALANCE = True
LAMBDA = 0.4  # For Improvement Heuristic (Rnd < lambda)
GAMMA = 0.4   # For Improvement Heuristic (Rnd < gamma)
TAU = 0.5     # For Sort Containers (Rnd < tau)

# --- Data Structures ---

class Item:
    """
    Represents an item to be packed.
    """
    def __init__(self, item_id, width, height, length, weight, volume, profit,
                 priority=0, is_mandatory=False, is_fragile=False, item_type=None):
        self.item_id = item_id
        self.width = width
        self.height = height
        self.length = length
        self.weight = weight
        self.volume = volume
        self.profit = profit
        self.priority = priority
        self.is_mandatory = is_mandatory
        self.is_fragile = is_fragile
        self.item_type = item_type

        # Current position and orientation if packed
        self.position = None  # (x, y, z) tuple
        self.orientation = None # (width, height, length) tuple after rotation

    def get_dimensions_in_orientation(self, rotation_order):
        """
        Returns the dimensions (width, height, length) for a given rotation order.
        """
        dims = [self.width, self.height, self.length]
        # Generate all 6 permutations
        permutations = [
            (dims[0], dims[1], dims[2]),
            (dims[0], dims[2], dims[1]),
            (dims[1], dims[0], dims[2]),
            (dims[1], dims[2], dims[0]),
            (dims[2], dims[0], dims[1]),
            (dims[2], dims[1], dims[0])
        ]
        return permutations[rotation_order]

    def __repr__(self):
        return f"Item(ID={self.item_id}, Vol={self.volume}, Prio={self.priority})"


class FreeSpace:
    """
    Represents an empty rectangular space within a container.
    """
    def __init__(self, x, y, z, width, height, length):
        self.x = x
        self.y = y
        self.z = z
        self.width = width
        self.height = height
        self.length = length
        self.volume = width * height * length

    @property
    def max_x(self):
        return self.x + self.width

    @property
    def max_y(self):
        return self.y + self.height

    @property
    def max_z(self):
        return self.z + self.length

    def __repr__(self):
        return f"FreeSpace(x={self.x}, y={self.y}, z={self.z}, w={self.width}, h={self.height}, l={self.length})"


class Container:
    """
    Represents a container for packing items.
    """
    def __init__(self, container_id, width, height, length, max_weight,
                 cost=0, is_mandatory=False):
        self.container_id = container_id
        self.width = width
        self.height = height
        self.length = length
        self.max_weight = max_weight
        self.volume_capacity = width * height * length
        self.cost = cost
        self.is_mandatory = is_mandatory

        # Store references to actual Item objects (not copies)
        self.packed_items = []  # List of Item objects that are packed in this container
        self.packed_volume = 0.0
        self.packed_weight = 0.0
        self.packed_profit = 0.0
        
        # Initialize with one large free space representing the entire container
        self.free_spaces = [FreeSpace(0, 0, 0, self.width, self.height, self.length)]

    def add_item(self, item, position, orientation_dimensions):
        """
        Adds an item to the container. Updates packed volume, weight, and profit.
        `item` here is the *actual* Item object, not a copy.
        """
        self.packed_items.append(item)
        item.position = position
        item.orientation = orientation_dimensions
        self.packed_volume += (orientation_dimensions[0] * orientation_dimensions[1] * orientation_dimensions[2])
        self.packed_weight += item.weight
        self.packed_profit += item.profit

    def remove_item(self, item_to_remove):
        """
        Removes an item from the container. Updates packed volume, weight, and profit.
        `item_to_remove` is the *actual* Item object.
        """
        initial_packed_items_len = len(self.packed_items)
        self.packed_items = [
            item for item in self.packed_items if item is not item_to_remove
        ]

        if len(self.packed_items) < initial_packed_items_len:
            if item_to_remove.orientation:
                self.packed_volume -= (item_to_remove.orientation[0] * item_to_remove.orientation[1] * item_to_remove.orientation[2])
            self.packed_weight -= item_to_remove.weight
            self.packed_profit -= item_to_remove.profit
            item_to_remove.position = None
            item_to_remove.orientation = None
            return True
        return False

    def empty_container(self):
        """
        Empties all items from the container.
        """
        for item in list(self.packed_items):
            item.position = None
            item.orientation = None
        self.packed_items = []
        self.packed_volume = 0.0
        self.packed_weight = 0.0
        self.packed_profit = 0.0
        # Reset free spaces to the whole container
        self.free_spaces = [FreeSpace(0, 0, 0, self.width, self.height, self.length)]

    def get_volume_packed_ratio(self):
        """Calculates the ratio of packed volume to total capacity."""
        return self.packed_volume / self.volume_capacity if self.volume_capacity > 0 else 0


class Solution:
    """
    Represents a complete packing solution.
    Solution contains references to original Item and Container objects.
    Their packed/unpacked status is derived from Item.position and Container.packed_items.
    """
    def __init__(self, containers_snapshot, all_original_items):
        self.containers = containers_snapshot
        self.all_original_items = all_original_items
        self.unpacked_items = [item for item in self.all_original_items if item.position is None]

        self.dispersion = 0.0
        self.utilization = 0.0
        self.moment = 0.0
        self.distance = 0.0
        self.is_feasible = True
        self.packed_profit = sum(c.packed_profit for c in self.containers)

    def calculate_metrics(self, optimise_balance=True):
        """
        Calculates dispersion, utilization, moment, and distance for the solution.
        """
        total_packed_volume = sum(c.packed_volume for c in self.containers)
        total_container_capacity = sum(c.volume_capacity for c in self.containers)

        total_all_item_volume = sum(item.volume for item in self.all_original_items)

        self.dispersion = total_packed_volume / total_all_item_volume if total_all_item_volume > 0 else 0
        self.utilization = total_packed_volume / total_container_capacity if total_container_capacity > 0 else 0

        total_moment = 0.0
        total_distance = 0.0
        for container in self.containers:
            if optimise_balance:
                total_moment += self._calculate_moment(container)
            total_distance += self._calculate_distance(container)

        self.moment = total_moment
        self.distance = total_distance
        self.packed_profit = sum(c.packed_profit for c in self.containers)

    def _calculate_moment(self, container):
        """Placeholder for actual moment calculation (Equation 4)."""
        return random.random() * 100

    def _calculate_distance(self, container):
        """Placeholder for actual distance calculation (Equation 5)."""
        return random.random() * 50

    def is_better_than(self, other_solution):
        """
        Determines if the current solution is better than another.
        Prioritizes packed profit, then moment (lower is better), then distance (lower is better).
        """
        self.calculate_metrics(OPTIMISE_BALANCE)
        other_solution.calculate_metrics(OPTIMISE_BALANCE)

        if self.packed_profit > other_solution.packed_profit:
            return True
        elif self.packed_profit < other_solution.packed_profit:
            return False
        else:
            if self.moment < other_solution.moment:
                return True
            elif self.moment > other_solution.moment:
                return False
            else:
                return self.distance < other_solution.distance
        return False

    def get_solution_worksheet(self):
        """
        Generates a detailed output of the solution.
        """
        worksheet = []
        for container in self.containers:
            for item in container.packed_items:
                if item.position and item.orientation:
                    worksheet.append({
                        "container_id": container.container_id,
                        "item_id": item.item_id,
                        "x": item.position[0],
                        "y": item.position[1],
                        "z": item.position[2],
                        "packed_width": item.orientation[0],
                        "packed_height": item.orientation[1],
                        "packed_length": item.orientation[2]
                    })
        return worksheet


# --- Algorithm Class ---

class LargeNeighborhoodSearch:
    """
    Implements the Large Neighborhood Search (LNS) algorithm for container packing.
    Encapsulates all the sub-algorithms (Feasibility Check, Sort Items,
    Constructive Heuristic, Improvement Heuristic, etc.).
    """
    def __init__(self, items, containers, compatibility_data,
                 cpu_time_limit=CPU_TIME_LIMIT,
                 optimise_balance=OPTIMISE_BALANCE,
                 lambda_param=LAMBDA,
                 gamma_param=GAMMA,
                 tau_param=TAU,
                 constructive_heuristic_type="wall-building"):
        self.initial_items = items
        self.initial_containers = containers
        self.compatibility_data = compatibility_data
        self.cpu_time_limit = cpu_time_limit
        self.optimise_balance = optimise_balance
        self.lambda_param = lambda_param
        self.gamma_param = gamma_param
        self.tau_param = tau_param
        self.constructive_heuristic_type = constructive_heuristic_type

        self.best_known_solution = None

        for item in self.initial_items:
            item.position = None
            item.orientation = None
        for container in self.initial_containers:
            container.empty_container()


    def _perform_initial_feasibility_checks_internal(self, item, container):
        """
        Helper for AddItemToContainer: Checks basic feasibility (volume/weight).
        """
        if container.packed_volume + item.volume > container.volume_capacity:
            return False
        if container.packed_weight + item.weight > container.max_weight:
            return False
        return True
    
    def _is_overlapping(self, item_bbox1, item_bbox2):
        """Checks if two bounding boxes overlap."""
        x1_min, y1_min, z1_min = item_bbox1[0]
        x1_max, y1_max, z1_max = item_bbox1[1]

        x2_min, y2_min, z2_min = item_bbox2[0]
        x2_max, y2_max, z2_max = item_bbox2[1]

        return not (x1_max <= x2_min or x1_min >= x2_max or
                    y1_max <= y2_min or y1_min >= y2_max or
                    z1_max <= z2_min or z1_min >= z2_max)

    def _update_free_spaces(self, container, packed_item_position, packed_item_orientation):
        """
        Updates the list of free spaces after an item has been packed.
        This is a core part of the Maximal Empty Space (MES) approach.
        It generates new free spaces by "cutting" the occupied space from existing free spaces.
        """
        px, py, pz = packed_item_position
        pw, ph, pl = packed_item_orientation
        packed_item_bbox = ((px, py, pz), (px + pw, py + ph, pz + pl))

        new_free_spaces = []
        
        # Take a copy of the current free spaces to iterate over, as we'll modify the original list
        current_free_spaces = list(container.free_spaces)
        
        for fs in current_free_spaces:
            fs_bbox = ((fs.x, fs.y, fs.z), (fs.max_x, fs.max_y, fs.max_z))

            if not self._is_overlapping(fs_bbox, packed_item_bbox):
                new_free_spaces.append(fs) # If no overlap, free space remains
                continue

            # If there is an overlap, the free space needs to be cut
            # Create new free spaces by cutting the current free space around the packed item
            
            # Cut along X-axis
            if fs.x < px:
                new_free_spaces.append(FreeSpace(fs.x, fs.y, fs.z, px - fs.x, fs.height, fs.length))
            if fs.max_x > px + pw:
                new_free_spaces.append(FreeSpace(px + pw, fs.y, fs.z, fs.max_x - (px + pw), fs.height, fs.length))

            # Cut along Y-axis (only if there's an overlap in X and Z)
            x_overlap_start = max(fs.x, px)
            x_overlap_end = min(fs.max_x, px + pw)
            z_overlap_start = max(fs.z, pz)
            z_overlap_end = min(fs.max_z, pz + pl)

            if x_overlap_end > x_overlap_start and z_overlap_end > z_overlap_start:
                if fs.y < py:
                    new_free_spaces.append(FreeSpace(x_overlap_start, fs.y, z_overlap_start, x_overlap_end - x_overlap_start, py - fs.y, z_overlap_end - z_overlap_start))
                if fs.max_y > py + ph:
                    new_free_spaces.append(FreeSpace(x_overlap_start, py + ph, z_overlap_start, x_overlap_end - x_overlap_start, fs.max_y - (py + ph), z_overlap_end - z_overlap_start))

            # Cut along Z-axis (only if there's an overlap in X and Y, and not fully covered by previous Y-cuts)
            x_overlap_start_z = max(fs.x, px)
            x_overlap_end_z = min(fs.max_x, px + pw)
            y_overlap_start_z = max(fs.y, py)
            y_overlap_end_z = min(fs.max_y, py + ph)

            if x_overlap_end_z > x_overlap_start_z and y_overlap_end_z > y_overlap_start_z:
                if fs.z < pz:
                    new_free_spaces.append(FreeSpace(x_overlap_start_z, y_overlap_start_z, fs.z, x_overlap_end_z - x_overlap_start_z, y_overlap_end_z - y_overlap_start_z, pz - fs.z))
                if fs.max_z > pz + pl:
                    new_free_spaces.append(FreeSpace(x_overlap_start_z, y_overlap_start_z, pz + pl, x_overlap_end_z - x_overlap_start_z, y_overlap_end_z - y_overlap_start_z, fs.max_z - (pz + pl)))
        
        # Filter out invalid free spaces (zero volume or outside container boundaries)
        new_free_spaces = [
            fs for fs in new_free_spaces
            if fs.width > 0 and fs.height > 0 and fs.length > 0 and
            fs.x >= 0 and fs.y >= 0 and fs.z >= 0 and
            fs.max_x <= container.width + 1e-9 and fs.max_y <= container.height + 1e-9 and fs.max_z <= container.length + 1e-9
        ]
        
        # A simple merge step for directly adjacent spaces (optional, but helps prevent fragmentation)
        # This part can be significantly more complex for full 3D MER consolidation.
        # For a basic approach, we just add unique spaces.
        
        container.free_spaces = []
        for new_fs in new_free_spaces:
            is_duplicate = False
            for existing_fs in container.free_spaces:
                if (existing_fs.x == new_fs.x and existing_fs.y == new_fs.y and existing_fs.z == new_fs.z and
                    existing_fs.width == new_fs.width and existing_fs.height == new_fs.height and existing_fs.length == new_fs.length):
                    is_duplicate = True
                    break
            if not is_duplicate:
                container.free_spaces.append(new_fs)


    def _check_bottom_support_internal(self, new_item_orientation, container, ox, oy, oz):
        """
        Helper for AddItemToContainer: Checks for sufficient support from items below.
        This is crucial for gravity-based packing.
        """
        # If the item is placed on the container floor (y=0), it's always supported.
        if oy == 0:
            return True

        ow, oh, ol = new_item_orientation
        
        # Define the bottom surface of the new item
        new_item_bottom_x_range = (ox, ox + ow)
        new_item_bottom_z_range = (oz, oz + ol)

        supported_area = 0.0
        
        # Iterate over packed items to find supporting surfaces
        for existing_item in container.packed_items:
            ex, ey, ez = existing_item.position
            ew, eh, el = existing_item.orientation
            
            # Check if the existing item's top surface is directly below the new item's bottom surface
            # Use a small tolerance for floating point comparisons
            if abs((ey + eh) - oy) < 1e-6: # The existing item's top surface is at the new item's bottom Y-coordinate
                
                # Calculate the 2D overlap area in the XZ plane
                overlap_x_start = max(new_item_bottom_x_range[0], ex)
                overlap_x_end = min(new_item_bottom_x_range[1], ex + ew)
                
                overlap_z_start = max(new_item_bottom_z_range[0], ez)
                overlap_z_end = min(new_item_bottom_z_range[1], ez + el)
                
                overlap_x = max(0, overlap_x_end - overlap_x_start)
                overlap_z = max(0, overlap_z_end - overlap_z_start)
                
                supported_area += (overlap_x * overlap_z)
        
        bottom_area_new_item = ow * ol
        # Define a threshold for required support (e.g., 95% of its own base area)
        support_threshold_ratio = 0.95 

        # If the supported area is less than the required threshold, the item is not sufficiently supported
        if supported_area < bottom_area_new_item * support_threshold_ratio:
            return False
        
        return True

    def _add_item_to_container_internal(self, item_to_pack, container_k):
        """
        Attempts to add an item to a container using a Maximal Empty Space (MES) strategy.
        This modifies the `item_to_pack`'s position and orientation directly.
        """
        if not self._perform_initial_feasibility_checks_internal(item_to_pack, container_k):
            return False

        # Sort free spaces to find the "best" place to put the item.
        # Prioritize free spaces with smallest height (to build layers), then by smallest volume.
        container_k.free_spaces.sort(key=lambda fs: (fs.y, fs.height, fs.volume))

        possible_orientations = []
        for r_idx in range(6):
            possible_orientations.append(item_to_pack.get_dimensions_in_orientation(r_idx))
        
        # Remove duplicate orientations if the item has symmetric dimensions
        possible_orientations = list(set(possible_orientations))

        best_position = None
        best_orientation = None
        
        # Try to fit the item into each free space with each possible orientation
        for fs in container_k.free_spaces:
            for oriented_width, oriented_height, oriented_length in possible_orientations:
                # Check if the item fits within the current free space
                if (oriented_width <= fs.width and
                    oriented_height <= fs.height and
                    oriented_length <= fs.length):

                    # Candidate position is the origin of the free space
                    candidate_x, candidate_y, candidate_z = fs.x, fs.y, fs.z

                    # Perform physics-based checks (overlap and support)
                    temp_item_bbox = (
                        (candidate_x, candidate_y, candidate_z),
                        (candidate_x + oriented_width, candidate_y + oriented_height, candidate_z + oriented_length)
                    )

                    # Check for overlaps with already packed items
                    is_overlap = False
                    for existing_item in container_k.packed_items:
                        ex, ey, ez = existing_item.position
                        ew, eh, el = existing_item.orientation
                        existing_item_bbox = ((ex, ey, ez), (ex + ew, ey + eh, ez + el))
                        if self._is_overlapping(temp_item_bbox, existing_item_bbox):
                            is_overlap = True
                            break
                    if is_overlap:
                        continue # Cannot place here due to overlap

                    # Check for bottom support
                    if not self._check_bottom_support_internal(
                        (oriented_width, oriented_height, oriented_length), container_k, 
                        candidate_x, candidate_y, candidate_z
                    ):
                        continue # Cannot place here due to lack of support

                    # If all checks pass, this is a valid placement.
                    # For now, we take the first valid placement.
                    best_position = (candidate_x, candidate_y, candidate_z)
                    best_orientation = (oriented_width, oriented_height, oriented_length)
                    break # Found a spot for this free space and orientation
            if best_position:
                break # Found a spot for this item in some free space

        if best_position is None:
            return False # Item could not be placed

        # If a suitable position is found, add the item and update free spaces
        container_k.add_item(item_to_pack, best_position, best_orientation)
        self._update_free_spaces(container_k, best_position, best_orientation)
        
        return True # Item was successfully added


    def _feasibility_check(self):
        """
        Algorithm 1, Step 1: Checks the feasibility of the instance.
        """
        print("Performing feasibility check...")
        for item in self.initial_items:
            if not any(item.volume <= c.volume_capacity and item.weight <= c.max_weight for c in self.initial_containers):
                print(f"Feasibility Error: Item {item.item_id} is too large or heavy for any container.")
                return False

        total_mandatory_volume = sum(item.volume for item in self.initial_items if item.is_mandatory)
        total_mandatory_weight = sum(item.weight for item in self.initial_items if item.is_mandatory)
        total_container_capacity_volume = sum(c.volume_capacity for c in self.initial_containers)
        total_container_capacity_weight = sum(c.max_weight for c in self.initial_containers)

        if total_mandatory_volume > total_container_capacity_volume:
            print("Feasibility Error: Total mandatory item volume exceeds total container capacity.")
            return False
        if total_mandatory_weight > total_container_capacity_weight:
            print("Feasibility Error: Total mandatory item weight exceeds total container capacity.")
            return False

        return True

    def _sort_items(self, items_to_sort, item_sort_criterion="volume"):
        """
        Algorithm 4: Sorts the set of items based on priority, size, and profit.
        """
        mandatory_items = sorted([item for item in items_to_sort if item.is_mandatory],
                                 key=lambda item: item.priority, reverse=True)
        non_mandatory_items = sorted([item for item in items_to_sort if not item.is_mandatory],
                                     key=lambda item: item.priority, reverse=True)

        if item_sort_criterion == "volume":
            max_weight = max([item.weight for item in self.initial_items]) + 1 if self.initial_items else 1
            mandatory_items.sort(key=lambda item: item.volume * max_weight + item.weight, reverse=True)
        elif item_sort_criterion == "weight":
            max_volume = max([item.volume for item in self.initial_items]) + 1 if self.initial_items else 1
            mandatory_items.sort(key=lambda item: item.weight * max_volume + item.volume, reverse=True)
        else: # Default or 'max_dimension'
            max_volume = max([item.volume for item in self.initial_items]) + 1 if self.initial_items else 1
            mandatory_items.sort(key=lambda item: max(item.width, item.height, item.length) * max_volume + item.volume, reverse=True)

        non_mandatory_items.sort(key=lambda item: item.profit / item.volume if item.volume > 0 else 0, reverse=True)
        return mandatory_items + non_mandatory_items

    def _sort_containers(self, containers_to_sort, solution):
        """
        Algorithm 5: Sorts containers based on priority, packed volume, and cost/volume capacity ratio.
        """
        if random.random() < self.tau_param:
            containers_to_sort.sort(key=lambda c: c.is_mandatory, reverse=True)

            mandatory_containers = [c for c in containers_to_sort if c.is_mandatory]
            non_mandatory_containers = [c for c in containers_to_sort if not c.is_mandatory]

            mandatory_containers.sort(key=lambda c: c.packed_volume, reverse=True)
            non_mandatory_containers.sort(key=lambda c: (
                c.packed_volume,
                -(c.cost / c.volume_capacity if c.volume_capacity > 0 else 0)
            ), reverse=True)

            sorted_containers = mandatory_containers + non_mandatory_containers
        else:
            sorted_containers = random.sample(containers_to_sort, len(containers_to_sort))
        return sorted_containers

    def _constructive_heuristic(self, items_to_consider, containers_to_use):
        """
        Algorithm 2: Constructs an initial solution by adding items to containers.
        `items_to_consider` are the specific item instances available for packing in this heuristic run.
        """
        print(f"\nRunning Constructive Heuristic ({self.constructive_heuristic_type})...")

        # Sort containers (Algorithm 5)
        temp_solution_for_sorting = Solution(containers_to_use, items_to_consider)
        sorted_containers = self._sort_containers(containers_to_use, temp_solution_for_sorting)

        # Sort items (Algorithm 4)
        sorted_items_for_constructive = self._sort_items(items_to_consider)

        # Try to pack items into containers
        for k in sorted_containers:
            # Re-initialize free spaces for the container for this packing attempt
            # This is critical for the MES approach.
            k.free_spaces = [FreeSpace(0, 0, 0, k.width, k.height, k.length)]

            for item_type_i in sorted_items_for_constructive:
                is_item_added = True
                while is_item_added:
                    item_instance_to_pack = None
                    # Find an *unpacked instance* of item_type_i from the `items_to_consider` list
                    for unpacked_item_inst in items_to_consider:
                        if unpacked_item_inst is item_type_i and unpacked_item_inst.position is None:
                            item_instance_to_pack = unpacked_item_inst
                            break

                    if item_instance_to_pack:
                        is_item_added = self._add_item_to_container_internal(item_instance_to_pack, k)
                    else:
                        is_item_added = False # No more items of this type to pack or no more space

        incumbent_solution = Solution(self.initial_containers, self.initial_items)

        incumbent_feasible = True
        for original_mandatory_item in [item for item in self.initial_items if item.is_mandatory]:
            if original_mandatory_item.position is None:
                incumbent_feasible = False
                break
        incumbent_solution.is_feasible = incumbent_feasible
        incumbent_solution.calculate_metrics(self.optimise_balance)

        return incumbent_solution

    def _improvement_heuristic(self, incumbent_solution, best_known_solution):
        """
        Algorithm 3: Improves an existing solution using destruction and reconstruction.
        Incumbent and best_known are `Solution` objects, which contain references to the actual
        Item and Container objects.
        """
        # Ensure the current state of self.initial_containers and self.initial_items
        # reflects the best_known_solution before starting destruction.
        for container in self.initial_containers:
            container.empty_container() # Clears items and resets free_spaces

        for bk_container in best_known_solution.containers:
            original_container_instance = next((c for c in self.initial_containers if c.container_id == bk_container.container_id), None)
            if original_container_instance:
                for bk_item in bk_container.packed_items:
                    original_item_instance = next((item for item in self.initial_items if item is bk_item), None)
                    if original_item_instance:
                        original_container_instance.add_item(original_item_instance, bk_item.position, bk_item.orientation)
                        self._update_free_spaces(original_container_instance, original_item_instance.position, original_item_instance.orientation)


        # Destruction Phase
        removed_items_for_repacking = []
        for k_container in self.initial_containers:
            Rnd_container = random.random()
            if Rnd_container < self.lambda_param: # Use lambda for container destruction probability
                items_to_remove_from_container = list(k_container.packed_items)
                for item_obj in items_to_remove_from_container:
                    if k_container.remove_item(item_obj):
                        removed_items_for_repacking.append(item_obj)
                k_container.free_spaces = [FreeSpace(0, 0, 0, k_container.width, k_container.height, k_container.length)]

            else:
                items_in_container = list(k_container.packed_items)
                if items_in_container:
                    num_to_remove = random.randint(0, len(items_in_container) // 2)
                    items_to_unpack = random.sample(items_in_container, num_to_remove)
                    for item_obj in items_to_unpack:
                        if k_container.remove_item(item_obj):
                            removed_items_for_repacking.append(item_obj)
                    
                    # Re-pack remaining items in this container to consolidate free spaces
                    temp_packed_items = list(k_container.packed_items)
                    k_container.empty_container()
                    for item_repack in temp_packed_items:
                        self._add_item_to_container_internal(item_repack, k_container)

        all_unpacked_for_reconstruction = [item for item in self.initial_items if item.position is None]
        for item in removed_items_for_repacking:
            if item.position is not None:
                item.position = None
                item.orientation = None
        
        items_to_pack_in_reconstruction = list(set(all_unpacked_for_reconstruction + removed_items_for_repacking))

        reconstructed_solution = self._constructive_heuristic(
            items_to_pack_in_reconstruction,
            self.initial_containers
        )
        reconstructed_solution.calculate_metrics(self.optimise_balance)

        if reconstructed_solution.is_feasible and reconstructed_solution.is_better_than(best_known_solution):
            best_known_solution = reconstructed_solution

        Rnd_reorg = random.random()
        if Rnd_reorg < self.gamma_param:
            temp_containers_for_reorg = []
            for c_orig in self.initial_containers:
                temp_c = Container(c_orig.container_id, c_orig.width, c_orig.height, c_orig.length, c_orig.max_weight, c_orig.cost, c_orig.is_mandatory)
                temp_containers_for_reorg.append(temp_c)
            
            for rc in reconstructed_solution.containers:
                temp_c_match = next((tc for tc in temp_containers_for_reorg if tc.container_id == rc.container_id), None)
                if temp_c_match:
                    for r_item in rc.packed_items:
                        temp_c_match.add_item(r_item, r_item.position, r_item.orientation)
                        self._update_free_spaces(temp_c_match, r_item.position, r_item.orientation)
            
            temp_items_for_reorg = [item for item in self.initial_items if item.position is None]

            for k_reorg in temp_containers_for_reorg:
                Rnd_reorg_container = random.random()
                items_to_repack_from_k_reorg = []

                if Rnd_reorg_container < (1 - k_reorg.get_volume_packed_ratio()) / 2:
                    items_in_k_reorg_copy = list(k_reorg.packed_items)
                    for item_obj in items_in_k_reorg_copy:
                        if k_reorg.remove_item(item_obj):
                            items_to_repack_from_k_reorg.append(item_obj)
                    k_reorg.free_spaces = [FreeSpace(0, 0, 0, k_reorg.width, k_reorg.height, k_reorg.length)]
                else:
                    items_in_container_reorg = list(k_reorg.packed_items)
                    if items_in_container_reorg:
                        num_to_remove_reorg = random.randint(0, len(items_in_container_reorg) // 2)
                        items_to_unpack_reorg = random.sample(items_in_container_reorg, num_to_remove_reorg)
                        for item_obj_reorg in items_to_unpack_reorg:
                            if k_reorg.remove_item(item_obj_reorg):
                                items_to_repack_from_k_reorg.append(item_obj_reorg)
                        
                        temp_packed_items_in_k = list(k_reorg.packed_items)
                        k_reorg.empty_container()
                        for item_repack_in_k in temp_packed_items_in_k:
                            self._add_item_to_container_internal(item_repack_in_k, k_reorg)

                temp_items_for_reorg.extend(items_to_repack_from_k_reorg)
            
            reorganized_solution = self._constructive_heuristic(
                list(set(self.initial_items)),
                temp_containers_for_reorg
            )
            reorganized_solution.calculate_metrics(self.optimise_balance)

            if reorganized_solution.is_feasible and reorganized_solution.is_better_than(best_known_solution):
                for container_orig in self.initial_containers:
                    container_orig.empty_container()
                
                for r_container in reorganized_solution.containers:
                    container_target = next((c for c in self.initial_containers if c.container_id == r_container.container_id), None)
                    if container_target:
                        for r_item in r_container.packed_items:
                            item_target = next((i for i in self.initial_items if i is r_item), None)
                            if item_target:
                                container_target.add_item(item_target, r_item.position, r_item.orientation)
                                self._update_free_spaces(container_target, item_target.position, item_target.orientation)
                
                best_known_solution = Solution(self.initial_containers, self.initial_items)
                best_known_solution.calculate_metrics(self.optimise_balance)
                
        return best_known_solution

    def run(self):
        """
        Main execution method for the Large Neighborhood Search Algorithm.
        """
        start_time = time.time()
        print("Starting Large Neighborhood Search Algorithm...")

        if not self._feasibility_check():
            print("Initial feasibility check failed. Cannot proceed.")
            return None, (time.time() - start_time)

        sorted_initial_items = self._sort_items(self.initial_items)

        incumbent_solution = self._constructive_heuristic(
            sorted_initial_items,
            self.initial_containers
        )
        self.best_known_solution = incumbent_solution

        print(f"\nInitial Solution - Feasible: {incumbent_solution.is_feasible}, "
              f"Packed Volume: {sum(c.packed_volume for c in incumbent_solution.containers):.2f}, "
              f"Unpacked Items: {len(incumbent_solution.unpacked_items)}, "
              f"Total Profit: {sum(c.packed_profit for c in incumbent_solution.containers):.2f}")

        print(f"\nStarting Improvement Phase (Time Limit: {self.cpu_time_limit} seconds)...")
        improvement_start_time = time.time()
        iteration_count = 0
        while (time.time() - improvement_start_time) < self.cpu_time_limit:
            iteration_count += 1
            print(f"--- Iteration {iteration_count} (Time: {(time.time() - improvement_start_time):.2f}s) ---")
            
            updated_best_solution = self._improvement_heuristic(incumbent_solution, self.best_known_solution)

            self.best_known_solution = Solution(self.initial_containers, self.initial_items)
            self.best_known_solution.calculate_metrics(self.optimise_balance)

            incumbent_solution = Solution(self.initial_containers, self.initial_items)
            incumbent_solution.calculate_metrics(self.optimise_balance)

            print(f"  Current Best - Feasible: {self.best_known_solution.is_feasible}, "
                  f"Packed Volume: {sum(c.packed_volume for c in self.best_known_solution.containers):.2f}, "
                  f"Unpacked Items: {len(self.best_known_solution.unpacked_items)}, "
                  f"Total Profit: {sum(c.packed_profit for c in self.best_known_solution.containers):.2f}")

        total_time = time.time() - start_time
        print(f"\nLNS Algorithm Finished. Total Time: {total_time:.2f} seconds.")

        solution_worksheet = self.best_known_solution.get_solution_worksheet()
        return solution_worksheet, total_time


# Helper for plotting cuboids (outside the class for reusability with matplotlib)
def draw_cuboid(ax, origin, dimensions, color='blue', alpha=0.6, edgecolors='black', linestyle='-'):
    """
    Plots a cuboid on a 3D matplotlib axis with faces filled.
    """
    x, y, z = origin
    dx, dy, dz = dimensions

    vertices = np.array([
        [x, y, z],
        [x + dx, y, z],
        [x, y + dy, z],
        [x, y, z + dz],
        [x + dx, y + dy, z],
        [x + dx, y, z + dz],
        [x, y + dy, z + dz],
        [x + dx, y + dy, z + dz]
    ])

    faces = [
        [vertices[0], vertices[1], vertices[4], vertices[2]], # Bottom
        [vertices[0], vertices[1], vertices[5], vertices[3]], # Front
        [vertices[0], vertices[2], vertices[6], vertices[3]], # Left
        [vertices[1], vertices[4], vertices[7], vertices[5]], # Right
        [vertices[2], vertices[4], vertices[7], vertices[6]], # Back
        [vertices[3], vertices[5], vertices[7], vertices[6]]  # Top
    ]

    ax.add_collection3d(Poly3DCollection(faces, facecolor=color, linewidth=1, edgecolors=edgecolors, alpha=alpha))


def visualize_packing(final_solution_worksheet, containers_data):
    """
    Generates a 3D visualization of the packed items within containers.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    max_x = max(c.width for c in containers_data) if containers_data else 0
    max_y = max(c.height for c in containers_data) if containers_data else 0
    max_z = max(c.length for c in containers_data) if containers_data else 0

    container_colors = plt.cm.get_cmap('Pastel1', len(containers_data))
    for i, container in enumerate(containers_data):
        draw_cuboid(ax, (0, 0, 0), (container.width, container.height, container.length),
                    color=container_colors(i), alpha=0.1, edgecolors='gray', linestyle='--')
        ax.text(container.width / 2, container.height / 2, container.length + 0.5,
                f"Container {container.container_id}", color='gray', fontsize=8, ha='center', va='bottom')

    unique_item_ids = sorted(list(set(entry['item_id'] for entry in final_solution_worksheet)))
    item_colors = plt.cm.get_cmap('viridis', len(unique_item_ids))
    item_id_to_color_idx = {item_id: i for i, item_id in enumerate(unique_item_ids)}

    for entry in final_solution_worksheet:
        container_id = entry['container_id']
        item_id = entry['item_id']
        x, y, z = entry['x'], entry['y'], entry['z']
        packed_width, packed_height, packed_length = entry['packed_width'], entry['packed_height'], entry['packed_length']

        item_color = item_colors(item_id_to_color_idx.get(item_id, 0))

        draw_cuboid(ax, (x, y, z), (packed_width, packed_height, packed_length),
                    color=item_color, alpha=0.8)
        ax.text(x + packed_width/2, y + packed_height/2, z + packed_length/2,
                f"I{item_id}", color='white', fontsize=6, ha='center', va='center', weight='bold')


    ax.set_xlabel('Width (X)')
    ax.set_ylabel('Height (Y)')
    ax.set_zlabel('Length (Z)')
    ax.set_title('3D Packing Visualization')

    ax.set_xlim([0, max_x * 1.1])
    ax.set_ylim([0, max_y * 1.1])
    ax.set_zlim([0, max_z * 1.1])

    ax.grid(True)

    plt.tight_layout()
    plt.savefig('3d_packing_visualization.png')
    plt.show()

# --- Main execution block to generate visualization ---
# Konversi kaki ke sentimeter
FEET_TO_CM = 30.48

# Dimensi item yang lebih "normal" dan proporsional dengan kontainer
items_data = [
    # Item yang cukup besar, mendekati sepertiga dimensi kontainer
    Item(1, 70, 70, 70, 20, 70**3, 500, priority=1, is_mandatory=True, item_type="large_box"),
    # Item berukuran sedang
    Item(2, 50, 50, 50, 15, 50**3, 300, priority=2, is_mandatory=True, item_type="medium_box"),
    # Item panjang dan tipis (seperti pipa atau batang)
    Item(3, 30, 30, 150, 25, 30*30*150, 400, priority=1, is_mandatory=True, item_type="long_pipe"),
    # Item datar dan lebar (seperti palet)
    Item(4, 100, 20, 80, 18, 100*20*80, 350, priority=0, is_mandatory=True, item_type="flat_pallet"),
    # Item yang lebih kecil tapi masih signifikan
    Item(5, 40, 40, 40, 10, 40**3, 200, priority=2, is_mandatory=True, item_type="small_box"),
    # Item sedang dengan berat lebih tinggi
    Item(6, 60, 60, 60, 30, 60**3, 450, priority=0, is_mandatory=True, item_type="heavy_box"),
    # Item berukuran sedang lainnya
    Item(7, 45, 45, 90, 12, 45*45*90, 250, priority=0, is_mandatory=True, item_type="tall_box"),
    # Item panjang dan sempit
    Item(8, 25, 25, 200, 22, 25*25*200, 380, priority=1, is_mandatory=True, item_type="narrow_long_box"),
]

containers_data = [
    Container(101, 
              width=7.7 * FEET_TO_CM,  # Lebar 20ft container (sekitar 234.7 cm)
              height=7.8 * FEET_TO_CM, # Tinggi 20ft container (sekitar 237.5 cm)
              length=19.38 * FEET_TO_CM, # Panjang 20ft container (sekitar 590.7 cm)
              max_weight=28000, # dalam kg
              cost=100, is_mandatory=True),
]

all_items_instances = []
for item in items_data:
    # Buat lebih banyak instance item untuk pengujian packing yang lebih baik
    for _ in range(random.randint(5, 20)): # Jumlah instance item yang lebih banyak
        new_item_instance = Item(
            item.item_id, item.width, item.height, item.length, item.weight, item.volume, item.profit,
            item.priority, item.is_mandatory, item.is_fragile, item.item_type
        )
        all_items_instances.append(new_item_instance)
random.shuffle(all_items_instances) # Acak urutan item

compatibility_rules = {
    "cannot_coexist": [] # Aturan kompatibilitas, dikosongkan untuk contoh ini
}

lns_solver = LargeNeighborhoodSearch(
    all_items_instances, containers_data, compatibility_rules,
    cpu_time_limit=500, # Batas waktu CPU dalam detik
    constructive_heuristic_type="layer-building"
)
final_solution_worksheet, total_run_time = lns_solver.run()

if final_solution_worksheet:
    print("\n--- Final Solution Worksheet ---")
    for entry in final_solution_worksheet:
        print(f"Container {entry['container_id']}: Item {entry['item_id']} at ({entry['x']:.2f},{entry['y']:.2f},{entry['z']:.2f}) "
              f"Dims: ({entry['packed_width']:.2f}x{entry['packed_height']:.2f}x{entry['packed_length']:.2f})")
    
    # Hitung volume dan rasio yang terisi untuk solusi terbaik
    final_packed_volume = sum(c.packed_volume for c in lns_solver.best_known_solution.containers)
    final_container_capacity = sum(c.volume_capacity for c in lns_solver.best_known_solution.containers)
    final_packed_ratio = final_packed_volume / final_container_capacity if final_container_capacity > 0 else 0
    final_unpacked_items_count = len(lns_solver.best_known_solution.unpacked_items)

    print(f"\nTotal execution time: {total_run_time:.2f} seconds.")
    print(f"Final Packed Volume: {final_packed_volume:.2f} cubic cm")
    print(f"Total Container Capacity: {final_container_capacity:.2f} cubic cm")
    print(f"Volume Packed Ratio: {final_packed_ratio:.2%}")
    print(f"Number of Unpacked Items: {final_unpacked_items_count}")


    visualize_packing(final_solution_worksheet, containers_data)
else:
    print("No solution found to visualize.")