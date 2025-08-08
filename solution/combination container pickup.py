from itertools import product

class Item:
    def __init__(self, w, l, h, weight, name):
        self.width = w
        self.length = l
        self.height = h
        self.weight = weight
        self.name = name

    @property
    def volume(self):
        return self.width * self.length * self.height



def find_combinations(items, container_size, max_weight):
    container_vol = container_size[0] * container_size[1] * container_size[2]
    valid_combos = []
    item_name = [item.name for item in items]

    max_counts = []
    for item in items:
        max_by_volume = container_vol // item.volume
        max_by_weight = max_weight // item.weight
        max_counts.append(int(min(max_by_volume, max_by_weight, 200)))

    for counts in product(*(range(m + 1) for m in max_counts)):
        total_vol = sum(item.volume * count for item, count in zip(items, counts))
        total_weight = sum(item.weight * count for item, count in zip(items, counts))

        if total_vol <= container_vol and total_weight <= max_weight:
            valid_combos.append((counts, total_vol, total_weight))
        

    return valid_combos


if __name__ == "__main__":
    D = 243.8
    L = 609.6
    H = 259.0
    MAX_WEIGHT = 24000

    items = [
        Item(15, 30, 20, 6, "Box1"),
        Item(150, 30, 30, 10, "Box2"),
        Item(40, 40, 100, 7, "Box3")
    ]

    valid_combos = find_combinations(items, (D, L, H), MAX_WEIGHT)
    print(f"Total kombinasi valid: {len(valid_combos)}")
    for combos, vol, weight in valid_combos[-10:]:  # tampilkan 10 contoh
        for combo, item in zip(combos, items):
            print(f"barang {item.name} dengan jumlah {combo}")
        print(f"Volume={vol:.2f}", f"Weight={weight:.2f}")

