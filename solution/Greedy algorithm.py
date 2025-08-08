import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch

class Item:
    def __init__(self, dx, dy, dz, weight, name=None):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.weight = weight
        self.volume = dx * dy * dz
        self.name = name or f"Item_{dx}x{dy}x{dz}_{weight}kg"
        self.orientations = [
            (dx, dy, dz),
            (dx, dz, dy),
            (dy, dx, dz),
            (dy, dz, dx),
            (dz, dx, dy),
            (dz, dy, dx)
        ]

    def __repr__(self):
        return self.name

class ContainerPackingEnv:
    def __init__(self, container_size=(10, 10, 10), items=[], max_containers=10, max_weight_per_container=3000):
        self.container_size = container_size
        self.width, self.depth, self.height = container_size
        self.items = sorted(items, key=lambda x: (-x.volume, -x.weight))
        self.max_containers = max_containers
        self.max_weight_per_container = max_weight_per_container  # ⬅️ Tambahan
        self.reset()

    def reset(self):
        self.containers = []
        self.unplaced = self.items.copy()
        self.placed_items = []
        self._add_new_container()
        return self._get_state()

    def _add_new_container(self):
        if len(self.containers) >= self.max_containers:
            return False
        self.containers.append({
            'free': [(0, 0, 0, self.width, self.depth, self.height)],
            'placed': [],
            'weight': 0,
            'volume_used': 0
        })
        return True

    def _get_state(self):
        current_container = self.containers[-1] if self.containers else None
        current_utilization = (current_container['volume_used'] / 
                             (self.width * self.depth * self.height)) if current_container else 0
        return (len(self.unplaced), len(self.containers), current_utilization)
    
    def _can_place(self, container, item, pos):
        x, y, z = pos

        if (x + item.dx > self.width or
            y + item.dy > self.depth or
            z + item.dz > self.height):
            return False

        for placed_item in container['placed']:
            pi = placed_item['item']
            px, py, pz = placed_item['x'], placed_item['y'], placed_item['z']

            if not (x + item.dx <= px or x >= px + pi.dx or
                    y + item.dy <= py or y >= py + pi.dy or
                    z + item.dz <= pz or z >= pz + pi.dz):
                return False

        # Tambahan: pastikan alas ditopang
        if z == 0:
            return True  # Sudah di lantai

        # Periksa apakah alas ditopang oleh item lain
        supported = False
        for px in range(x, x + item.dx):
            for py in range(y, y + item.dy):
                supported_here = False
                for placed in container['placed']:
                    pi = placed['item']
                    bx, by, bz = placed['x'], placed['y'], placed['z']
                    if (bx <= px < bx + pi.dx and
                        by <= py < by + pi.dy and
                        bz + pi.dz == z):  # Harus pas tepat di bawah
                        supported_here = True
                        break
                if not supported_here:
                    return False  # Titik di bawah tidak ditopang
        return True


    def step(self):
        if not self.unplaced:
            return self._get_state(), 0, True, {}

        item = self.unplaced[0]
        best_fit = None
        best_score = float('inf')
        reward = 0
        info = {'placed': False, 'new_container': False}

        # Coba semua container yang sudah ada
        for container in self.containers:
            for orientation in item.orientations:
                temp_item = Item(*orientation, item.weight, item.name)

                # Urutkan ruang kosong (space) berdasarkan z, y, x
                sorted_free_spaces = sorted(container['free'], key=lambda s: (s[2], s[1], s[0]))

                for space in sorted_free_spaces:
                    # space = (sx, sy, sz, sw, sd, sh)
                    sx, sy, sz, sw, sd, sh = space
                    pos = (sx, sy, sz)

                    if self._can_place(container, temp_item, space) and \
                    (container['weight'] + temp_item.weight <= self.max_weight_per_container):

                        # Penalti tinggi seperti sebelumnya
                        height_penalty = pos[2] / self.height * 100

                        # Sisa ruang sebagai penalti dasar
                        space_left = (
                            self.width * self.depth * self.height -
                            (container['volume_used'] + temp_item.volume)
                        )

                        # Total skor
                        score = (space_left) + (height_penalty * 0.1)

                        if score < best_score:
                            best_score = score
                            best_fit = (container, temp_item, space, pos)

        if best_fit:
            container, temp_item, space, pos = best_fit
            self.unplaced.pop(0)
            container['placed'].append({
                'item': temp_item,
                'x': pos[0],
                'y': pos[1],
                'z': pos[2]
            })
            container['weight'] += temp_item.weight
            container['volume_used'] += temp_item.volume
            self.placed_items.append(temp_item)

            # Update ruang kosong
            self._update_free_spaces(container, space, pos, temp_item)

            # Hitung reward
            volume_utilization = temp_item.volume / (self.width * self.depth * self.height)
            height_penalty = pos[2] / self.height * 0.2
            reward = volume_utilization * (1 - height_penalty)

            info['placed'] = True
            return self._get_state(), reward, len(self.unplaced) == 0, info

        # Jika tidak bisa ditempatkan di kontainer manapun, buat kontainer baru
        if len(self.containers) < self.max_containers:
            if self._add_new_container():
                info['new_container'] = True
                return self.step()

        return self._get_state(), 0, len(self.unplaced) == 0, info


    def _can_place(self, container, item, space):
        # space sekarang formatnya: (x, y, z, w, d, h)
        sx, sy, sz, sw, sd, sh = space

        # Pastikan muat dalam ruang kosong ini
        if item.dx > sw or item.dy > sd or item.dz > sh:
            return False

        # Pastikan tidak keluar dari ukuran kontainer
        if sx + item.dx > self.width or sy + item.dy > self.depth or sz + item.dz > self.height:
            return False

        # Cek tabrakan dengan barang yang sudah ada
        for placed_item in container['placed']:
            pi = placed_item['item']
            px, py, pz = placed_item['x'], placed_item['y'], placed_item['z']

            if not (sx + item.dx <= px or sx >= px + pi.dx or
                    sy + item.dy <= py or sy >= py + pi.dy or
                    sz + item.dz <= pz or sz >= pz + pi.dz):
                return False

        return True


    def _update_free_spaces(self, container, space, pos, item):
        # Hapus ruang kosong lama
        container['free'].remove(space)

        sx, sy, sz, sw, sd, sh = space
        x, y, z = pos

        # Buat ruang di kanan
        if x + item.dx < sx + sw:
            container['free'].append((x + item.dx, sy, sz,
                                    (sx + sw) - (x + item.dx), sd, sh))

        # Buat ruang di depan
        if y + item.dy < sy + sd:
            container['free'].append((sx, y + item.dy, sz,
                                    sw, (sy + sd) - (y + item.dy), sh))

        # Buat ruang di atas
        if z + item.dz < sz + sh:
            container['free'].append((sx, sy, z + item.dz,
                                    sw, sd, (sz + sh) - (z + item.dz)))

        # Hapus ruang yang dimensinya nol atau negatif
        container['free'] = [s for s in container['free'] if s[3] > 0 and s[4] > 0 and s[5] > 0]
    def _is_space_available(self, container, space, item):
        sx, sy, sz, sw, sd, sh = space

        # Pastikan item muat di dalam blok ruang kosong
        if item.dx <= sw and item.dy <= sd and item.dz <= sh:
            # Cek apakah posisi ini tabrakan dengan barang lain
            for placed in container['placed']:
                px, py, pz = placed['x'], placed['y'], placed['z']
                pw, pd, ph = placed['item'].dx, placed['item'].dy, placed['item'].dz

                overlap_x = not (sx + item.dx <= px or sx >= px + pw)
                overlap_y = not (sy + item.dy <= py or sy >= py + pd)
                overlap_z = not (sz + item.dz <= pz or sz >= pz + ph)

                if overlap_x and overlap_y and overlap_z:
                    return False  # Ada tabrakan
            return True

        return False  # Tidak muat secara dimensi


    def _remove_redundant_spaces(self, container):
        to_remove = set()
        free_spaces = container['free']
        for i, space1 in enumerate(free_spaces):
            for j, space2 in enumerate(free_spaces):
                if i != j and (space1[0] >= space2[0] and
                               space1[1] >= space2[1] and
                               space1[2] >= space2[2]):
                    to_remove.add(i)
        for i in sorted(to_remove, reverse=True):
            if i < len(free_spaces):
                free_spaces.pop(i)

def draw_container_outline(ax, origin, width, depth, height):
    x0, y0, z0 = origin
    x1, y1, z1 = x0 + width, y0 + depth, z0 + height

    lines = [
        [(x0, y0, z0), (x1, y0, z0)],
        [(x1, y0, z0), (x1, y1, z0)],
        [(x1, y1, z0), (x0, y1, z0)],
        [(x0, y1, z0), (x0, y0, z0)],

        [(x0, y0, z1), (x1, y0, z1)],
        [(x1, y0, z1), (x1, y1, z1)],
        [(x1, y1, z1), (x0, y1, z1)],
        [(x0, y1, z1), (x0, y0, z1)],

        [(x0, y0, z0), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y0, z1)],
        [(x1, y1, z0), (x1, y1, z1)],
        [(x0, y1, z0), (x0, y1, z1)],
    ]

    for line in lines:
        xs, ys, zs = zip(*line)
        ax.plot(xs, ys, zs, color='black', linewidth=1)

def visualize_all_containers(env):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')
    offset_x = 0

    # Mapping warna berdasarkan nama item
    item_colors = {}
    legend_elements = []

    def get_color_for_item(name):
        if name not in item_colors:
            color = [random.uniform(0.3, 0.9) for _ in range(3)]
            item_colors[name] = color
            legend_elements.append(Patch(facecolor=color, edgecolor='k', label=name))
        return item_colors[name]

    for container_idx, container in enumerate(env.containers):
        utilization = (container['volume_used'] / (env.width * env.depth * env.height)) * 100
        num_items = len(container['placed'])
        max_weight = env.max_weight_per_container
        total_weight = container['weight']

        # Gambar outline container
        draw_container_outline(ax, origin=(offset_x, 0, 0), width=env.width, depth=env.depth, height=env.height)

        ax.text(offset_x + env.width/2, -env.depth*0.2, env.height*0.5,
            f"Container {container_idx+1}\n"
            f"Utilization: {utilization:.1f}%\n"
            f"Items: {num_items}\n"
            f"Weight: {total_weight:.1f} / {max_weight} kg",
            color='blue', ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

        for placement in container['placed']:
            item = placement['item']
            x, y, z = placement['x'], placement['y'], placement['z']
            x += offset_x

            vertices = [
                [x, y, z],
                [x + item.dx, y, z],
                [x + item.dx, y + item.dy, z],
                [x, y + item.dy, z],
                [x, y, z + item.dz],
                [x + item.dx, y, z + item.dz],
                [x + item.dx, y + item.dy, z + item.dz],
                [x, y + item.dy, z + item.dz]
            ]

            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],
                [vertices[4], vertices[5], vertices[6], vertices[7]],
                [vertices[0], vertices[1], vertices[5], vertices[4]],
                [vertices[2], vertices[3], vertices[7], vertices[6]],
                [vertices[1], vertices[2], vertices[6], vertices[5]],
                [vertices[0], vertices[3], vertices[7], vertices[4]]
            ]

            color = get_color_for_item(item.name)
            poly = Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=0.8)
            ax.add_collection3d(poly)
            ax.text(x + item.dx/2, y + item.dy/2, z + item.dz/2, 
                    f"{item.name}\n{item.dx}x{item.dy}x{item.dz}cm", 
                    color='black', ha='center', va='center', fontsize=6)

        offset_x += env.width + 1

    # Set skala dan aspek
    ax.set_xlim(0, offset_x)
    ax.set_ylim(-env.depth*0.3, env.depth*1.1)
    ax.set_zlim(0, env.height*1.1)
    ax.set_box_aspect([offset_x, env.depth*1.4, env.height*1.1])

    ax.set_xlabel('Width (cm)')
    ax.set_ylabel('Depth (cm)')
    ax.set_zlabel('Height (cm)')

    # Total stats
    total_utilization = sum(c['volume_used'] for c in env.containers) / \
                        (env.width * env.depth * env.height * len(env.containers)) * 100
    total_items = sum(len(c['placed']) for c in env.containers)

    ax.set_title(f'Container Packing Visualization\n'
                 f'Total Containers: {len(env.containers)} | '
                 f'Avg Utilization: {total_utilization:.1f}% | '
                 f'Total Items: {total_items}', pad=20)

    # Tambahkan legenda
    ax.legend(handles=legend_elements, title="Item Legend", loc='upper right', bbox_to_anchor=(1.15, 1.0))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ukuran container dalam cm (mengikuti palet standar)
    D = 243.8  # Lebar palet standar
    L = 609.6  # Panjang palet standar
    H = 259.0  # Tinggi maksimum (dalam cm, ~2.6m)
    
    items = [
        *[Item(15, 30, 20, 6, "Box1") for _ in range(200)],
        *[Item(150, 30, 30, 10, "Box2") for _ in range(150)],
        *[Item(40, 40, 100, 7, "Box3") for _ in range(100)]
    ]
    
    # Filter item yang tidak muat dalam container
    valid_items = []
    for item in items:
        fits = False
        for orient in item.orientations:
            if orient[0] <= D and orient[1] <= L and orient[2] <= H:
                fits = True
                break
        if fits:
            valid_items.append(item)
        else:
            print(f"Item {item.name} dengan dimensi {item.dx}x{item.dy}x{item.dz} tidak muat dalam container dalam orientasi apapun")
    
    env = ContainerPackingEnv(container_size=(D, L, H), items=valid_items, max_containers=10, max_weight_per_container=24000)

    done = False
    while not done:
        state, reward, done, info = env.step()
        if len(env.unplaced) > 0 and len(env.containers) >= env.max_containers:
            print("Peringatan: Tidak semua item bisa dimasukkan karena keterbatasan container")
            break

    print(f"\nHasil Packing:")
    print(f"Total container digunakan: {len(env.containers)}")
    print(f"Item yang berhasil ditempatkan: {len(env.placed_items)}")
    print(f"Item yang tidak terpasang: {len(env.unplaced)}")
    
    if len(env.unplaced) > 0:
        print("\nItem yang tidak terpasang:")
        for item in env.unplaced:
            print(f"- {item.name} (Dimensi: {item.dx}x{item.dy}x{item.dz})")

    # Hitung utilitas setiap container
    print("\nUtilisasi Container:")
    for i, container in enumerate(env.containers, 1):
        utilization = container['volume_used'] / (env.width * env.depth * env.height) * 100
        print(f"Container {i}: {utilization:.2f}% terisi ({container['volume_used']}/{env.width*env.depth*env.height} cm³)")

    visualize_all_containers(env)