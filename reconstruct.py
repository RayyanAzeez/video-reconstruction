import os, cv2, time
import numpy as np
from PIL import Image
from tqdm import tqdm
import imagehash
from skimage.metrics import structural_similarity as ssim

########################
# 1) Frame extraction
########################
def extract_frames(video_path, out_dir="frames"):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(f"{out_dir}/{idx:04d}.jpg", frame)
        idx += 1
    cap.release()
    return idx

########################
# 2) pHash coarse order
########################
def phash_order(frames_dir="frames"):
    files = sorted(os.listdir(frames_dir))
    hashes = []
    for f in tqdm(files, desc="phash"):
        h = imagehash.phash(Image.open(os.path.join(frames_dir,f)))
        # store integer for stable sorting
        hashes.append((f, int(str(h), 16)))
    hashes.sort(key=lambda x: x[1])
    ordered = [x[0] for x in hashes]
    return ordered

########################
# Utilities: load small grayscale arrays
########################
def load_small_gray(fname, size=(128,128)):
    img = Image.open(fname).convert("L")
    img = img.resize(size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr

########################
# 3) Build SSIM similarity matrix (on a sequence subset)
########################
def build_ssim_matrix(filenames, frames_dir="frames", size=(128,128)):
    n = len(filenames)
    imgs = [load_small_gray(os.path.join(frames_dir,f), size=size) for f in filenames]
    S = np.zeros((n,n), dtype=np.float32)
    for i in tqdm(range(n), desc="SSIM matrix"):
        for j in range(i, n):
            if i == j:
                val = 1.0
            else:
                try:
                    val = ssim(imgs[i], imgs[j], data_range=1.0)
                except Exception:
                    val = np.corrcoef(imgs[i].ravel(), imgs[j].ravel())[0,1]
                    if np.isnan(val): val = 0.0
                    val = float(max(min(val,1.0), -1.0))
                if val < 0: val = 0.0
            S[i,j] = val
            S[j,i] = val
    return S

########################
# 4) Greedy nearest neighbor on distance matrix
########################
def greedy_nn_from(dist, start_idx):
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    order = [start_idx]
    visited[start_idx] = True
    cur = start_idx
    for _ in range(n-1):
        row = dist[cur].copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        order.append(nxt)
        visited[nxt] = True
        cur = nxt
    return order

########################
# 5) 2-opt improvement
########################
def two_opt_path(order, dist):
    n = len(order)
    improved = True
    it = 0
    while improved and it < 500:
        improved = False
        it += 1
        for i in range(0, n-2):
            a, b = order[i], order[i+1]
            for j in range(i+2, n):
                c, d = order[j% n], order[(j+1) % n] if j+1 < n else None
                if j+1 >= n:
                    continue
                cur_cost = dist[a,b] + dist[c,d]
                new_cost = dist[a,c] + dist[b,d]
                if new_cost + 1e-12 < cur_cost:
                    order[i+1:j+1] = reversed(order[i+1:j+1])
                    improved = True
    return order

########################
# 6) Choose start frame
########################
def choose_start(dist):
    row_sums = dist.sum(axis=1)
    start = int(np.argmin(row_sums))
    return start

########################
# 7) Full pipeline tying pHash + SSIM + greedy + 2-opt + orientation check
########################
def reconstruct(frames_dir="frames", out_video="reconstructed_fixed.mp4", ssim_size=(128,128), chunk_size=None):
    files = sorted(os.listdir(frames_dir))
    n = len(files)
    print(f"Found {n} frames")
    coarse = phash_order(frames_dir)
    print("Coarse pHash order computed.")
    if chunk_size is None or chunk_size >= n:
        seq = coarse
        print("Building full SSIM matrix (this may take time for large n).")
        S = build_ssim_matrix(seq, frames_dir=frames_dir, size=ssim_size)
        D = 1.0 - S
        start = choose_start(D)
        order_idx = greedy_nn_from(D, start)
        order_idx = two_opt_path(order_idx, D)
        ordered_files = [seq[i] for i in order_idx]
    else:
        ordered_files = []
        i = 0
        while i < n:
            chunk = coarse[i: min(n, i + chunk_size)]
            S = build_ssim_matrix(chunk, frames_dir=frames_dir, size=ssim_size)
            D = 1.0 - S
            start = choose_start(D)
            order_idx = greedy_nn_from(D, start)
            order_idx = two_opt_path(order_idx, D)
            chunk_ordered = [chunk[k] for k in order_idx]
            ordered_files.extend(chunk_ordered if i==0 else chunk_ordered[5:])
            i += chunk_size - 5
        print("Final global refine across stitched sequence.")
        Sg = build_ssim_matrix(ordered_files, frames_dir=frames_dir, size=ssim_size)
        Dg = 1.0 - Sg
        start = choose_start(Dg)
        order_idx = greedy_nn_from(Dg, start)
        order_idx = two_opt_path(order_idx, Dg)
        ordered_files = [ordered_files[i] for i in order_idx]

    ########################
    # orientation check
    ########################

    def enforce_correct_direction(order_files, frames_dir="frames", ssim_size=(128,128)):
        def continuity_score(seq):
            score = 0
            for a,b in zip(seq, seq[1:]):
                A = load_small_gray(os.path.join(frames_dir, a), size=ssim_size)
                B = load_small_gray(os.path.join(frames_dir, b), size=ssim_size)
                score += np.sum(np.abs(B - A))
            return score

        forward_score  = continuity_score(order_files)
        reverse_score  = continuity_score(list(reversed(order_files)))

        if reverse_score > forward_score:
            print("Reversed detected — flipping order.")
            return list(reversed(order_files))
        else:
            print("Correct orientation detected.")
            return order_files
    ordered_files = enforce_correct_direction(ordered_files, frames_dir=frames_dir, ssim_size=ssim_size)
    
    first = cv2.imread(os.path.join(frames_dir, ordered_files[0]))
    h, w, _ = first.shape
    fps = 30
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    for fname in tqdm(ordered_files, desc="Writing video"):
        frame = cv2.imread(os.path.join(frames_dir, fname))
        writer.write(frame)
    writer.release()
    return out_video, ordered_files
    

########################
# 8) Run
########################
if __name__ == "__main__":
    start_t = time.time()
    video = "jumbled_video.mp4"
    print("Extracting frames...")
    n = extract_frames(video, out_dir="frames")
    print(f"{n} frames extracted.")
    out, ordering = reconstruct(frames_dir="frames", out_video="reconstructed_output.mp4", ssim_size=(128,128), chunk_size=None)

    print("Done. Output:", out)
    print("Elapsed (s):", time.time() - start_t)
