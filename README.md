# Visual Navigation Game (Example Player Code)

This is the course project platform for NYU ROB-GY 6203 Robot Perception. 
For more information, please reach out to AI4CE lab (cfeng at nyu dot edu).

# Instructions for Players
1. Install
```commandline
conda update conda
git clone https://github.com/ai4ce/vis_nav_player.git
cd vis_nav_player
conda env create -f environment.yaml
conda activate game
```

2. Play using the default keyboard player
```commandline
python source/player.py
```

3. Modify the player.py to implement your own solutions, 
unless you have photographic memories!

# Baseline Solution
## How to run the baseline
1. Download the exploration data and extract it to `./data`. Under your data folder, you should at least have:
   ```
   data
   ├── data_info.json
   ├── images
   ```
2. Run the baseline solution by `python source/baseline.py`. The first run may take longer as we need to download data for the maze and computes the features for localization and navigation.
3. Press `q` to show the navigation panel.

## How the baseline works
The baseline (`source/baseline.py`) implements a visual place recognition pipeline:

1. **Feature Extraction** — RootSIFT descriptors from exploration images
2. **Codebook** — K-Means clustering (k=128) to build a visual vocabulary
3. **VLAD Encoding** — Aggregate local descriptors into a global vector per image (with intra-normalization and power normalization)
4. **Graph Construction** — Temporal edges (consecutive frames) + visual shortcut edges (top-K most similar non-adjacent frames)
5. **Localization & Planning** — Match current FPV to database via VLAD similarity, then Dijkstra shortest path to goal node