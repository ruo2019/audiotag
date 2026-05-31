# --- Configuration ---
AUDIO_FILE = 'Tau.mp3'  # <<< CHANGE THIS TO YOUR AUDIO FILE (can be .wav or .mp3)
TARGET_SR = 11025              # Target sample rate, best to choose from [8000, 11025, 16000, 22050] (lower = less data, 16k is probably the best for its size, but still can reach the 5MB graph size limit, you can use 10000 or 9000 too.)
N_PARTIALS = 30                # Sine waves per frame (higher = more detail, more accurate, a bit more noisy)
GAIN_DIGITS=2                  # Number of decimal places for gain values (higher = more precise, but bigger graph size)
N_FFT = 1024                   # FFT window size (advanced, don't change unless you know what you're doing) Default: 1024
HOP_LENGTH = 256               # Samples between frames (also advanced)
MAX_LIST_SIZE = 9999           # Max items per list (DO NOT CHANGE)
GAIN_SCALING_POWER = 1.0       # Compress gain dynamics (0.5 is sqrt) (turn up to 0.7-1.0 if the resulting graph is not clear)

# --- Advanced Tuning ---
PEAK_MIN_HEIGHT_RATIO = 0.01   # Discard quiet peaks
PEAK_MIN_DISTANCE_HZ = 24      # Min Hz between peaks

# --- Helper Functions --- (Same as before)
def find_n_strongest_peaks(data, n, min_height, min_distance):
    peaks_indices, _ = scipy.signal.find_peaks(data, height=min_height, distance=min_distance)
    sorted_peaks = sorted(peaks_indices, key=lambda i: data[i], reverse=True)[:n]
    return sorted_peaks, [data[i] for i in sorted_peaks]

def analyze_audio(filename, target_sr, n_fft, hop_length, n_partials):
    y, sr = librosa.load(filename, sr=target_sr, mono=True)
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    freqs = librosa.fft_frequencies(sr=target_sr, n_fft=n_fft)

    num_frames = magnitude.shape[1]
    all_freqs = []
    all_gains = []
    max_gain = 0.0

    for i in range(num_frames):
        frame_mag = magnitude[:, i]
        min_h = np.max(frame_mag) * PEAK_MIN_HEIGHT_RATIO if np.max(frame_mag) > 0 else 0
        min_dist_bins = int(PEAK_MIN_DISTANCE_HZ * n_fft / target_sr)

        idx, mag = find_n_strongest_peaks(frame_mag, n_partials, min_h, max(1, min_dist_bins))
        frame_freq = [freqs[i] for i in idx]
        frame_gain = mag

        # Pad with zeros if fewer than N_PARTIALS peaks
        if len(frame_freq) < n_partials:
            frame_freq += [0.0] * (n_partials - len(frame_freq))
            frame_gain += [0.0] * (n_partials - len(frame_gain))

        all_freqs.append(frame_freq)
        all_gains.append(frame_gain)

        if frame_gain:
            max_gain = max(max_gain, max(frame_gain))

    # Normalize gains (0 to 1)
    if max_gain > 1e-6:
        all_gains = [[(g / max_gain) ** GAIN_SCALING_POWER for g in frame] for frame in all_gains]
    else:
        all_gains = [[0.0 for _ in range(n_partials)] for _ in all_gains]

    return all_freqs, all_gains, target_sr

# --- NEW: Transpose Data for F(x)/G(x) Access ---
def transpose_chunks(all_freqs, all_gains, n_partials, max_list_size):
    # Flatten all frames into sequential lists (like original chunks)
    flat_freqs = [f for frame in all_freqs for f in frame]
    flat_gains = [g for frame in all_gains for g in frame]

    num_total_items = len(flat_freqs)
    num_possible_functions = math.floor(num_total_items / max_list_size)

    # Split into sublists where F(1) = first max_list_size items, F(2) = next, etc.
    # BUT: We actually need to group by *position* across chunks.
    # So instead, we'll generate F(1) as [frame1[1], frame2[1], frame3[1], ...]
    # up to max_list_size items per F(i). Same for G(i).
    # This requires iterating through all frames and picking the i-th element each time.

    # Determine how many F(i)/G(i) functions we'll need
    frames_per_F = math.floor(max_list_size)
    total_frames = len(all_freqs)
    needed_F_functions = math.ceil(total_frames / frames_per_F) * n_partials

    print(f"\nTransposing data for F(i)/G(i) access:")
    print(f"  Total frames: {len(all_freqs)}")
    print(f"  Partials per frame (N): {n_partials}")
    print(f"  Max frames per F(i)/G(i): {frames_per_F}")
    print(f"  Needed F(i)/G(i) functions: ~{needed_F_functions}")

    # Initialize lists-of-lists for F and G outputs
    F_output = [[] for _ in range(n_partials)]
    G_output = [[] for _ in range(n_partials)]

    # For each frame, distribute its N partials into the F_output and G_output
    for frame_idx in range(len(all_freqs)):
        for partial_idx in range(n_partials):
            F_output[partial_idx].append(all_freqs[frame_idx][partial_idx])
            G_output[partial_idx].append(all_gains[frame_idx][partial_idx])

    # Now split each F_output[partial_idx] and G_output[partial_idx] into chunks
    # of max_list_size to stay under calculator limits
    F_final = []
    G_final = []

    for partial_idx in range(n_partials):
        partial_freqs = F_output[partial_idx]
        partial_gains = G_output[partial_idx]

        num_chunks_for_partial = math.ceil(len(partial_freqs) / max_list_size)

        for chunk_idx in range(num_chunks_for_partial):
            start = chunk_idx * max_list_size
            end = start + max_list_size

            freq_chunk = partial_freqs[start:end]
            gain_chunk = partial_gains[start:end]

            F_final.append(freq_chunk)
            G_final.append(gain_chunk)

    return F_final, G_final

# --- Save or Print Transposed Data ---
def save_transposed_data(F_data, G_data, prefix):
    print("\nSaving transposed data to files...")

import json
import math

def print_transposed_data(F_data, G_data):
    # Assume N_PARTIALS, MAX_LIST_SIZE, GAIN_DIGITS, AUDIO_FILE are accessible from global scope
    # Or pass them as arguments if needed

    if not F_data or not G_data:
        print("Error: Input data is empty.")
        return

    n_partials = N_PARTIALS
    max_list_size = MAX_LIST_SIZE

    # --- Calculate Chunk Info ---
    # F_data is structured as [P0_C0, P0_C1, ..., P1_C0, P1_C1, ...]
    if n_partials == 0:
        print("Error: N_PARTIALS is zero.")
        return
    num_chunks_per_partial = len(F_data) // n_partials
    if num_chunks_per_partial == 0:
         print("Warning: No full chunks found. Data might be incomplete or N_PARTIALS too high.")
         # Attempt to infer total_frames from the first partial's data if it exists
         if F_data:
             total_frames = len(F_data[0])
             num_chunks_per_partial = 1 # Assume at least one chunk if data exists
         else:
             total_frames = 0
    else:
        # Calculate total_frames based on the length of the chunks for the first partial
        len_last_chunk_p0 = len(F_data[num_chunks_per_partial - 1])
        total_frames = (num_chunks_per_partial - 1) * max_list_size + len_last_chunk_p0

    if total_frames == 0:
        print("Error: Calculated total frames is zero.")
        return

    print(f"\nGenerating Desmos State:")
    print(f"  Total frames: {total_frames}")
    print(f"  Partials per frame (N): {n_partials}")
    print(f"  Max list size (L): {max_list_size}")
    print(f"  Number of chunks: {num_chunks_per_partial}")

    # --- Gain Formatting ---
    def get_str(gain, digits=GAIN_DIGITS):
        max_val_str = str(10**digits)
        scaled_gain = gain * (10**digits)
        # Handle potential floating point inaccuracies near zero
        if scaled_gain < 0.5: # Threshold for rounding down to 0
             return "0"
        rounded_gain = round(scaled_gain)
        if rounded_gain >= (10**digits):
            return max_val_str
        else:
            # Ensure output is integer string
            return str(int(rounded_gain))

    gain_scale_factor_str = f"0.{'0'*(GAIN_DIGITS-1)}1" # e.g., "0.01" for GAIN_DIGITS=2

    # --- Generate F/G Strings and Expressions ---
    expressions_list = []
    expressions_list.append({"type": "folder", "id": "2", "title": "Frequencies and Amplitudes (do not open)", "collapsed": True})
    id_counter = 3 # Start IDs after the folder

    for k in range(1, num_chunks_per_partial + 1):
        func_suffix = "" if k == 1 else f"_{k}"
        F_k_lists_str = []
        G_k_lists_str = []

        # Ensure chunk length is correct, especially for the last chunk
        current_chunk_length = max_list_size
        if k == num_chunks_per_partial:
            current_chunk_length = total_frames - (k - 1) * max_list_size

        for p in range(n_partials):
            data_index = p * num_chunks_per_partial + (k - 1)
            # Slice data to ensure correct length for the last chunk
            freq_chunk_data = F_data[data_index][:current_chunk_length]
            gain_chunk_data = G_data[data_index][:current_chunk_length]
            freq_chunk_str = "[" + ",".join([f"{round(x)}" for x in freq_chunk_data]) + "][i]"
            gain_chunk_str = "[" + ",".join([get_str(x, digits=GAIN_DIGITS) for x in gain_chunk_data]) + "][i]"

            F_k_lists_str.append(freq_chunk_str)
            G_k_lists_str.append(gain_chunk_str)

        # Note: Desmos `[L1, L2][i]` accesses `L1[i]` and `L2[i]`. Index `i` must be valid for inner lists.
        f_string_k = f"F{func_suffix}(i) = [" + ",".join(F_k_lists_str) + "]"
        g_string_k = f"G{func_suffix}(i) = [" + ",".join(G_k_lists_str) + "]"

        expressions_list.append({"type": "expression", "id": str(id_counter), "folderId": "2", "color": "#2d70b3", "latex": f_string_k, "hidden": True})
        id_counter += 1
        expressions_list.append({"type": "expression", "id": str(id_counter), "folderId": "2", "color": "#388c46", "latex": g_string_k, "hidden": True})
        id_counter += 1

    # --- Generate Tone Expressions ---
    for k in range(1, num_chunks_per_partial + 1):
        func_suffix = "" if k == 1 else f"_{k}"
        lower_bound_py = (k - 1) * max_list_size  # 0-based start frame index for this chunk
        upper_bound_py = k * max_list_size        # 0-based end frame index (exclusive) for this chunk

        # Adjust for the last chunk
        if k == num_chunks_per_partial:
            upper_bound_py = total_frames

        # Desmos uses 1-based indexing for slider `I`
        desmos_lower_bound = lower_bound_py + 1
        desmos_upper_bound = upper_bound_py # If total_frames is 100, max index is 100

        # Condition string for Desmos
        if num_chunks_per_partial == 1:
             # If only one chunk, no need for index bounds, covers I = 1 to total_frames
             condition = f"\\left\\{{d_{{i}}=1\\right\\}}"
        elif k == 1:
             # First chunk: 1 <= I <= max_list_size
             condition = f"\\left\\{{{desmos_lower_bound} \\le I \\le {desmos_upper_bound}, d_{{i}}=1\\right\\}}"
        else:
             # Subsequent chunks: lower_bound_py < I <= upper_bound_py
             # Need strict inequality for lower bound because previous chunk included its upper bound
             condition = f"\\left\\{{{lower_bound_py} < I \\le {desmos_upper_bound}, d_{{i}}=1\\right\\}}"
# "tone \left(F_2(I-9999),0.01G_2(I-9999)\right)\left{9999 < I \le 13351, d_{i}=1\right}"

        # Index for F_k/G_k should be 1-based within the chunk: I - lower_bound_py
        # '\\operatorname{tone}\\left(F\\left(i\\right),G\\left(i\\right)\\right)\\left\\{d_{i}=1\\right\\}'
        tone_latex = "\\operatorname{tone}\\left(F{func_suffix}(I-{lower_bound_py}),{gain_scale_factor_str}G{func_suffix}(I-{lower_bound_py})\\right){condition}"

        tone_latex = tone_latex.replace("{func_suffix}", func_suffix)
        tone_latex = tone_latex.replace("{lower_bound_py}", str(lower_bound_py))
        tone_latex = tone_latex.replace("{gain_scale_factor_str}", gain_scale_factor_str)
        tone_latex = tone_latex.replace("{func_suffix}", func_suffix)
        tone_latex = tone_latex.replace("{condition}", condition)

        expressions_list.append({"type": "expression", "id": str(id_counter), "color": "#6042a6", "latex": tone_latex})
        id_counter += 1

    # --- Add Remaining Expressions ---
    var_folder_id = str(id_counter)
    expressions_list.append({"type": "folder", "id": var_folder_id, "title": "Variables", "collapsed": True})
    id_counter += 1

    expressions_list.append({"type": "expression", "id": str(id_counter), "folderId": var_folder_id, "color": "#000000", "latex": "I=1", "hidden": True, "slider": {"hardMin": True, "hardMax": True, "min": "1", "max": str(total_frames)}})
    slider_id = str(id_counter) # Save slider ID if needed later, though seems not
    id_counter += 1

    expressions_list.append({"type": "expression", "id": str(id_counter), "folderId": var_folder_id, "color": "#388c46", "latex": "d_{i}=1", "hidden": True})
    di_id = str(id_counter) # Save d_i ID
    id_counter += 1

    color_folder_id = str(id_counter)
    expressions_list.append({"type": "folder", "id": color_folder_id, "title": "Colors", "collapsed": True})
    id_counter += 1

    expressions_list.append({"type": "expression", "id": str(id_counter), "folderId": color_folder_id, "color": "#6042a6", "latex": f"C_{{middle}}=\\left\\{{d_{{i}}=1:\\operatorname{{rgb}}\\left(44,133,67\\right),\\operatorname{{rgb}}\\left(176,58,58\\right)\\right\\}}"})
    id_counter += 1
    expressions_list.append({"type": "expression", "id": str(id_counter), "folderId": color_folder_id, "color": "#000000", "latex": "C_{bar}=\\operatorname{rgb}\\left(90,90,90\\right)"})
    id_counter += 1

    display_folder_id = str(id_counter)
    expressions_list.append({"type": "folder", "id": display_folder_id, "title": "Display", "collapsed": True})
    id_counter += 1

    # Play/Pause button
    expressions_list.append({
        "type": "expression", "id": str(id_counter), "folderId": display_folder_id, "color": "#c74440",
        "latex": f"\\left(x-1500\\right)^{{2}}+\\left(y+920\\right)^{{2}}\\le300^{{2}}",
        "colorLatex": "C_{middle}",
        "clickableInfo": {
            "enabled": True,
            # Reset I to 1 on click if I reaches the end
            "latex": f"\\left\\{{I={total_frames}:I\\to 1\\right\\}},d_{{i}}\\to1-d_{{i}}"
        }
    })
    id_counter += 1

    # Pause bars
    expressions_list.append({"type": "expression", "id": str(id_counter), "folderId": display_folder_id, "color": "#000000", "latex": f"\\operatorname{{polygon}}\\left(\\left(1430,-1040\\right),\\left(1430,-800\\right)\\right)\\left\\{{d_{{i}}=1\\right\\}}", "fill": False, "lineWidth": "5"})
    id_counter += 1
    expressions_list.append({"type": "expression", "id": str(id_counter), "folderId": display_folder_id, "color": "#000000", "latex": f"\\operatorname{{polygon}}\\left(\\left(1570,-1040\\right),\\left(1570,-800\\right)\\right)\\left\\{{d_{{i}}=1\\right\\}}", "fill": False, "lineWidth": "5"})
    id_counter += 1
    # Play triangle
    expressions_list.append({"type": "expression", "id": str(id_counter), "folderId": display_folder_id, "color": "#000000", "latex": f"\\operatorname{{polygon}}\\left(\\left(1410,-1040\\right),\\left(1650,-920\\right),\\left(1410,-800\\right)\\right)\\left\\{{d_{{i}}=0\\right\\}}", "fill": True, "fillOpacity": "1"})
    id_counter += 1
    # Progress bar base
    expressions_list.append({"type": "expression", "id": str(id_counter), "folderId": display_folder_id, "color": "#000000", "latex": "\\operatorname{polygon}\\left(\\left(0,-420\\right),\\left(3000,-420\\right)\\right)", "fill": False, "colorLatex": "C_{bar}", "lineWidth": "10"})
    id_counter += 1
    # Progress bar indicator
    # Scale I (from 1..total_frames) to 0..3000
    expressions_list.append({
        "type": "expression", "id": str(id_counter), "folderId": display_folder_id, "color": "#fa7e19",
        "latex": f"\\left(\\left(I-1\\right)\\cdot\\frac{{3000}}{{{total_frames-1 if total_frames > 1 else 1}}},{{-420}}\\right)", # Handle division by zero if total_frames=1
        "pointSize": "10", "movablePointSize": "10"
    })
    id_counter += 1
    # Title Label
    audio_file_name_no_ext = AUDIO_FILE.split('.')[0] if '.' in AUDIO_FILE else AUDIO_FILE
    expressions_list.append({
        "type": "expression", "id": str(id_counter), "folderId": display_folder_id, "color": "#000000",
        "latex": "\\left(1500,-210\\right)", "showLabel": True, "label": f"{audio_file_name_no_ext}",
        "hidden": True, # Keep label visible
        "labelSize": "1.4", "labelOrientation": "center"
    })
    id_counter += 1

    # Empty expression for spacing/placeholder?
    expressions_list.append({"type": "expression", "id": str(id_counter), "folderId": display_folder_id, "color": "#388c46"})
    id_counter += 1

    # Frequency display polygon - using F(I-0) - only shows first chunk visually
    # expressions_list.append({"type": "expression", "id": str(id_counter), "folderId": display_folder_id, "color": "#388c46", "latex": f"\\operatorname{{polygon}}\\left(\\left(F\\left(I\\right),0\\right),\\left(F\\left(I\\right),20*G\\left(I\\right)*{gain_scale_factor_str}\\right)\\right)\\left\\{{1 \\le I \\le {max_list_size}, d_i=1\\right\\}}"}) # Restrict to first chunk

    for k in range(1, num_chunks_per_partial + 1):
        print(f"Adding chunk {k}")
        func_suffix = "" if k == 1 else f"_{k}"
        lower_bound_py = (k - 1) * max_list_size  # 0-based start frame index for this chunk
        upper_bound_py = k * max_list_size        # 0-based end frame index (exclusive) for this chunk

        # Adjust for the last chunk
        if k == num_chunks_per_partial:
            upper_bound_py = total_frames

        # Desmos uses 1-based indexing for slider `I`
        desmos_lower_bound = lower_bound_py + 1
        desmos_upper_bound = upper_bound_py # If total_frames is 100, max index is 100

        # Create frequency display polygon for each chunk

        # Condition string for Desmos
        if num_chunks_per_partial == 1:
             # If only one chunk, no need for index bounds, covers I = 1 to total_frames
             condition = f"\\left\\{{d_{{i}}=1\\right\\}}"
             freq_poly_latex = f"\\operatorname{{polygon}}\\left(\\left(F{func_suffix}\\left(I\\right),0\\right),\\left(F{func_suffix}\\left(I\\right),20 G{func_suffix}\\left(I\\right)\\right)\\right){condition}"
        elif k == 1:
             # First chunk: 1 <= I <= max_list_size
             condition = f"\\left\\{{{desmos_lower_bound} \\le I \\le {desmos_upper_bound}, d_{{i}}=1\\right\\}}"
             freq_poly_latex = f"\\operatorname{{polygon}}\\left(\\left(F{func_suffix}\\left(I\\right),0\\right),\\left(F{func_suffix}\\left(I\\right),20 G{func_suffix}\\left(I\\right)\\right)\\right){condition}"
        else:
             # Subsequent chunks: lower_bound_py < I <= upper_bound_py
             # Need strict inequality for lower bound because previous chunk included its upper bound
             condition = f"\\left\\{{{lower_bound_py} < I \\le {desmos_upper_bound}, d_{{i}}=1\\right\\}}"
             freq_poly_latex = f"\\operatorname{{polygon}}\\left(\\left(F{func_suffix}\\left(I-{k-1}0000\\right),0\\right),\\left(F{func_suffix}\\left(I-{k-1}0000\\right),20 G{func_suffix}\\left(I-{k-1}0000\\right)\\right)\\right){condition}"

        expressions_list.append({"type": "expression", "id": str(id_counter), "folderId": display_folder_id, "color": "#388c46", "latex": freq_poly_latex})
        id_counter += 1

    id_counter += 1

    # Empty expression for spacing/placeholder?
    expressions_list.append({"type": "expression", "id": str(id_counter), "color": "#c74440"})
    id_counter += 1

    # --- Create Graph State Dictionary ---
    state_dict = {
        "version": 11,
        "randomSeed": "b1e7f5519720974b3160f63ec0f72c88", # Consider making this dynamic if needed
        "graph": {
            "viewport": { # Keep viewport as before, user can adjust
                "xmin": -568.4082546987474, "ymin": -3541.228726318303,
                "xmax": 4408.543369051632, "ymax": 3892.699015486061
            },
            "showGrid": False, "showYAxis": False, "xAxisNumbers": False,
            "yAxisNumbers": False, "polarNumbers": False, "userLockedViewport": True
        },
        "expressions": {"list": expressions_list,
            "ticker": {
                # Update I based on d_i, stop at total_frames. dt is typically ~30ms
                # Ensure I stays within [1, total_frames]
                "handlerLatex": f"I \\to \\min({total_frames}, \\max(1, I + d_{{i}} * \\operatorname{{dt}} / {(HOP_LENGTH / sr) * 1000:.2f} )) , d_{{i}} \\to \\left\\{{ I = {total_frames} : 0, d_{{i}} \\right\\}}",
                "minStepLatex": "0", # Ticker step is dt
                "open": True, "playing": True # Start playing automatically
            }},
        "includeFunctionParametersInRandomSeed": True,
        "doNotMigrateMovablePointStyle": True
    }

    # --- Final Output ---
    # Convert dictionary to JSON string, ensure separators have no spaces for compactness
    graph_state_json = json.dumps(state_dict, separators=(',', ':'))
    graph_command = f"Calc.setState({graph_state_json})"

    try:
        pyperclip.copy(graph_command)
        with open('GRAPH_COMMAND.txt', 'w') as f:
            f.write(graph_command)
        print("\nSuccessfully wrote Desmos state to [ GRAPH_COMMAND.txt ]")
    except IOError as e:
        print(f"\nError writing to GRAPH_COMMAND.txt: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    import subprocess
    import sys

    def install_package(package):
        try:
            __import__(package)
            print(f"{package} is already installed.")
        except ImportError:
            print(f"{package} not found. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"{package} installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error installing {package}: {e}")
                sys.exit(1)

    install_package("librosa")
    install_package("pyperclip")
    install_package("numpy")
    install_package("scipy")

    import librosa
    import pyperclip
    import numpy as np
    import math
    import scipy.signal
    print("Analyzing audio...")
    all_freqs, all_gains, sr = analyze_audio(
        AUDIO_FILE, TARGET_SR, N_FFT, HOP_LENGTH, N_PARTIALS
    )

    if all_freqs:
        print("Transposing data for F(i)/G(i) access...")
        F_data, G_data = transpose_chunks(all_freqs, all_gains, N_PARTIALS, MAX_LIST_SIZE)

        print_transposed_data(F_data, G_data)

        print("\n--- Playback Setup ---")
        print(f"Ticker Interval: {(HOP_LENGTH / sr) * 1000:.2f} ms")
        print(f"Partials (N): {N_PARTIALS}")
        print(f"Total F(i)/G(i) pairs: {len(F_data)}")
        print("\n--- STEPS TO RUN: ---")
        print("1. Open a new Desmos graph")
        print("2. Open developer tools (F12)")
        print("3. Go to the Console tab")
        print("4. Paste and hit Enter.")
        print("5. Close the developter tools")
    else:
        print("Audio analysis failed.")
