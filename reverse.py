from pydub import AudioSegment
import os
import sys

def reverse_mp3(input_file):
    # Load the MP3 file
    try:
        audio = AudioSegment.from_mp3(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error: Unable to load '{input_file}'. {e}")
        return

    # Reverse the audio
    reversed_audio = audio.reverse()

    # Create the output file path
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_reversed{ext}"

    # Export the reversed audio to the new file
    reversed_audio.export(output_file, format="mp3")

    print(f"Reversed MP3 saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 reverse.py <path_to_mp3_file>")
    else:
        # Input MP3 file path (passed as a command-line argument)
        mp3_file = sys.argv[1]
        reverse_mp3(mp3_file)
