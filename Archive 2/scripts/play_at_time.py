import pygame
import datetime
import pytz
import time
import os

def start_alarm():
    # 1. Get User Input
    print("--- Pacific Time MP3 Player ---")
    target_time_input = input("Enter the time to play (e.g., 02:30 PM or 11:05 AM): ").strip()
    filename = input("Enter the MP3 filename (including .mp3): ").strip()

    # Verify if file exists
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found in this folder.")
        return

    # Normalize the time format to ensure it matches the comparison string later
    try:
        # This converts user input into a standardized HH:MM AM/PM string
        standard_time = datetime.datetime.strptime(target_time_input, "%I:%M %p").strftime("%I:%M %p")
        print(f"Alarm set for: {standard_time} (Pacific Time)")
    except ValueError:
        print("Invalid time format! Please use HH:MM AM/PM (e.g., 08:15 AM).")
        return

    # 2. Initialize Pygame Mixer
    pygame.mixer.init()

    pacific_tz = pytz.timezone('US/Pacific')
    print("Waiting for the exact time...")
    pygame.mixer.music.load(filename)

    # 3. The Monitoring Loop
    while True:
        # Get current time in Pacific Time zone
        now_pacific = datetime.datetime.now(pacific_tz)
        current_time_str = now_pacific.strftime("%I:%M %p")

        # Optional: Print current time to console every second to show it's working
        # Use \r to overwrite the same line
        print(f"Current PT: {current_time_str}", end="\r")

        if current_time_str == standard_time:
            print(f"\nTime reached! Playing: {filename}")
            try:
                pygame.mixer.music.play()

                # Keep the script running while the music plays
                while pygame.mixer.music.get_busy():
                    time.sleep(1)
                print("\nPlayback finished.")
            except Exception as e:
                print(f"\nAn error occurred during playback: {e}")
            break

if __name__ == "__main__":
    start_alarm()
