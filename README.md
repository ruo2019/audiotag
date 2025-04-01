# Audiotag

Instructions:

1. Put mp3 files into `./static/mp3/`
2. Open Terminal / iTerm
3. `python app.py`
4. Navigate to `127.0.0.1:5000` in a new browser tab
5. Add tags with `Enter`
6. If you click `Next MP3` and nothing happens, do *not* click it again. Just refresh the page.
7. After tagging all mp3 files, run `python app.py -mood "some mood, some other mood, etc" -top 10 # top 10 mp3s that match`

Note: will load a 110M sentence-transformers model to run locally.
