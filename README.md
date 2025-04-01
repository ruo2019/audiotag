# Audiotag

Instructions:

1. `git clone https://github.com/qingy1337/audiotag.git`
2. `cd audiotag`
3. Put mp3 files into `./static/mp3/`
4. Open Terminal / iTerm
5. `python app.py`
6. Navigate to `127.0.0.1:5000` in a new browser tab
7. Add tags with `Enter`
8. If you click `Next MP3` and nothing happens, do *not* click it again. Just refresh the page.
9. After tagging all mp3 files, run `python app.py -mood "some mood, some other mood, etc" -top 10 # top 10 mp3s that match`

Note: will load a 110M sentence-transformers model to run locally.
