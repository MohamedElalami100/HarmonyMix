from flask import Flask, request, render_template
import pickle
from models_Training import allSongs_df
import json

app = Flask(__name__)

is_model = pickle.load(open('is_model.pkl', 'rb'))
mf_model = pickle.load(open('mf_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tryModel', methods=['GET', 'POST'])
def tryModel():
    allSongs_json = allSongs_df.to_json(orient='records')
    
    if request.method == 'POST':
        playlistLength = request.form.get("playlistLength")
        algorithm = request.form.get("algorithm")
        selectedSongs_json = request.form.get("songsArray")
        if selectedSongs_json:
            selectedSongs = json.loads(selectedSongs_json)   
        
        if len(selectedSongs) < 2:
            error_message = "Please select at least 2 songs."
            return render_template('Try-HarmonyMix.html',allSongs = allSongs_json ,error=error_message)
        
        if algorithm == '0':
            playlist = is_model.get_similar_items(selectedSongs, playlistLength)
        elif algorithm == '1':
            playlist = mf_model.computeEstimatedRatings(selectedSongs, [10] * len(selectedSongs), playlistLength)
        
        return render_template('Try-HarmonyMix.html',allSongs = allSongs_json ,playlist = playlist)
        
    return render_template('Try-HarmonyMix.html', allSongs = allSongs_json)

@app.route('/exploreModelAlgorithms')
def exoloreModelAlgorithms():
    return render_template('Explore-Model-Algorithms.html')

if __name__ == '__main__':
    app.run()