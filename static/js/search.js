    const songSuggestions = allSongs.map(song => song.song);
    const inputField = document.getElementById("name-91da");
    const suggestionBox = document.getElementById("suggestion-box");

    inputField.addEventListener("input", function() {
        const query = inputField.value.toLowerCase();
        const suggestions = songSuggestions.filter(song => song.toLowerCase().includes(query));

        // Clear previous suggestions
        suggestionBox.innerHTML = '';

        if (suggestions.length > 0) {
            const ul = document.createElement("ul");

            suggestions.forEach(suggestion => {
                const li = document.createElement("li");
                li.textContent = suggestion;

                li.addEventListener("click", function() {
                    inputField.value = suggestion;
                    suggestionBox.innerHTML = ''; // Clear suggestion box after selection
                });

                ul.appendChild(li);
            });

            suggestionBox.appendChild(ul);
            suggestionBox.style.display = 'block';
        } else {
            suggestionBox.style.display = 'none';
        }
    });

    // Hide the suggestion box when clicking outside of it
    document.addEventListener("click", function(event) {
        if (!suggestionBox.contains(event.target) && event.target !== inputField) {
            suggestionBox.style.display = 'none';
        }
    });
