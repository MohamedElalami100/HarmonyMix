        // JavaScript to handle adding values to the list
        var songsArray = [];

        // Function to add the input value to the list and update the UI
        function addValueToList() {
            var inputValue = document.getElementById("name-91da").value;
            if (inputValue) {
                songsArray.push(inputValue);
                updateTableUI();
                // Clear the input field
                document.getElementById("name-91da").value = "";
            }
        }

        // Function to update the table on the UI
        function updateTableUI() {
            var tableBody = document.querySelector(".u-table-body-1"); // Get the table body element
            var section = document.querySelector(".u-section-2"); // Get the section element

            // Show the section if there are songs in the array
            section.style.display = "block";
            
            // Clear the table body
            tableBody.innerHTML = "";
        
            for (var i = 0; i < songsArray.length; i++) {
                var song = songsArray[i];
                song = song.split(',')
                
                // Create a new row
                var row = document.createElement("tr");
        
                // Create and populate the cells
                var titleCell = document.createElement("td");
                titleCell.textContent = song[0];
                titleCell.className = "u-border-1 u-border-palette-5-dark-1 u-table-cell u-table-cell-4";
        
                var artistCell = document.createElement("td");
                artistCell.textContent = song[1];
                artistCell.className = "u-border-1 u-border-palette-5-dark-1 u-table-cell u-table-cell-4";
        
                var releaseYearCell = document.createElement("td");
                if (song.length > 2 ) {
                    releaseYearCell.textContent = song[2];
                }
                else{
                    releaseYearCell.textContent = "";

                }                
                releaseYearCell.className = "u-border-1 u-border-palette-5-dark-1 u-table-cell u-table-cell-4";
        
                // Append the cells to the row
                row.appendChild(titleCell);
                row.appendChild(artistCell);
                row.appendChild(releaseYearCell);
        
                // Append the row to the table body
                tableBody.appendChild(row);
            }
        
            // Update the hidden input field with the JSON representation of songsArray
            document.getElementById("songsArrayInput").value = JSON.stringify(songsArray);
        }


        // Add a click event listener to the "Add to List" button
        document.getElementById("add-music").addEventListener("click", addValueToList);
        
        document.getElementById("generatePlaylistButton").addEventListener("click", function(event) {
            event.preventDefault(); // Prevent the default link behavior (navigation)
        
            // Find the form element by its ID and submit it
            var form = document.getElementById("playlistForm");
            if (form) {
              form.submit();
            }
        });
