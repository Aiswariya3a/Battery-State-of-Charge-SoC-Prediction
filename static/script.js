document.getElementById('fetchButton').addEventListener('click', function() {
    fetch('/fetch_random')
    .then(response => response.json())
    .then(data => {
        console.log("API Response:", data);  // Debugging to check JSON structure
        
        if ("Predicted Next SoC" in data) {
            document.getElementById('predictionResult').innerText = data["Predicted Next SoC"] + "%";
        
            // Update Chart.js
            predictionChart.data.labels.push("New Prediction");
            predictionChart.data.datasets[0].data.push(data["Predicted Next SoC"]);
            predictionChart.update();
        } else {
            document.getElementById('predictionResult').innerText = "Error: No prediction";
        }
    })
    .catch(error => {
        console.error("Error fetching data:", error);
        document.getElementById('predictionResult').innerText = "Error fetching data";
    });
});
