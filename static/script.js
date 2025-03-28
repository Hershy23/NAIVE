function predict() {
    let input = document.getElementById("features").value.trim();
    
    if (!input) {
        document.getElementById("result").innerText = "Please enter features.";
        return;
    }

    let features = input.split(",").map(val => parseFloat(val.trim()));  // Convert to float

    if (features.some(isNaN)) {
        document.getElementById("result").innerText = "Invalid input. Enter valid numbers only.";
        return;
    }

    console.log("Sending Features:", features);

    fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: features })
    })
    .then(response => response.json())
    .then(data => {
        console.log("Server Response:", data);
        if (data.error) {
            document.getElementById("result").innerText = "Error: " + data.error;
        } else {
            document.getElementById("result").innerText = "Predicted Output: " + data.prediction;
        }
    })
    .catch(error => {
        console.error("Fetch Error:", error);
        document.getElementById("result").innerText = "Request failed. Ensure Flask is running on 127.0.0.1:5000.";
    });
}
