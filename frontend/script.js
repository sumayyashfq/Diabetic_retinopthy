function predictDR() {
    const fileInput = document.getElementById("imageInput");

    if (fileInput.files.length === 0) {
        alert("Please upload a retinal image!");
        return;
    }

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert("Error: " + data.error);
            } else {
                // Save result to session storage to pass complex data (metrics, plots) to the next page
                sessionStorage.setItem('predictionResult', JSON.stringify(data));
                // Redirect to result page (no params needed now)
                window.location.href = "result.html";
            }
        })
        .catch(error => {
            console.error("Error:", error);
            alert("Failed to connect to the backend.");
        });
}
