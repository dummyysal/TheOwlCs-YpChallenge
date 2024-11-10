document
  .getElementById("uploadForm")
  .addEventListener("submit", async (event) => {
    event.preventDefault();
    const fileInput = document.getElementById("fileInput").files[0];
    const formData = new FormData();
    formData.append("file", fileInput);

    // Envoyer le fichier au backend
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    visualizeData(data);
  });

function visualizeData(data) {
  const ctx = document.getElementById("pcapChart").getContext("2d");
  const timestamps = data.map((packet) =>
    new Date(packet.timestamp * 1000).toLocaleTimeString()
  );
  const lengths = data.map((packet) => packet.length);

  new Chart(ctx, {
    type: "line",
    data: {
      labels: timestamps,
      datasets: [
        {
          label: "Packet Length over Time",
          data: lengths,
          borderWidth: 1,
        },
      ],
    },
    options: {
      scales: {
        x: {
          title: { display: true, text: "Time" },
        },
        y: {
          title: { display: true, text: "Packet Length (bytes)" },
          beginAtZero: true,
        },
      },
    },
  });
}