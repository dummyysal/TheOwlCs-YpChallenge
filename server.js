const express = require("express");
const multer = require("multer");
const pcapParser = require("pcap-parser");
const fs = require("fs");
const path = require("path");

const app = express();
const upload = multer({ dest: "uploads/" });

app.use(express.static(path.join(__dirname, "")));

app.post("/upload", upload.single("file"), (req, res) => {
  const filePath = req.file.path;
  const parser = pcapParser.parse(filePath);
  const packets = [];

  parser.on("packet", (packet) => {
    packets.push({
      timestamp: packet.header.timestampSeconds,
      length: packet.header.origLen,
      // Ajouter d'autres informations nécessaires à extraire ici
    });
  });

  parser.on("end", () => {
    fs.unlinkSync(filePath); // Supprime le fichier après traitement
    res.json(packets); // Renvoie les données analysées
  });
});

app.listen(3000, () => console.log("Server started on http://localhost:3000"));
