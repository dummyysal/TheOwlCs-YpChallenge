# ✨ The Owl Cybersecurity Incident Response Automation

OWL is an advanced cybersecurity solution developed to detect and analyze network anomalies, prevent cyberattacks, and provide actionable insights for network protection. It leverages machine learning models, APIs, and NLP capabilities to identify security threats and offer targeted recommendations. 
Created as part of the IEEE CS ENIT SBC initiative for the CS & YP Challenge at TSP12.

## 🛠️ Project Structure
```
│
├── TheOwl/                    # Full project code for EC2 instance deployment  
│
├── API_Calls/                  # API calls project for model deployment
│
├── Models_Links/               # Links for the model files and resources
│
├── Demo_Test_Video/            # Link for demo test video on eNSP topology
│
└── README.md                   # Project documentation
```
## Key Features

`1. Anomaly Detection`:Usage of pre-trained machine learning models to analyze network traffic and identify unusual patterns. 
`2. Predictive Analysis and Security Recommendations`: Providing insights into detected anomalies, predicting the type and severity of potential attacks.
`3.Owl Chat Assistant`: An interactive chatbot capable of answering questions about cybersecurity tactics (e.g., MITRE ATT&CK) and specific vulnerabilities.

##  Technology Stack

- Backend: Flask & FastAPI for creating robust APIs
- Machine Learning: Anomaly detection models (trained on the UNSW_NB15 dataset and network logs ) 
- Natural Language Processing (NLP): Provides recommendations and detailed CVE information 
- Cloud Hosting: Deployed on AWS EC2 for scalability and public accessibility
- Security & Scalability: Utilizes AWS security features to ensure a secure and scalable architecture
  




## Installation and Setup

 - 1. Clone this repository:

```
git clone https://github.com/dummyysal/TheOwlCs-YpChallenge.git
cd TheOwlCs-YpChallenge

```
- 2. Install Required Libraries:
```
pip install -r requirements.txt
```

- 3. Set Up Environment Variables

```
Configure AWS access keys, database credentials, and any other sensitive information in a .env file.
Start the Flask and FastAPI servers locally
```



### 🤝 Contributions
Contributions are welcome! Please fork the repository and create a pull request with your updates. Make sure to adhere to the project's coding style and add tests for any new features.
