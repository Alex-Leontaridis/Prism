# Context for Prism: Automated Neural Architecture Search Web Application

Project Overview:
Prism is a web application designed to automatically discover optimal neural network architectures for user-provided datasets through Neural Architecture Search (NAS). The project emphasizes training models from scratch without relying on pre-trained models or external LLM APIs, focusing on transparency, reproducibility, and lightweight, interpretable architectures.

Core Idea:
The system implements a NAS engine that explores a defined search space of neural network architectures using a combination of random search and evolutionary mutation strategies. It generates candidate models, trains and evaluates them on datasets like CIFAR-10 or MNIST, tracks performance metrics, and iteratively refines architectures to find the best-performing models.

Key Features:
- User-friendly web interface for dataset upload, search space customization, and NAS job control
- Backend API server managing NAS job lifecycle and dataset storage
- NAS Engine Core handling search strategy, model generation, training, and evaluation
- Search space definition module specifying layer types, hyperparameters, and architecture constraints
- Model generator converting architecture specifications (JSON) into trainable neural network models
- Dataset manager supporting standard datasets (CIFAR-10, MNIST) and user-uploaded data (CSV, images)
- Training and evaluation pipeline providing performance metrics (accuracy, loss, training time)
- Visualization tools for monitoring search progress and comparing architectures
- Open-source, reproducible codebase with deployment-ready infrastructure

Project Goals:
- Build a fully functional end-to-end NAS pipeline from architecture generation to model evaluation
- Ensure code transparency, ease of reproducibility, and comprehensive documentation
- Deploy the application as a live web service accessible via browser
- Create demo videos and project write-ups explaining the model, training process, and system design

Technical Stack:
- Frontend: React or Next.js for user interface
- Backend API: FastAPI or Flask for serving endpoints and managing NAS workflow
- ML Framework: PyTorch or TensorFlow/Keras for model building and training
- Containerization: Docker and optionally Kubernetes for deployment and scaling
- Storage: Support for cloud object storage (AWS S3, Azure Blob), databases, and local file systems

Current Phase & Immediate Task:
Before building the NAS engine, validate the training pipeline by creating a dummy training script that:

- Loads a standard dataset (MNIST or CIFAR-10)
- Defines a simple neural network (CNN or MLP)
- Trains for several epochs while logging accuracy and loss
- Saves the trained model weights in reusable format
- Serves as a baseline for integrating NAS-generated architectures later

Subsequent Tasks (Planned):
- Define and implement the search space encoding and mutation operations
- Develop the NAS engine search loop (random + evolutionary)
- Integrate model generator converting JSON specs to ML models
- Build dataset manager supporting user uploads and preprocessing
- Develop frontend UI for user interaction and progress visualization
- Deploy backend and NAS engine with scaling infrastructure
- Prepare documentation, demo videos, and open-source release

Deliverables:
- Clean, well-commented code for each module
- Clear README with setup, usage, and development instructions
- Functional web app deployed and accessible online
- Project write-up explaining approach, model details, and results

---

