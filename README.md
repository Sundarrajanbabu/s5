# ML Model CI/CD Pipeline

This project demonstrates a complete CI/CD pipeline for a machine learning model using GitHub Actions. It includes model training, testing, and automated validation checks for a simple CNN trained on the MNIST dataset.

## Project Structure
├── model/
│ ├── init.py
│ └── network.py
├── tests/
│ └── test_model.py
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml
├── train.py
├── requirements.txt
├── .gitignore
└── README.md
## Features

- Simple CNN architecture for MNIST digit classification
- Automated training pipeline
- Model validation checks:
  - Parameter count (< 100,000)
  - Input/output dimension verification
  - Model accuracy validation (> 80%)
- CPU-only implementation for CI/CD compatibility
- Automated model artifact storage

## Model Architecture

The model is a simple CNN with:
- 2 convolutional layers
- 2 max pooling layers
- 2 fully connected layers
- ReLU activation functions

## Requirements

- Python 3.8+
- PyTorch (CPU version)
- torchvision
- pytest

## Local Setup

1. Clone the repository:
bash
git clone <repository-url>
cd <repository-name>

2. Create and activate a virtual environment:
bash
python -m venv venv
source venv/bin/activate

3. Install dependencies:
bash
pip install -r requirements.txt

4. Run the training script:
bash
python train.py

5. Run the tests:
bash
pytest tests/test_model.py

## GitHub Actions


## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Sets up Python environment
2. Installs dependencies
3. Trains the model
4. Runs validation tests
5. Stores the trained model as an artifact

The pipeline is triggered on every push to the repository.

## Model Artifacts

Trained models are saved with timestamps in the format: `model_<timestamp>.pth` in the `model/` directory.  


When running locally, these files are ignored by git (specified in .gitignore).

## Testing

The test suite verifies:
1. Model parameter count is within limits
2. Model accepts 28x28 input images
3. Model outputs 10 classes (digits 0-9)
4. Model achieves >80% accuracy on test set

## Notes

- The implementation uses CPU-only PyTorch to ensure compatibility with CI/CD environments
- Training is limited to 1 epoch for CI/CD efficiency
- Model artifacts are automatically stored in GitHub Actions

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.