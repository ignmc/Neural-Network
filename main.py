from experiment_dataset import experiment
from dataset import get_processed_data


EPOCHS = 200
DATASET = 'winequality-red.csv'
LAYERS = (11, 9, 8, 6)
LEARNING_RATE = 0.1

if __name__ == '__main__':
    data, labels = get_processed_data(DATASET)
    experiment(data, labels, EPOCHS, LAYERS, LEARNING_RATE)
