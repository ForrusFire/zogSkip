import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import NotesDataset, split_and_load_dataset


MODEL_NAME = "LSTMAttLSTM"
DATASET = "beethoven_mozart"
PARAMETER_SET = "A"

Model = getattr(importlib.import_module("models." + MODEL_NAME + ".model"), "Model")
pm = importlib.import_module("models." + MODEL_NAME + ".parameters" + ".parameters_" + PARAMETER_SET)


PATH_TO_NOTES = "data/preprocessed/" + DATASET + "_notes"
PATH_TO_SAVE_WEIGHTS = "models/" + MODEL_NAME + "/weights/weights_" + DATASET + "_" + PARAMETER_SET
PATH_TO_SAVE_LOSSES = "models/" + MODEL_NAME + "/losses/loss_" + DATASET + "_" + PARAMETER_SET + ".txt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


############ Parameters ############

## Dataset
sequence_length = 100
train_val_split_percentage = 0.80
train_batchsize = 64
val_batchsize = 64

## Model
LSTM1_num_units = pm.LSTM1_num_units
SeqSelfAttention_num_units = pm.SeqSelfAttention_num_units
LSTM2_num_units = pm.LSTM2_num_units
dropout_prob = pm.dropout_prob

## Training
num_epochs = 100
lr = 1e-3
alpha = 0.9
momentum = 0.0
epsilon = 1e-7

###################################


def load_dataset(notes_path, sequence_length):
    print("Loading dataset...")
    # Create custom dataset
    dataset = NotesDataset(notes_path, sequence_length)
    num_classes = dataset.get_num_classes()

    # Split into training and val sets, and load them into data loaders
    train_loader, val_loader = split_and_load_dataset(dataset, train_val_split_percentage, train_batchsize, val_batchsize)
    print("Dataset loaded")

    return (train_loader, val_loader, num_classes)


def train_model(model, loss_fn, optimizer, train_loader, val_loader, num_epochs):
    """
    Trains the model and returns the training and validation loss lists
    """
    train_losses = []
    val_losses = []

    print("Pretrained model:")
    print(model.state_dict())

    print("Training...")
    train_step = make_train_step(model, loss_fn, optimizer)

    for i in range(num_epochs):
        batch_losses = []

        # Iterate through each batch for training
        for x_batch, y_batch in train_loader:
            # Move x and y batch data into GPU if available
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            loss = train_step(x_batch, y_batch)
            batch_losses.append(loss)

        # Add average loss to train losses list
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)

        # Compute validation losses
        with torch.no_grad():
            batch_losses = []

            # Iterate through each batch for validation
            for x_val, y_val in val_loader:
                x_val = x_val.to(DEVICE)
                y_val = y_val.to(DEVICE)

                model.eval() # Sets model in prediction mode
                yhat = model(x_val)

                loss = loss_fn(yhat, y_val).item()
                batch_losses.append(loss)

            # Add average loss to val losses list
            val_loss = np.mean(batch_losses)
            val_losses.append(val_loss)

        # Display progress
        print(f"{i+1} of {num_epochs} epochs trained...")

        # Save model every 25 epochs
        if (i+1 % 25 == 0):
            save_model(model, PATH_TO_SAVE_WEIGHTS, i+1)


    print("Trained model:")
    print(model.state_dict())

    return train_losses, val_losses


def make_train_step(model, loss_fn, optimizer):
    """
    Returns a function that performs the training step
    """
    def train_step(x, y):
        # Performs a training step and returns the loss
        model.train() # Sets model in training mode

        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad() # Resets optimizer

        return loss.item()

    return train_step


def save_model(model, save_path, epoch):
    # Saves the model
    print("Saving model...")
    torch.save(model.state_dict(), save_path + "_" + str(epoch) + "epochs.pth")
    print("Model saved")


def save_losses(train_losses, val_losses, save_path):
    # Saves the losses
    losses = [train_losses, val_losses]

    print("Saving losses...")
    with open(save_path, 'w') as file:
        file.write(str(losses))
    print("Losses saved")



if __name__ == "__main__":
    train_loader, val_loader, num_classes = load_dataset(PATH_TO_NOTES, sequence_length)

    # Create model
    model = Model(
        LSTM1_num_units,
        SeqSelfAttention_num_units,
        LSTM2_num_units,
        dropout_prob,
        sequence_length,
        num_classes
    ).to(DEVICE)

    # Create loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, momentum=momentum, eps=epsilon)

    # Perform training
    train_losses, val_losses = train_model(model, loss_fn, optimizer, train_loader, val_loader, num_epochs)

    # Save model and losses
    save_model(model, PATH_TO_SAVE_WEIGHTS, num_epochs)
    save_losses(train_losses, val_losses, PATH_TO_SAVE_LOSSES)

    print("Training complete!")