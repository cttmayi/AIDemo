import torch

def cross_entropy_loss(logits, targets):
    # logits: [batch_size, num_classes]
    # targets: [batch_size]
    
    # Step 1: Softmax
    softmax = torch.nn.Softmax(dim=1)
    
    # Step 2: Convert logits to probabilities
    probabilities = softmax(logits)
    print('probabilities', probabilities)
    # Step 3: Gather the probabilities corresponding to the targets
    # We use the .gather() method to select the probabilities of the target classes
    # The dimension=1 argument specifies that we are selecting along the classes axis
    correct_probabilities = probabilities.gather(1, targets.view(-1, 1))
    print('correct_probabilities', correct_probabilities)
    
    # Step 4: Calculate the negative log likelihood and mean to get the loss
    # We use .squeeze(1) to remove the extra dimension added by .view()
    # Then we take the negative log of the correct probabilities
    # Finally, we take the mean over the batch
    loss = -correct_probabilities.log().squeeze(1).mean()
    print('correct_probabilities.log()', correct_probabilities.log())
    print('correct_probabilities.log().squeeze(1)', correct_probabilities.log().squeeze(1))

    return loss

# Example usage:
logits = torch.tensor([[1.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
targets = torch.tensor([0, 2])

loss = cross_entropy_loss(logits, targets)
print(loss)  # Output will be the average loss over the batch