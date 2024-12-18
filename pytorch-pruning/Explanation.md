## Finetune.py 
## This file provides functionality to either train or load and prune fine-tuned model

## Training 
1. Load ModifiedVGG16Model 
Utilizes pretrained VGGs feature extraction layers (self.features - conv and pooling) while freezing them to retain the learned weights from pretraining. param.requires_grad = False ensures that these layers are not updated during training.
Forward pass is just getting the features than classifying them 
- Features are representations that encode meaningful information about the input data
- After passing through self.features (VGG16's convolutional layers), the image is transformed into a tensor of size 7×7×512 (for default VGG16).
2. Train with PrunningFineTuner_VGG16's train() method 

## Prunning
1.Load a model 
2. prune() 
2.1. Somehow also train the model for certain epochs 
2.2. Make all parameters trainable 
2.3. Get #filters over all conv layer , and fix the number of filters to prune . Calculate # iterations required to prune that many 

2.4. In each iteration do:
1. Rank filters 
via get_candidates_to_prune which uses prunner 
(See later explanation for prunner)
2. Rest is kinda irrelevant for us 
Which is actual pruning and finetuning 

## FilterPrunner
1. reset sets filter_ranks ={}
2. Train_epoch 
- Do self.prunner.forward 
What forward do:
- Register hook whenever a layer is convolutional layer 
- append output of the layer to input x to "activations" array 
- add a hook to output tensor x - calls compute_rank whever gradient is calculated for that x 
- You have to call forward first cause it defines class variables 
- activation_index is like index for conv layers 
- Start with 0th layer then gets incremeneted 

##### Then compute rank!
- The compute_rank method calculates a measure of each filter’s importance based on the product of its activation values and their corresponding gradients (a “Taylor approximation” of the filter’s influence on the loss). The variable taylor represents the average of this product across the batch and spatial dimensions, resulting in a single importance score per filter.

- The grad_index variable is used to properly align the backward passes with the recorded activations, ensuring that each activation is matched with its corresponding gradient in reverse order of the forward pass.


## TODO:
- We need to check if our model also have features. if not we need to make exactly same - structure bc of "activation_index" and so one

- i want to know what activation's dimension is 
- and also the output o