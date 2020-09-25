# DLP-Lab3
## Diabetic Retinopathy Detection 
(The report please refer to DLP_LAB3_report.pdf) 

#### Lab Objective
* In this lab, you will need to analysis diabetic retinopathy (糖尿病所引發視網膜病變) in the following three steps.
  * Step 1. You need to write your own custom DataLoader through PyTorch framework.   
  * Step 2. You need to classify diabetic retinopathy grading via the ResNet [1].
  * Step 3. You have to calculate the confusion matrix to evaluate the performance.


#### Requirements
* Implement the ResNet18、ResNet50 architecture and load parameters from a pretrained model
* Compare and visualize the accuracy trend between the pretrained model and without pretraining in same architectures, you need to plot each epoch accuracy (not loss) during training phase and testing phase.
* Implement your own custom DataLoader
* Calculate the confusion matrix and plotting


#### Dataset - Diabetic Retinopathy Detection (kaggle)
* Diabetic retinopathy is the leading cause of blindness in the working-age population of the developed world.
* This dataset provided with a large set of high-resolution retina images taken under a variety of imaging conditions.  Format: .jpeg
* Reference : https://www.kaggle.com/c/diabetic-retinopathy-detection#description

![Dataset](/picture/kaggle.png "Dataset")

#### Prepare Data
* 28,099 images for training 
* 7025 for testing
* The image resolution is 512x512 and has been preprocessed.
* Download link:
* https://drive.google.com/open?id=1RTmrk7Qu9IBjQYLczaYKOvXaHWBS0o72

![Prepare Data](/picture/Prepare_Data.png "Prepare Data")


#### ResNet
* ResNet (Residual Network) is the Winner of ILSVRC 2015 in image classification, detection, and localization, as well as Winner of MS COCO 2015 detection, and segmentation 

![ResNet](/picture/ResNet.png "ResNet")


* To solve the problem of vanishing/exploding gradients, a skip / shortcut connection is added to add the input x to the output after few weight layers as below 

![ResNet](/picture/ResNet2.png "ResNet")

* ResNet can avoid vanishing gradient problem

![ResNet](/picture/ResNet3.png "ResNet")
![ResNet](/picture/ResNet4.png "ResNet")

* ResNe18(Basic block),  ResNet50(Bottleneck block)
![ResNet](/picture/ResNet5.png "ResNet")


#### Using Pretrained Model 
* Using pretrained model by torchvision module

![Using Pretrained Model](/picture/Using_Pretrained_Model.png "Using Pretrained Model")

#### Result Comparison
* Compare and visualize the accuracy trend between the pretrained model and without pretraining in same architectures, you need to plot each epoch accuracy (not loss) during training phase and testing phase.

![Result Comparison](/picture/Result_Comparison.png "Result Comparison")

#### Confusion Matrix
* Calculate the confusion matrix and plotting

![Confusion Matrix](/picture/Confusion_Matrix.png "Confusion Matrix")

#### Hyper Parameters
* Batch size= 4        
* Learning rate = 1e-3        
* Epochs = 10 (resnet18),  5 (resnet50)
* Optimizer: SGD      Momentum = 0.9     Weight_decay = 5e-4
* Loss function: torch.nn.CrossEntropyLoss()

* You can adjust the hyper-parameters according to your own ideas.

* If you use “nn.CrossEntropyLoss”, don’t add softmax after final fc layer because this criterion combines LogSoftMax and NLLLoss in one single class.


