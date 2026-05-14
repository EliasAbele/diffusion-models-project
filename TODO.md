# TODO

just write your name behing a task if you take it on

## Bigger Tasks

### Implement time embedding into the unet
Normal unets take in the noisy image as well as the time $t$, which sets the amplitude of the noise, and use that to predict the noise pattern. The unet i implemented so far does not take in the time $t$, but just the noisy image. that does still work, but it can be assumed that the model would improve if we also train it on t.
This is a bit of a bigger task. I have already written/started writing two time embedding classes (a linear embedding and a sinusoidal embedding, which is normally used), they are in the ea4 notebook but integrating them into the unet structure isn't trivial at all. the theory behind it also isn't obvious. the noisyMNIST dataset class also has to be slightly altered, that has also already been done in the ea4 notebook. 
we should be able to turn the time embedding of by not giving in a time simply or by setting some parameter to false (alhtough the first option is better probably)

### implement attention layer into the unet
a further significant improvement in the performance of the network could be achieved by implementing an attention layer at the bottleneck of the unet architecture.

### implementing a conditioned network
currently, we cannot tell the model what to generate. we can also condition the network on what number to predict. in the training, we then give in noisy image, t and the number which the image actually is. how to embed this information i don't know, you would have to read up on this. 
if we simply give in None as the conditioning, we should get the standard unconditioned training (i think (not sure) that is the standard way these architectures are build). I think this means that during the training, we also have to train the model on labeled and unlabeled images.
the generate_image function will also have to be adapted, (we use prediction_conditioned-prediction_unconditioned as an update), see lilian weng blog article


## Medium Tasks

### DDRM: image completion
we can't only generate new images, but also complete images (so let's say we only have the left half of the image). for that, we wouldn't need to train our unet again or change anything in that architecture, we only have to alter the generation function: as i understood it (but we can read up on that more), we generate an image like normally, but instead of deducting the noise in our known half of the image, we update the picture there with the picture we have already in each step. 
I am not sure how complicated this is, it could turn out that this step here means only changing 2-3 lines of code in the generation function to get this to work

### training/downloading an image classifier to quantify our image quality
There is currently no good way to quantify how good our images are. One way would be to train a binary classifier on differentiating between generated and real images. However, Christoph said that they normally just get 100% everything right, because they will also pick up on features which are completely invisible to humans. Therefore, he said there also classifiers which in the first layers have set feature maps, which pick up more on features humans also pick up on. research that and download/build such a classifier, which then allows us to judge our image quality

## (Smaller) Tasks

### test unet_C0_128_convs_2

i trained for a longer time the above model (can be moved in models/unet_C0_128_convs_2/trained.pkl), which should be the best model so far. i haven't tested yet how the quality of the images are that are being produced

### workflow: write a function that sends an image generation task to the gpu
I (or rather Claude) has written a very nice python functions, which takes in a model and train and test loaders and sends them as a job to a node. very convenient, but we also want sth similar for generating images, because that isn't really feasible either without a gpu. additionaly, one can build a higher level function which takes the train_on_cluster function and the function for generating images and combines them, so we can send a training job, the training is being done, and at the end, with the trained model, we also get back a tensor with n_images generated images.

### write dataloaders for the galaxy data
there is a galaxy dataset, which we can load, train our model on that, and then see what images it creates. we can try that for the current state of our model, and for more complex models

