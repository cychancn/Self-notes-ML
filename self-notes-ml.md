# Preface
Self-learning notes on the lectures of Machine Learning produced by Prof. Lee Hung-yi
http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html

## Topics
These topics are categorized based on the titles of the lectures
- Regression
- Basic Concepts
- Gradient Descent
- Classification
- DL
- CNN
- RNN

## Regression
Three important steps of machine learning
- Find a suitable model (Example, linear model, y = b+ w*x, quadratic model)
- Determine its fitness (Using a loss function, eg: summation of (y_hat - f(x) )^2)
- Picking the best function (Finding the parameters that has a minimum loss function value)

Why use gradient descent?
An efficient method to find the best function
Start by randomly picking initial parameters, calculate gradient of function 
No local optimals since loss function L is a convex function
Problems:
Overfitting - A more complex model does not lead to better performance on testing(real) data
Designing the model - Discovering hidden factors requires domain knowledge

Solutions to overfitting:
Regularization, adding a term with a new parameter lambda to the loss function: lambda * \sum (weight_i)^2, which aims to make the function "smoother", and be less sensitive to the errors/noises of the testing data.

## Basic Concepts
Sample mean is not population mean
Biased estimator sample variance: s^2 = 1/N \sum_n (x^n - m)^2, E[s^2] = (N-1)/N * sigma^2 != sigma^2
Variance and bias of model: Given different sampling data as training sets, there will be different "best fit" functions. Bias is calculated as t
Simpler models have a larger bias, Complex models have a larger variance
Can't fit training data -> large bias -> underfitting
Can fit training data but does not fit testing data -> large variance -> overfiting
Therefore there is a trade-off between bias and variance
Problems:
No access to enough data
Model fits public testing sets but not private testing sets
Solutions:
Generate training data
N-fold Cross validation: Seperate the full training set to N-1 sets of training set and one set of validation set, train N models, and compare errors of different models under these different settings, then pick the best model and train the model under the full training set

## Gradient Descent 
Learning rates:
Low learning rates cause slow progresses, high learning rates cause wrong progresses.
Visualizing suitable learning rate:
Check value of loss function with no of parameters updates under different learning rates
Adaptive learning rates: 
Vanilla gradient descent:
Uses learning rate eta^t, where eta^t = eta/(\sqrt(t+1))
w^{t+1} < - w^t - eta^t* g^t
Adagrad:
Uses (eta^t) /sigma^t, where sigma^t = \sqrt(1/(t+1) * \sum^t_i (g^i) ^2)
w^{t+1} < - w^t - eta^t /sigma^t * g^t = w^t - eta/(\sqrt(\sum^t_i (g^i) ^2)) *g^t
Intuition: to determine how close the points are to the optimum, the best step is to descent with rate (|first derivative|/second derivative), but not to only consider its first derivative
Stochastic Gradient Descent:

Feature Scaling:
To normalize the ranges of the variables, which allows it to descent much faster
For a value x_i^r (component i with data entry r), do x_i^r < - (x_i^r - m_i)/ sigma^i


Why gradient descent works:
Taylor series approximates well enough with only lower order terms for a small enough range of values.
Problems of gradient descent:
Slow at plateau, stuck at paddle points, stuck at local minima 

### Classification 1:
Problems of using linear regression:
Linear regression requires optimal points. Using closeness to optimal points as classifications, the model will try to correct for far data points and penalize examples that are too correct.
Alternatives:
Use a different function (model) (Instead of closeness to points, use a range of points such as >0, < 0 as classifications)
The loss function will be redefined as the sum of the dirac delta function

Classification: (Using logistic regression)
Finding probability of object x belonging to class i, found by bayes' theorem.
Problem:
Cannot directly find prior probability since the data most likely will not be in the training set
Solution:
Assuming data is sampled from a gaussian distribution, than the probability of x can be found.
The covariance matrix and the mean required for the mentioned gaussian distribution is determined from the maximum likelihood, which means with the data sampled, we choose the most probable mean mu^* and covariance matrix sigma^*.
Assume gaussian function f(mu, sigma, x) is the probability of data x appearing in a gaussian distribution with mu and sigma, the likelihoodfunction will be:
L(mu, sigma) = product of all i's (f(mu,sigma,x^i)), then maximize L
mu* = 1/N \sum x^i
sigma* = 1/N (\sum x^i - mu*)*(x^i - mu*)^T
Problem: less accurate
Solution: Modifying the model
Use the same covariance matrix, since the data in each classification came from the same training set
Likelihood function: (L,mu^1,mu^2,...,mu^n,sigma) = product of all (k,i) f(mu^k,sigma,x^i), where k is the classifications 1 to n, and data x^i belongs to classification k.
mu^i* is the same,
sigma* = \sum_i p(class(i))(sigma^i)
The boundary becomes linear when they uses the same boundary, but with a higher accuracy

The desired probability is a sigmoid function 1/ (1 + e^(-z)) = sigma(z), where z can be simplified as (w*x + b)
Conclusion:
Estimate population mean of each classification, their common covariance matrix, and their population percentage of each classifcation in the whole data set, and then calculate w and b

### Classification 2:
Goodness of function:
L(w,b) = product of i,j f(w,b,x^i) (1- f(w,b,x^j)), where x^i belongs to category 1, and x^j belongs to category 2, and maximize this argument L.
w*,b* = max(w,b)(L(w,b)) = min(w,b)(-ln L(w,b)),
which can then be modeled as calculating the cross entropy between two bernoulli distribution f(w,b) and y^hat^i, where y^hat^i is 1 if data i belongs to class 1, else 0.
L(w,b) = \sum_n [-(y^hat^i)ln f(w,b,x^i) + (1-y^hat^i) ln (1- f(w,b,x^i)]
Regression:
After some math, the results are the same as linear regression
w^{i+1} < - w^{i} - eta \sum_i (-(y^hat^i - f(w,b,x^i))x^i
Why use cross entropy instead of least square error as the loss function??
After some math, we find that points far from the optimal will have zero gradient, which cannot descend and is problematic
Generative vs Discriminative:
Generative "generates" the dependance between data components
Less training data needed, more robust to noise, priors and class-dependent probabilities can be estimated.
Multi-class classification,
z = w * x + b, k, y_k = e^{z_k}/ (\sum_k e^{z_k}), where k is a component of the vector
called a softmax function, then calculate cross entropy between y_m and y^hat_m: \sum_m -y^hat^mlny^m, where m are the classifications.

Limitations:
Logistic regression essentially draws a line between a feature space, cannot perform nicely if the data is strangely distributed. 
Solutions:
Feature Transformation, transform data into another feature space.
But hard to find a good transformation, therefore we have to cascade logistic regression models,
giving different weights to the components, adding a bias vector and feeding it into different logistic regression models to transform to components with new values. 

### Deep learning:
Each logistical regression unit is called a neuron, and with a network structure, it is called a neural network
Neurons: Unit that does logistic regression
Input layer: the representation vector of the data
Hidden layers: the interaction between layers of neurons
Output layer: the representation vector of the outcome
Deep: Multiple hidden layers

Fully connect feedforward network:
Input of neurons in each layer must directly come from the previous layer.
l
Layers required and neurons for each layer: Trial and Error + Intuition
How to determine the network structure: (the relations between each layers)
Convolutional Neural Network (CNN)

Deep learning: A neuron network structure with many hidden layers, with a multi-classifier (softmax) topping it off.

Goodness of deep learning networks:
Same as logical regression, calculate cross entropy of the data, between the distribution of its real classification and the calculated distirbution of the data. C(y,y^hat) = -\sum_k y^hat_kln(y_k), then loss function will be L= \sum_i C^i. Difference will be in calculating y_k.
Gradient Descent:
Still the same, calculate gradient than descent, but with much more parameters (w1,w2,...,b1,b2,...)

### Backpropagation
Backpropagation: Efficient way to compute gradients in neural networks.
Based on chain rule: deltaC/deltaw = deltaC/deltaz * deltaz/deltaw
Forward pass: Compute deltaz/deltaw
Backward pass: Compute deltaC/deltaz
Activation function: the work of the neuron, a = output, z=input
C is the result after the activation function (output after being feeded into the neuron)
so deltaC/deltaw is straightforward, is only the corresponding component of the input into calculating z. (the x^i of the w^ix^i's)
deltaC/deltaz is equals deltaa/deltaz * deltaC/deltaa = sigma'(z) deltaz/delta * deltaC/deltaz' + deltaz''/deltaa * deltaC/deltaz'' = sigma'(z) [w_3 deltaC/deltaz' + w_4 deltaC/deltaz''].
We can easily calculate deltaC/deltaz^*, where z^* is the output at the output layer, therefore can recursively calculate the backward passes.

### Tips

Emphasis on the  three important steps
- Define a set of function (modelling)
- Goodness of function (loss function)
- Picking the best function (regression)

next train the model, then sees if the results is good, then repeat if its not

Do not always blame overfitting, the neural network probably hit some limitations of neural networks

Light layers is based on random data, and probably hit a local minima and converges, therefore is inaccurate for the testing data, but accurate for training data    

Vanishing Gradient Problem:
deltaC/deltaw = change in C/ change in w ??
Nope, since the sigmoid function is between 0 and 1, large changes in inputs results in small changes in outputs, and as the layers become deeper the change is less significant, so the gradient is small.

ReLU:
sigma(z) = a= 0 if z<0, a=z if z>=0
Benefits: fast to compute, biological reasons, infinite sigmoid with different biases, addresses the vanishing gradient problem since it transforms into a thinner linear network
The function is complex enough since it changes its opearation range if the change of the input is significant enough

Parametric ReLU:
sigma(z) = a= alpha * z if z<0, a=z if z>=0, alpha can be learned by treating it as a parameter into the model.

Maxout:
Learnable activation function, achieves the same functionality as ReLU, has different activation areas for different input, but do not have to be an ReLU function. 
a = max{z_1,z_2,...}, ReLU is a special case where z_2 = 0, z_1 = z

RMSProp
Error surfaces are complex for NN, slow learning rates on inaccurate points
w^{t+1} < - w^{t} - eta/sigma^t g^t, sigma^t =  \sqrt(alpha(sigma^{t-1}^2+ (1-alpha)(g^t )^2)

Hard to find optimal network parameters:
No need to worry about local minimum since there is less local minimum for a large network.
Implementing momentum to overcome these difficulties
movement v^{t+1}  = lambda v^{0} - eta * g^t
Move: theta^{t+1} = theta^{t} + v^{t+1}

Adam:
RMSProp + Momentum 

Early stopping:
Stops when validation set/testing set has the lowest error rate at points near the optimum of the training set

Regularization:
L'(theta) = L(theta) + lambda 1/2 ||theta||_2 , ||theta||2 = \sum_i w_i^2
w^{t+1} = (1 - eta* lambda)* w^{t} - eta* g^t

Dropout:
Each neuron has p% to dropout, uses new network for training, changes network structure
Assuming dropout rate is p%, testing weights must be all multiplied be (1-p)%
Train the neurons to perform better even with less neurons -> performs better in total.
The minibatches trained are embedded in the change in parameters of the network, so all training data is still learnt by the neural network.

### Why deep

Modularization, deep neural networks implements modularization and each layer can have a smaller but more specific "task", so more accurate.
Most tasks requires modularization in their origianl tasks, such as speech. Speech is modularized into different acoustic features, phonemes.
Tasks are meant to dealt layer by layer, such as classifying phonemes
More effective use of parameters
Analogy between logical circuits, parity checks

End-to-end learning
Do not specify all functions(tasks) of the neurons, allows the NN to learn its part. 
Eg, speech recognition starting from log, spectrogram.
Paper from google attempted to only specify input and output, but its results tied with the best existing results, not an improvement.

Complex tasks
Machine has to be able to differentiate between objects that are different but have similar inputs, and objects that are same but have different inputs.

### CNN
Taking away some unneccessary parameters in the NN, by adding convolution(crafted) layers between layers. 
Skipping layers between the convultion layer and the neuron network layer
Image processing: 
Not all pixels in an image is neccessary 
Some neurons is doing similar tasks
Apply filters on results to produce multiple images, such as using RGB filters
Done by calculating inner products between each channels (sections of the photo) of inputs
Can reduce parameters used in the NN by reducing parameters while filtering
Max pooling:
Further reduces image size, but contains more values per channel (as much as the number of filters)
(Image is much deeper but smaller)
The machine aims to find input photo x that maximize activation of neurons 
a^k = \sum_i \sum_j a^k_{ij}, where x^* = max_x ( a^k)

Deep style, Deep dream
input -> CNN -> content, input -> CNN -> style,
the CNN that is responsible for generating images aims to find the image that has the most similar content and most similar style.

Playing GO
Using CNN networks allows the machine to perform much better, since chess focuses on the game one pattern/group per time. 

### RNN

Recurrent neural network, a network with a hidden network that stores values.
Produces a different output with the same input (while having a different memory)
Sample: parsing sentences and check whether the word is the time of departure or destination
Changing the sequence order will change the output of the NN
arrive Taipei is different than leave Taipei, luckily the machine stores the word arrive/leave in memory and knows how to differentiate

Elman Network
Value of the output produced by the input of the memory layer will be stored

Jordan Network
Output of NN will be stored

These are samples of LSTM (Long Short-term Memory)
Consists of an input gate, memory cell, output gate, forget gate, together they consist a long short-term memory.
Parameters of the states of gate is also learned by the machine.

f(z_i) input gate, f(z_f) forget gate, f(z_o) output gate, h(c') normal output after activation function of the memory cell,  g(z) normal input value after an activation function
memory c' = g(z) f(z_i) + cf(z_f). c' passes through h(c') then multiplies f(z_o) as the final output value. 
Activation function f is usually a sigmoid to show its openess
More complicated forms use output of LSTM cells in the last iteration as an input of the next iteration
Essentially RNN are neural networks that make use of LSTM cells.

RNN are trained by BPTT, still using regression.
RNN sometimes suddenly broke with high epoch. THis is due to rough error surfaces(very flat or steep). Very steep error surfaces plus a high learning rate will cause parameters to be out of bounds.
This is because the memory propagates through iterations, after n iterations the changes wlil be amplified to the n power. The gradient is therefore very high
Solutions: clippping, limit the range of values of gradients so the paramters will not be out of bounds

LSTM solves gradient vanishing, since the memory is not washed away(unless forget gate is closed). Therefore the forget gate should be ensured that its mostly activated.

Gated Recurrent Unit (GRU)
simpler than LSTM, combines input gate and forget gate. These gates work against each other, when one is open the other is closed

Many to one applications, input is a vector sequence, but output is only one vector.
EG given movie comments, classify its criticism, 
Many to Many applications, both input and output are sequences but output is shorter.
EG speech recognition, used to trimming as the same word will be repeated many times to a machine.
Replace null symbols to solve not able to differentiate intentional repetition in speech

EG Machine translation, sequence to sequence learning. (Translating between languages with different lengths)
Syntactic parsing, recognizing audio, requires encoder and decoder to encode audio segment into a vector.
Encoding: audio segment into acoustic features, then put it into a RNN Encoder for training.
After it uses output of RNN Encoder into RNN Decoder for training

Attention-based model, can forget things like humans if things are insignificant or is too long ago
There is a reading head and writing head controller to write into a machine's memory.
Neural turing machine.

Application: Reading Comprehension, given new information, the reading head controller determines what information to retrieve to give the most accurate answer

Deep vs structured learning
RNN, LSTM can be Deep

HMM, CRF< Structured Perceptron/SVM considers the whole sequence, can explicitly consider label dependency and cost is the upper bound of error
Solutions: Integrate both, use structured as the outer layer and then apply deep neural networks.

