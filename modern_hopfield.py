import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

class Hopfield_Net:
    """
    Implements Modern Hopfield Network
    Takes dimension and beta as inputs for initialization.
    """
    def __init__(self,dim, beta):
        self.state = torch.zeros(dim) 
        self.weights = torch.zeros(dim) 
        self.beta = beta
        
       
    def add_memories(self, memory): 
        # Memory with 0 in all the entires cannot be stored as the first memory.
        if torch.equal(self.weights, torch.zeros(memory.shape[0])):
            self.weights = memory  
        else:
            self.weights = torch.vstack((self.weights, memory))


    def update(self): #update network
       """returns False if the update changed the weights"""
       old = self.state
       softmax = torch.nn.Softmax(dim=0)
       prob = softmax(self.beta*torch.matmul(self.weights,self.state))
       self.state = torch.matmul((torch.transpose(self.weights, 0, 1)), prob)
       
       return False if torch.equal(old, self.state) else True

    def retrieve(self, input, max_epoch):
        """Uses update() to retrieve a memory given an input"""
        self.state = input
        k = 0 
        while (k<max_epoch) and self.update():
           k += 1
        return self.state
    
#############################################################################################

def mnist_prototypes():
    """Prepares MNIST data by loading them from torchvision library 
    and creating a prototype digit"""
    # Define the transformation to convert images to PyTorch tensors
    tfm = transforms.Compose([transforms.ToTensor()])
    # Load the MNIST dataset with the specified transformation
    train = datasets.MNIST(root='./data', train=True, download=True, transform=tfm)
    X = torch.stack([train[i][0].reshape(-1) for i in range(len(train))])  # (N, 784)
    y = torch.tensor([train[i][1] for i in range(len(train))])             # (N,)
    # Build class prototypes (one memory per digit)
    prototypes = torch.zeros(10, 784)
    for digit in range(10):
        prototypes[digit] = X[y == digit].mean(dim=0)
    return prototypes

prototypes = mnist_prototypes()

#############################################################################################
def noising(input, noise_magnitude):
    """
    Input: one dimensional tensor
    
    Returns a noised input
    """
    noise = torch.randn(input.shape[0])
    return input*(1-noise_magnitude) + noise_magnitude * noise

def train_hop(dim, beta):
    """Returns a hopfield network with given dimension and beta. 
    MNIST digit memories are stored."""
    hopfield = Hopfield_Net(dim, beta)
    for digit in range (10):
        hopfield.add_memories(prototypes[digit])
    return hopfield

def plot_show(imgs, noise):
    """
    Displays the images in imgs.

    Takes in a list imgs of size 20 where imgs[i] is the 
    noised input when i is even and retrieved input when i is odd
    Noise level is to display the plot. 
    """
    n = len(imgs)//2
    fig, axes = plt.subplots(n,2,figsize=(5,8), constrained_layout=True)
    for i in range(n):
        axes[i,0].imshow(imgs[2*i].reshape(28,28), cmap='gray')
        axes[i,0].set_title("Noisy: " + str(i))
        axes[i,0].axis('off')

        axes[i,1].imshow(imgs[2*i+1].reshape(28,28), cmap='gray')
        axes[i,1].set_title("Retrieved: " + str(i))
        axes[i,1].axis('off')
    fig.suptitle(f"Noisy (Noise Level: {noise}) vs. Retrieved")
    plt.tight_layout

def display_result(x, y, title, xlabel, ylabel):
    """Displays the plot of y vs. x given title, xlabel, ylabel"""
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(x)

def evaluate_retrieval(hopfield, result, index, max_epoch, noise):
    """Evaluates the accuracy of the hopfield retrieval

    param result: 1-D tensor where each entry represents the 
    accuracy of the hopfield network under certain condition. 
    See functions const_beta, varying_noise.
    """
    for digit in range(10):
        noised_data = noising(prototypes[digit], noise)
        retrieved_data = hopfield.retrieve(noised_data, max_epoch)
        if torch.equal(retrieved_data, prototypes[digit]):
            result[index] +=1

#Set beta and max_epoch constant, and see how accuracy falls as noise_level increases
def const_beta(hopfield, variation, max_epoch):
    """Evaluates the accuracy of hopfield network as the 
    noise level of images increase from 0 (not included) to 1.

    param variation: number of different noise levels being tested.
    """
    result = torch.zeros(variation)
    step = 1 / variation 
    for index in range(variation):
        noise = round(step*(index+1), 1)
        evaluate_retrieval(hopfield, result, index, max_epoch, noise)
    return result/10

def varying_noise(beta = 80, max_epoch=10):
    """
    Evaluates the accuracy of the hopfield network of beta 80 and max_epoch 
    under varying noise level.

    Displays the final result in a plot. 
    """
    hopfield = train_hop(784,beta)
    rate = const_beta(hopfield, 10, max_epoch)
    noise_level = torch.arange(0.1,1.1,0.1)
    title = f"Beta: {beta} | Max_Epoch: {max_epoch}\nAccuracy vs. Noise Level"
    xlabel = "Noise-level"
    ylabel = "Accuracy"
    display_result(noise_level, rate, title, xlabel, ylabel)

#Set noise level constant, max_epoch constant, and see how accuracy rises as beta increases
def collect_result(start, end, step, noise, max_epoch):
    """Evaluates the accuracy of the hopfield network of varying beta and fixed max_epoch 
    for a fixed noise level.

    Beta varies from start (not included) to end increasing by step. 
    """
    n = len(range(start,end,step))
    result = torch.zeros(n)
    for index in range(n):
        beta = index*step+start
        hopfield = train_hop(784,beta)
        evaluate_retrieval(hopfield, result, index, max_epoch, noise)
    return result/10

def varying_beta(start, end, step, noise, max_epoch):
    """Evaluates hopfield network of varying beta and displays the result"""
    result = collect_result(start, end, step, noise, max_epoch)
    index = torch.arange(start, end, step)
    title = f"Noise: {noise} | Max_Epoch: {max_epoch}\nAccuracy vs. Beta"
    xlabel = "Beta"
    ylabel = "Accuracy"
    display_result(index, result, title, xlabel, ylabel)

#Set noise level constant, beta constant, and see how accuracy changes as max_epoch increases. 
def collect_result_epoch(start, end, step, noise, beta):
    """
    Evaluates the accuracy of hopfield network as max epoch increases while 
    holding beta and noise level constant

    Max epoch increases from start (not included) to end by step size.
    """
    n = len(range(start,end,step))
    result = torch.zeros(n)
    hopfield = train_hop(784, beta)
    for index in range(n):
        max_epoch = index*step + start
        evaluate_retrieval(hopfield, result, index, max_epoch, noise)
    return result/10

def varying_epoch(start, end, step, noise, beta):
    """
    Evaluates the accuracy of hopfield network of varying max epoch.
    Displays the result.
    """
    result = collect_result_epoch(start, end, step, noise, beta)
    index = torch.arange(start, end, step)
    title = f"Noise: {noise} | Beta: {beta}\nAccuracy vs. Max_Epoch"
    xlabel = "Max_Epoch"
    ylabel = "Accuracy"
    display_result(index, result, title, xlabel, ylabel)

#Visualizing noised digits and retrieval.
def visualization():
    """
    Sets beta, noise level, and max epoch using user input. 
    Displays the noised digits and hopfield retrievals.
    """
    beta = float(input("Beta: "))
    noise = float(input("Noise level: "))
    max_epoch = int(input("Max Epoch: "))
    hopfield = train_hop(784, beta)
    imgs = []
    for digit in range(10):
        noised_data = noising(prototypes[digit], noise)
        retrieved_data = hopfield.retrieve(noised_data, max_epoch)
        imgs.append(noised_data)
        imgs.append(retrieved_data)
    plot_show(imgs, noise)

# Below are the final results users can play around with.

varying_noise()
varying_beta(10, 100, 10, 0.1, 10)
varying_epoch(2, 22, 2, 0.1, 100)
visualization()
plt.show()

