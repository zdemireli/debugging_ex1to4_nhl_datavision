# debugging_ex1to4_nhl_datavision

#excersice_1
from typing import Set,List
def id_to_fruit(fruit_id:int,fruits: Set[str]->str
  sorted_fruits=list(fruits) #convert set to list#
  idx=0
  for fruit in sorted_fruits:
    if fruit_id==idx:
      return fruit
    idx +=1
  raise RuntimeError(f"Fruit with id {fruit_id} does not exist")

#exercis_2
  import numpy as np
  def swap(coords:np.ndarray)->np.array:
    coords_copy=coords.copy() #original array
    coords_copy[:,[0,1,2,3]=coords[:[1,0,3,2]]  #by considering [x1,y1,x2,y2]#
    return coords_copy
#exercise_3
  import csy
  import numpy as np
  import matplot.pyplot as plt
  def plot_data(csv_file_path:str):
    results=[]
    with open(csv_file_path,newline='') as result csv #to prevent empty rows#
    csv_reader=csv.reader(result_csv,delimiter=',')
    next(csv_reader)
    for row in csv_reader:
      row=[cell.strip() for cell in row]
      if len(row)<2:
        continue
       try:
         results.append([floot(row[0]),floot(row[1]))
         except ValueError:
         print(f" incorrect data:{row}")
    if not results:
      raise ValueError ("not found data in CSV file")
    results=np.aaray(results)
    plt.plot(results[:,0],results[:,1])
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.show()

#exercise_4
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
from IPython.display import display, clear_output
class Generator(nn.Module):
    def __init__(self, noise_dim=100, output_dim=784):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(noise_dim, 256),nn.ReLU(),nn.Linear(256,512),nn.ReLU(),nn.Linear(512,1024),nn.ReLU(),nn.Linear(1024,output_dim),nn.Tanh(),)
    def forward(self,x):
        output=self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output
class Discriminator(nn.Module):#leaky is used for better results#
    def __init__(self, input_dim=784):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(input_dim, 1024),nn.LeakyReLU(0.2),nn.Dropout(0.3),nn.Linear(1024,512),nn.LeakyReLU(0.2),nn.Dropout(0.3),nn.Linear(512,256),nn.LeakyReLU(0.2),nn.Dropout(0.3),nn.Linear(256,1),nn.Sigmoid(),)
    def forward(self,x):
        x=x.view(x.size(0),784)
        output=self.model(x)
        return output
        
def train_gan(batch_size: int = 32, num_epochs: int = 100, device: str = "cuda:0" if torch.cuda.is_available() else "cpu"):#32 is used for batch size#
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
  try:
    train_set = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)
   except:
         print("Failed to download MNIST, retrying with different URL")
        
        torchvision.datasets.MNIST.resources = [('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
             'f68b3c2dcbeaaa9fbdd348bbdeb94873'),('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
             'd53e105ee54ea40749a09fcbcd1e9432'),('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
             '9fb629c4189551a2d022fa330f9573f3'),('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
             'ec29112dd5afa0611ce80d1b7f02629c')]
         train_set = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)
         
     train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
     real_samples, mnist_labels = next(iter(train_loader))

    fig = plt.figure()
    for i in range(16):
        sub = fig.add_subplot(4, 4, 1 + i)
        sub.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
        sub.axis('off')

    fig.tight_layout()
    fig.suptitle("Real images")
    display(fig)

    time.sleep(3)

    # Set up training
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    lr = 0.0001
    loss_function = nn.BCELoss()
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    # train
    for epoch in range(num_epochs):
        for n, (real_samples, mnist_labels) in enumerate(train_loader):

            # Data for training the discriminator
            real_samples = real_samples.to(device=device)
            real_batch_size=real_samples.size(0)
            real_samples_labels=torch.ones((real_batch_size,1)).to(device=device) #real batch sized used to prevent discrepencies#
            latent_space_samples = torch.randn((real_batch_size, 100)).to(device=device)
            generated_samples = generator(latent_space_samples)
            generated_samples = generated_samples.view(real_batch_size, -1)
            generated_samples_labels = torch.zeros((real_batch_size, 1)).to(device=device)
            
            all_samples = torch.cat((real_samples.view(real_batch_size, -1), generated_samples))
            all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))
            
            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Data for training the generator
            latent_space_samples = torch.randn((real_batch_size, 100)).to(device=device)  

            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            generated_samples = generated_samples.view(real_batch_size, -1)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()

            # Show loss and samples generated
            if n %100==0: #to prevent unmatch situation at latest batch as multiples of 32# 
                name = f"Generate images\n Epoch: {epoch} Loss D.: {loss_discriminator:.2f} Loss G.: {loss_generator:.2f}"
                generated_samples = generated_samples.detach().cpu().numpy()
                fig = plt.figure()
                for i in range(16):
                    sub = fig.add_subplot(4, 4, 1 + i)
                    sub.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
                    sub.axis('off')
                fig.suptitle(name)
                fig.tight_layout()
                clear_output(wait=True)  #better result#
                display(fig)

             
       

