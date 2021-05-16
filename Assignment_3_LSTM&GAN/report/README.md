## Instructions to run the code
### Part 1
- Use the script in Part 1_new.ipynb as follows with specific arguments:
```
%matplotlib inline
%run train.py --input_length 5
```
- Or use command line to run the code
- Or set the argument parser by hand and invoke the `train(config)` method
### Part 2
- Run the main in `my_gan.py` with specific argument to train the GAN model
- The trained generator model will be saved in `mnist_generator.pt` under the same directory as `my_gan.py`
- In `Part 2.ipynb`, there are code blocks to load the trained generator model:
```
generator = Generator(latent_dim=100)
generator.load_state_dict(torch.load('mnist_generator.pt'))
```
- To reproduce the interpolation result, load the tensor of `0.pt` and `9.pt`
and do interpolation using the following script to generate intermediate latent space vector 
as well as the images.
```
v0 = torch.load('0.pt')
v9 = torch.load('9.pt')
noise_lst = []
num = 9
for i in range(num):
    noise = (i * v0 + (num-i) * v9)/num
    noise_lst.append(noise)

fig,axis = plt.subplots(1,num,figsize=(100,15))

for i in range(num):
    img = generator(noise_lst[i]).squeeze()
    axis[i].imshow(img.detach().numpy())

plt.show()
```