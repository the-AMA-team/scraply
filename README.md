# Scraply

### A no-code, deep learning platform -- The "Scratch" for Neural Networks

- Previously awarded Best Developer Tool (HackPrinceton'24)
- Watch our [new demo video](https://www.youtube.com/watch?v=QsKWQxAiWhE)
  
### 1) Drag and drop neural network layers. View your model's PyTorch configuration 

<img width="1470" alt="image" src="https://github.com/user-attachments/assets/179d4ed4-3cbc-4199-aeab-05ecf90e6895" />

### 2) Set training parameters - updated live
After training, you can also download a python notebook. See the code for everything you just did on Scraply!

<img width="1894" height="885" alt="image" src="https://github.com/user-attachments/assets/2d2a6e0a-4d96-4e26-acdf-f684970dfdd2" />

### 3) View outputs - includes special visualization with image datasets

<img width="1899" height="925" alt="image" src="https://github.com/user-attachments/assets/c5faa610-af0f-4682-84c5-a991bc2f602c" />

### What are PEEK Maps?:
The heat maps shown when initializing a CNN are generated from a method called PEEK[^1]. It was developed by Mackenzie Meni and the NETS Lab at the Florida Institute of Technology. 
PEEK visualizes neural network decision-making by computing entropy-based maps of convolutional layers. It is able to create heat maps by highlighting the most information-rich regions in the input.
Check out the paper [here](https://doi.org/10.2514/6.2024-2766)!

Some cool PEEK maps from the CIFAR10 image dataset:

<img width="1061" height="792" alt="image" src="https://github.com/user-attachments/assets/d8665169-5657-4e45-b122-6a2dcbb3a3a0" />

### Running locally:

The scraply server isn't deployed yet, therefore you need to run your own backend! We are working on cost-effective and possible funding/sponsor options to allow users to train their Scraply models for free :) 

1. visit [scraply](https://scraply-prod.vercel.app) (server status shows offline)
2. clone github repo `git clone https://github.com/the-AMA-team/scraply.git`
3. go to the api directory `cd scraply/dynamic-model-api/`
4. download python packages `pip install -r requirements.txt`
5. run server `uvicorn app:socket_app --host 0.0.0.0 --port 5000 --reload`

### Updates in Summer'25 Release:

- 👾 explainability features
- 👾 outputs tab with model insights
- 👾 live training graph
- 👾 stop/resume training 
  
### Coming ~~Soon~~ Someday:

- 👾 ability to run in browser (using tf.js)
- 👾 uploading custom datasets and data pre-processing
- 👾 encoder support for transformers

### Developed by the-AMA-team

Alan 🧑‍🍳: Cloud Ops

Mehek 🤓: Backend/AI

Adi 🤩: Frontend/UI


[^1]: M. Meni, T. Mahendrakar, O. D. Raney, R. T. White, M. L. Mayo, and K. R. Pilkiewicz (2024). *Taking a PEEK into YOLOv5 for Satellite Component Recognition via Entropy-based Visual Explanations.* AIAA SCITECH 2024 Forum. [https://arc.aiaa.org/doi/abs/10.2514/6.2024-2766](https://arc.aiaa.org/doi/abs/10.2514/6.2024-2766)
