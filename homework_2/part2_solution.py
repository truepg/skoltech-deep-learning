# Don't erase the template code, except "Your code here" comments.

import torch
# Your code here...
from torch import nn

from torch.utils.data import DataLoader, Dataset

from torchvision.datasets import ImageFolder

import torchvision.models as models
import torchvision.transforms as transforms

from torch.optim import Adam 

from tqdm.notebook import tqdm

from torch.utils.tensorboard import SummaryWriter

import tensorboard

import os
logs_base_dir = "./logs"
os.makedirs(logs_base_dir, exist_ok=True)

def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val', the dataloader should be deterministic.
    
    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train' or 'val'
        
    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    # Your code here
    if kind == 'val':
      params = {'batch_size': 128}

      dataset = ImageFolder(path + kind, transform=transforms.ToTensor())

      return DataLoader(dataset, **params)
    else:
      params = {'batch_size': 128,
                'shuffle': True,
                'num_workers': 2}

      dataset = ImageFolder(path + kind, transform=transforms.Compose([transforms.RandomAffine(15),
                                                                       transforms.RandomRotation(10),
                                                                       transforms.RandomHorizontalFlip(),
                                                                       transforms.ToTensor()]))
      return DataLoader(dataset, **params)

def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.
    
    return:
    model:
        `torch.nn.Module`
    """
    # Your code here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.densenet201()
    model = model.to(device)

    return model

def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.
    
    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    # Your code here
    return Adam(model.parameters(), lr=1e-3)

def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).
    
    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    # Your code here
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch = batch.to(device)
    pred = model(batch)
    
    return pred

def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.
    
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    # Your code here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_true_predicted = 0.0
        for i, data in enumerate(dataloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
    
            prediction = outputs.argmax(axis=1)
            val_true_predicted += (prediction == labels).sum().item()
    
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
        val_loss /= dataloader.dataset.__len__()
        val_accuracy = val_true_predicted / dataloader.dataset.__len__()

    return val_accuracy, val_loss
    

def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.
    
    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    # Your code here
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    model_name = type(model).__name__
    custom_str = 'CHECKING'
    exp_name = model_name + '_lr_' + str(lr) + '_batch_size_' + str(train_dataloader.batch_size) + custom_str

    writer = SummaryWriter(log_dir='./logs/' + exp_name)

    n_epoch = 50
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in tqdm(range(n_epoch)):
      model.train()
      train_loss = 0.0
      train_true_predicted = 0.0
      for i, data in enumerate(train_dataloader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        prediction = outputs.argmax(axis=1)
        train_true_predicted += (prediction == labels).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

      train_loss /= train_dataloader.dataset.__len__()
      train_accuracy = train_true_predicted / train_dataloader.dataset.__len__()

      torch.save(model.state_dict(), './logs/' + exp_name + '/checkpoint' + str(epoch) + '.pth')

      val_accuracy, val_loss = validate(val_dataloader, model)

      writer.add_scalar('Train Loss', train_loss, global_step=epoch)
      writer.add_scalar('Train Accuracy', train_accuracy, global_step=epoch)
      writer.add_scalar('Validation Loss', val_loss, global_step=epoch)
      writer.add_scalar('Validation Accuracy', val_accuracy, global_step=epoch)

def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.
    
    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    # Your code here
    model.load_state_dict(torch.load(checkpoint_path))

def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    # Your code here;
    # Your code here; 
    google_drive_link = "https://drive.google.com/file/d/1QTtgeTap1WNjcApGKWZ0TNdAyNeIGv29/view?usp=sharing"
    md5_checksum = "0b94d3e92b887a4111b45e97ffcfb6ec"
    
    return md5_checksum, google_drive_link
