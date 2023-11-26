from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
import os
from PIL import Image
from gradcam import *
from gradcam.utils import *
from torchvision.models import vgg16, squeezenet1_1, alexnet, resnet101, densenet161
import torch.nn as nn
import torch
import random as random
import matplotlib.pyplot as plt
import io

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import sklearn

import seaborn as sns

def show_grad_cam(preds, targets, paths, model, device, predicted_label = 'normal', true_label = 'normal', max_images = 10):
    
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    
    if predicted_label == 'normal':
        pred_label = 0
    elif predicted_label == 'pneumonia':
        pred_label = 1
    else:
        pred_label = 2
        
    if true_label == 'normal':
        target_label = 0
    elif true_label == 'pneumonia':
        target_label = 1
    else:
        target_label = 2

    examples = np.logical_and(preds == pred_label, targets == target_label)
    examples_indices = np.nonzero(examples)
    examples_paths = [paths[i] for i in examples_indices[0]]
        
    n_images = min(len(examples_paths),max_images)

    images    =  [do_grad_cam(x, model, device) for x in random.sample(examples_paths, n_images)]

    plt.figure(figsize=(30,15))
    columns = 5
    for i, image in enumerate(images):
        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(image)

def read_log_data(log_file):
    
    log_file = 'log_train.txt'
    
    log_mat = np.loadtxt(log_file, skiprows = 1)
    
    # put together values
    log_data = {"epoch": log_mat[:,0],
                "train loss": log_mat[:,1],
                "validation loss": log_mat[:,2],
                "train accuracy": log_mat[:,3],
                "validation accuracy": log_mat[:,4],
                "recall_normal": log_mat[:,5],
                "recall_pneumonia": log_mat[:,6],
                "recall_covid": log_mat[:,7]
               }
    
    return log_data

# Read file-list from textfile
def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data

def plot_confusion_matrix(conf_matrix):
    ax= plt.subplot()
    sns.heatmap(conf_matrix, annot=True, ax = ax, cmap = 'Blues'); 

    ax.set_xlabel('Vorhergesagte Klasse');ax.set_ylabel('Wahre Klasse'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['normal', 'pneumonia', 'COVID']); ax.yaxis.set_ticklabels(['normal', 'pneumonia', 'COVID']);

def log_metrics_to_file(path, numbers, epoch):
    
    if epoch == 0:
        with open(path, 'w') as file:
            file.write("epoch train_loss validation_loss train_accuracy validation_accuracy recall_normal recall_pneumonia recall_covid\n")
        
    with open(path, 'a') as file:
        for n in numbers:
            file.write('%f ' % n)
        file.write("\n")

# Eine Funktion, die ein Modell (Netzwerk) und einen Testdatensatz erhält, auf dem sie einige
# Maße bestimmt, die uns sagen, wie gut das Netzwerk auf dem Testdatensatz funktioniert
def compute_metrics(model, test_loader, device):
    
    # Da wir das Netzwerk nicht trainieren, sondern nur auswerten ("evaluate") wollen, teilen wir
    # dem Netzwerk mit. Für manche Layer (z.B. Batchnorm) macht das nämlich einen Unterschied.
    model.eval()
    
    val_loss = 0
    val_correct = 0
    
    # Eine Fehlerfunktion, die für alle Trainingsdatenpunkte bestimmt, wie groß die von Netzwerk vorhergesagte 
    # Wahrscheinlichkeit der *wahren* Klasse ist (bzw. der negative Logarithmus davon, der dann *minimiert* wird). 
    # Gute Erkärung dazu findet ihr hier: https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
    criterion = nn.CrossEntropyLoss()
    
    # Wir merken uns für jeden Datenpunkt, welche Wahrscheinlichkeiten ("p") vom Netzwerk für die einzelnen Klassen 
    # vorhergesagt wurden 
    score_list   = torch.Tensor([]).to(device)
    
    # Ausserdem merken wir uns, was die Vorhersage des Netzwerks für die Klasse 
    # der einzelnen Datenpunkte ist (d.h. welche Klasse die höchste Wahrscheinlichkeit hat)
    pred_list    = torch.Tensor([]).to(device).long()
    
    # Und wir merken uns die tatsächliche Klasse der Datenpunkte
    target_list  = torch.Tensor([]).to(device).long()
    
    # Damit wir die Bilder später auch beispielhaft darstellen können, merken wir uns ausserdem die entsprechenden
    # Dateinamen
    path_list    = []

    # Wir gehen einmal durch den gesamten Testdatensatz, den die Funktion übergeben bekommen hat
    for iter_num, data in enumerate(test_loader):
        
        # Convert image data into single channel data
        image, target = data['img'].to(device), data['label'].to(device)
        paths = data['paths']
        path_list.extend(paths)
        
        # Compute the loss
        with torch.no_grad():
            output = model(image)
        
        # Log loss
        val_loss += criterion(output, target.long()).item()

        # Calculate the number of correctly classified examples
        pred = output.argmax(dim=1, keepdim=True)
        val_correct += pred.eq(target.long().view_as(pred)).sum().item()
        
        # Bookkeeping 
        score_list   = torch.cat([score_list, nn.Softmax(dim = 1)(output)[:,1].squeeze()])
        pred_list    = torch.cat([pred_list, pred.squeeze()])
        target_list  = torch.cat([target_list, target.squeeze()])
        
    
    classification_metrics = classification_report(target_list.tolist(), pred_list.tolist(),
                                                  target_names = ['normal', 'pneumonia', 'COVID'],
                                                  output_dict= True)
    
    
    # sensitivity is the recall of the positive class
    recall_covid = classification_metrics['COVID']['recall']
    
    # specificity is the recall of the negative class 
    recall_normal = classification_metrics['normal']['recall']
    
    recall_pneumonia = classification_metrics['pneumonia']['recall']
    
    # accuracy
    accuracy = classification_metrics['accuracy']
    
    # confusion matrix
    conf_matrix = confusion_matrix(target_list.tolist(), pred_list.tolist())
    
    # put together values
    metrics_dict = {"accuracy": accuracy,
                    "metrics": classification_metrics,
                    "recall_normal": recall_normal,
                    "recall_covid": recall_covid,
                    "recall_pneumonia": recall_pneumonia,
                    "confusion matrix": conf_matrix,
                    "validation loss": val_loss / len(test_loader),
                    "score_list":  score_list.tolist(),
                    "pred_list": pred_list.tolist(),
                    "target_list": target_list.tolist(),
                    "paths": path_list}
    
    
    return metrics_dict

from collections import deque

class EarlyStopping(object):
    def __init__(self, patience = 8):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.previous_loss = int(1e8)
        self.previous_accuracy = 0
        self.init = False
        self.accuracy_decrease_iters = 0
        self.loss_increase_iters = 0
        self.best_running_accuracy = 0
        self.best_running_loss = int(1e7)
    
    def add_data(self, model, loss, accuracy):
        
        # compute moving average
        if not self.init:
            running_loss = loss
            running_accuracy = accuracy 
            self.init = True
        
        else:
            running_loss = 0.2 * loss + 0.8 * self.previous_loss
            running_accuracy = 0.2 * accuracy + 0.8 * self.previous_accuracy
        
        # check if running accuracy has improved beyond the best running accuracy recorded so far
        if running_accuracy < self.best_running_accuracy:
            self.accuracy_decrease_iters += 1
        else:
            self.best_running_accuracy = running_accuracy
            self.accuracy_decrease_iters = 0
        
        # check if the running loss has decreased from the best running loss recorded so far
        if running_loss > self.best_running_loss:
            self.loss_increase_iters += 1
        else:
            self.best_running_loss = running_loss
            self.loss_increase_iters = 0
        
        # log the current accuracy and loss
        self.previous_accuracy = running_accuracy
        self.previous_loss = running_loss        
        
    
    def stop(self):
        
        # compute thresholds
        accuracy_threshold = self.accuracy_decrease_iters > self.patience
        loss_threshold = self.loss_increase_iters > self.patience
        
        
        # return codes corresponding to exhuaustion of patience for either accuracy or loss 
        # or both of them
        if accuracy_threshold and loss_threshold:
            return 1
        
        if accuracy_threshold:
            return 2
        
        if loss_threshold:
            return 3
        
        
        return 0
    
    def reset(self):
        # reset
        self.accuracy_decrease_iters = 0
        self.loss_increase_iters = 0

# Just writes the list "my_list" to a textfile given by "path"
def write_list_to_textfile(my_list, path):

    with open(path, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)

# Helferfunktion, um Listen mit Dateinamen in Trainings-,
# Validierungs-, und Testset zu teilen
def split_list(filenames, training, val, test):
    
    sum = training + val + test
    training = training / sum
    val = val / sum
    test = test / sum
    
    num_files = len(filenames)
    
    random.shuffle(filenames)
    
    end_training = int(num_files*training)
    end_test = int(num_files*(training+test))
    
    filenames_training = filenames[:end_training]
    filenames_val = filenames[end_test:]
    filenames_test = filenames[end_training:end_test]
    
    return filenames_training, filenames_val, filenames_test

def true_fun(X):
    return np.cos(1.5 * np.pi * X)

# modified from https://scikit-learn.org/0.15/auto_examples/plot_underfitting_overfitting.html
def plot_overfitting_demo(degrees):
    
    np.random.seed(0)

    n_samples = 30

    X = np.sort(np.random.rand(n_samples))
    y = true_fun(X) + np.random.randn(n_samples) * 0.3

    plt.figure(figsize=(14, 5))
    for i in range(len(degrees)):
        ax = plt.subplot(1, len(degrees), i + 1)
        plt.setp(ax, xticks=(), yticks=())

        polynomial_features = PolynomialFeatures(degree=degrees[i],
                                                 include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                             ("linear_regression", linear_regression)])
        pipeline.fit(X[:, np.newaxis], y)
        
        yhat = pipeline.predict(X[:, np.newaxis])
        
        mse  = sklearn.metrics.mean_squared_error(y, yhat)
  
        # Evaluate the models using crossvalidation
        scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                                 scoring="neg_mean_squared_error", cv=10)

        X_test = np.linspace(0, 1, 100)
        plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Polynom")
        plt.plot(X_test, true_fun(X_test), label="Wahre Funktion")
        plt.scatter(X, y, edgecolor='b', s=20, label="Stichprobe")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim((0, 1))
        plt.ylim((-2, 2))
        plt.legend(loc="best")
        plt.title("Grad {}\n Mittlerer quadratischer \nFehler auf Trainingsdaten:\n{:.2f}\n Mittlerer quadratischer \nFehler auf neuen Daten:\n{:.2f}(+/- {:.2f})".format(
            degrees[i], mse, -scores.mean(), scores.std()))
    plt.show()

def simulate_gradient_descent_on_quadratic_potential(theta_start, learning_rate, n_steps):
    # Startwert für theta
    theta = theta_start

    # Stellen wir uns weiter vor, die Kostenfunktion hängt quadratisch von Theta ab 
    def cost(theta):

        cost = theta*theta;

        return cost

    # Dann können wir den Gradienten der Kostenfunktion direkt berechnen: 
    # Dieser ist die Ableitung der Kostenfunktion nach Theta
    def grad_cost(theta):
        grad = 2*theta

        return grad

    # Hier speichern wir den Verlauf der Parameter während des Gradientenabstiegs
    thetas = []
    # Hier speichern wir den Verlauf der Kostenfunktion während des Gradientenabstiegs
    costs = []

    for i in range(n_steps):

        # Speichere aktuelle Position
        thetas.append(theta)
        # Speichere aktuellen Wert der Kostenfunktion
        costs.append(cost(theta))

        # Mache einen Schritt entgegen dem Gradienten, gewichtet mit der Lernrate
        theta = theta - learning_rate*grad_cost(theta)

    theta_max = max(abs(theta_start),abs(thetas[-1]))
    xrange = np.arange(-theta_max,theta_max,0.01*theta_max)
    cost_line = cost(xrange)

    plt.figure(figsize=(20,10))    
    plt.subplot(1,2,1)
    for i in range(n_steps-1):
        plt.arrow(thetas[i],costs[i],thetas[i+1]-thetas[i],costs[i+1]-costs[i],head_width=0.1*theta_max, head_length=0.15*theta_max, length_includes_head = True, color = 'darkorange', overhang = 0.5)

    # Zeichne Kostenfunktion
    plt.plot(xrange, cost_line)
    # Zeichne Punkte
    plt.scatter(thetas,costs,s = 300.0, c=range(n_steps),cmap='jet',)

    plt.plot(thetas, costs)
    plt.xlabel('theta',fontsize='xx-large')
    plt.ylabel('Wert der Zielfunktion',fontsize='xx-large')

    plt.subplot(1,2,2)
    plt.plot(costs)
    plt.scatter(range(n_steps),costs,s = 300.0, c=range(n_steps),cmap='jet',)
    plt.xlabel('Optimierungsschritt',fontsize='xx-large')
    plt.ylabel('Wert der Zielfunktion',fontsize='xx-large')

def predict_imagenet_class_from_upload(vgg, uploader, top_n, device):
    for name, file_info in uploader.value.items():
        img = Image.open(io.BytesIO(file_info['content'])).convert('RGB')
        img_t = val_transformer(img)
        plt.imshow(img_t.permute(1,2,0))
        batch_t = torch.unsqueeze(img_t, 0).to(device)

        vgg.eval()
        out = vgg(batch_t)
        
        print_imagenet_predictions(out,top_n)


def print_imagenet_predictions(out,top_n):

    # Wir laden die Textdatei, die die Zuordnung zwischen Klassenindex (entspricht der Zeile in der Textdatei)
    # und dem Namen der Klasse enthält
    with open('imagenet_classes.txt') as f:
      classes = [line.strip() for line in f.readlines()]
    
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

    _, indices = torch.sort(out, descending=True)
    
    for idx in indices[0][:top_n]:
        print('Prediction: ' + classes[idx] + ('\nProbability: %f\n' % percentage[idx].item()))

### Nimmt den Pfad zum Bilderordner und eine Liste mit Dateinamen
### und zeigt 5 zufällige Beispiele an
def show_examples(data_path, filenames):

    files      = [os.path.join(data_path, x) for x in filenames]
    images    =  [np.asarray(Image.open(x)) for x in random.sample(files, 5)]

    plt.figure(figsize=(20,10))
    columns = 5
    for i, image in enumerate(images):
        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        if len(image.shape) == 2: # Einkanalbilder in Graustufen anzeigen, sonst erhält man ein ziemlich buntes Farbschema
            plt.imshow(image, cmap = 'gray')
        else:
            plt.imshow(image)

class ImageDataset(Dataset):
    def __init__(self, root_dir, classes, files_path, transform=None):
        self.root_dir = root_dir
        self.classes = classes
        self.files_path = files_path
        self.image_list = []

        #read the files from data split text files
        #covid_files = read_txt(covid_files)
        #normal_files = read_txt(normal_files)
        #pneumonia_files = read_txt(pneumonia_files)

        # combine the positive and negative files into a cummulative files list
        for cls_index in range(len(self.classes)):
            
            class_files = [[os.path.join(self.root_dir, x), cls_index] \
                            for x in self.files_path[cls_index]]
            self.image_list += class_files
                
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path = self.image_list[idx][0]
        
        # Read the image
        image = Image.open(path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = int(self.image_list[idx][1])

        data = {'img':   image,
                'label': label,
                'paths' : path}

        return data
    
    
normalize = transforms.Normalize(mean=[0,0,0], std=[1,1,1])

train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize
])

def do_grad_cam(path, model, device):
        
    if model.name == 'alexnet':
        gradcam = GradCAM.from_config(arch=model, model_type='alexnet', layer_name='features_11')
    elif model.name == 'vgg':
        gradcam = GradCAM.from_config(arch=model, model_type='vgg', layer_name='features_29')
    elif model.name == 'resnet': 
        gradcam = GradCAM.from_config(arch=model, model_type='resnet', layer_name='layer4')
    elif model.name == 'densenet':
        gradcam = GradCAM.from_config(arch=model, model_type='densenet', layer_name='features_norm5')
    elif model.name == 'squeezenet':
        gradcam = GradCAM.from_config(arch=model, model_type = 'squeezenet', layer_name = 'features_12_expand3x3_activation')
        
    # Initialise the grad cam object. 
    # we use model.features as the feature extractor and use the layer no. 35 for gradients. 
    #gradcam, gradcam_pp = cam_dict(model.name)
    
    # read in the image, and prepare it for the network
    orig_im = cv2.imread(path)
    img = Image.fromarray(orig_im)
    inp = val_transformer(img).unsqueeze(0).to(device)

    # main inference
    mask, _ = gradcam(inp)
    
    mask = mask.cpu()

    heatmap, result = visualize_cam(mask, inp)
    # create the heatmap 
    #heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    #heatmap = np.float32(heatmap) / 255
    
    #add the heatmap to the original image
    #cam = heatmap + np.float32(cv2.resize(orig_im, (224,224))/255.)
    #cam = cam / np.max(cam)
    
    # BGR -> RGB since OpenCV operates with BGR values. 
    heatmap = heatmap.permute(1,2,0)
    result = result.permute(1,2,0)
    
    return result

# Importiere vortrainierte Netzwerkarchitekturen
# C.f. https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
# C.f. https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

def get_pretrained_model(model_name, num_classes = 3, pretrained = False):
    
    if model_name == 'vgg':
        model = vgg16(pretrained=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        model.name = 'vgg'
    elif model_name == 'squeezenet':
        model = squeezenet1_1(pretrained=pretrained)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.name = 'squeezenet'
    elif model_name == 'resnet':
        model = resnet101(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.name = 'resnet'
    elif model_name == 'alexnet':
        model = alexnet(pretrained=pretrained)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs,num_classes)
        model.name = 'alexnet'
    elif model_name == "densenet":
        model = densenet161(pretrained=pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model.name = 'densenet'
        
    return model