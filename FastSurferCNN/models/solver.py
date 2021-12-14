# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# IMPORTS
import os
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import itertools
import glob

from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import utils
from skimage import color
from models.losses import CombinedLoss


##
# Helper functions
##
def create_exp_directory(exp_dir_name):
    """
    Function to create a directory if it does not exist yet.
    :param str exp_dir_name: name of directory to create.
    :return: None; creates a directory
    """
    if not os.path.exists(exp_dir_name):
        try:
            os.makedirs(exp_dir_name)
            print("Successfully Created Directory @ {}".format(exp_dir_name))
        except:
            print("Directory Creation Failed - Check Path")
    else:
        print("Directory {} Exists ".format(exp_dir_name))


def dice_confusion_matrix(batch_output, labels_batch, num_classes):
    """
    Function to compute the dice confusion matrix.
    :param torch.Tensor batch_output: a batch of outputs
    :param torch.Tensor labels_batch: a batch of corresponding ground-truth labels
    :param int num_classes: number of classes
    :return: torch.Tensor avg_dice: average dice score
    :return: torch.Tensor dice_cm: dice confusion matrix
    """
    dice_cm = torch.zeros(num_classes, num_classes)

    for i in range(num_classes):
        gt = (labels_batch == i).float()

        for j in range(num_classes):
            pred = (batch_output == j).float()
            inter = torch.sum(torch.mul(gt, pred)) + 0.0001
            union = torch.sum(gt) + torch.sum(pred) + 0.0001
            dice_cm[i, j] = 2 * torch.div(inter, union)

    avg_dice = torch.mean(torch.diagflat(dice_cm))

    return avg_dice, dice_cm


def iou_score(pred_cls, true_cls, nclass=79):
    """
    compute the intersection-over-union score
    both inputs should be categorical (as opposed to one-hot)
    :param torch.Tensor pred_cls: network prediction (categorical)
    :param torch.Tensor true_cls: ground truth (categorical)
    :return: np.ndarray: the intersection array
    :return: np.ndarray: the union array
    """
    intersect_ = []
    union_ = []

    for i in range(1, nclass):
        intersect = ((pred_cls == i).float() + (true_cls == i).float()).eq(2).sum().item()
        union = ((pred_cls == i).float() + (true_cls == i).float()).ge(1).sum().item()
        intersect_.append(intersect)
        union_.append(union)

    return np.array(intersect_), np.array(union_)


def precision_recall(pred_cls, true_cls, nclass=79):
    """
    Function to calculate recall (TP/(TP + FN) and precision (TP/(TP+FP) per class
    :param torch.Tensor pred_cls: network prediction (categorical)
    :param torch.Tensor true_cls: ground truth (categorical)
    :param int nclass: number of classes
    :return:
    :return: np.ndarray: an array of true positives
    :return: np.ndarray: an array of true positives - false negatives
    :return: np.ndarray: an array of true positives - false positives
    """
    tpos_fneg = []
    tpos_fpos = []
    tpos = []

    for i in range(1, nclass):
        all_pred = (pred_cls == i).float()
        all_gt = (true_cls == i).float()

        tpos.append((all_pred + all_gt).eq(2).sum().item())
        tpos_fpos.append(all_pred.sum().item())
        tpos_fneg.append(all_gt.sum().item())

    return np.array(tpos), np.array(tpos_fneg), np.array(tpos_fpos)


##
# Plotting functions
##
def plot_predictions(images_batch, labels_batch, batch_output, plt_title, file_save_name):
    """
    Function to plot predictions from validation set.
    :param torch.Tensor images_batch: batch of images
    :param torch.Tensir labels_batch: batch of output ground-truth labels
    :param torch.Tensor batch_output: batch of output predictions
    :param str plt_title: title of the plot
    :param str file_save_name: the filename of the saved figure
    :return: None
    """

    f = plt.figure(figsize=(20, 20))
    n, c, h, w = images_batch.shape
    mid_slice = c // 2
    images_batch = torch.unsqueeze(images_batch[:, mid_slice, :, :], 1)
    grid = utils.make_grid(images_batch.cpu(), nrow=4)

    plt.subplot(131)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Slices')

    grid = utils.make_grid(labels_batch.unsqueeze_(1).cpu(), nrow=4)[0]
    color_grid = color.label2rgb(grid.numpy(), bg_label=0)
    plt.subplot(132)
    plt.imshow(color_grid)
    plt.title('Ground Truth')

    grid = utils.make_grid(batch_output.unsqueeze_(1).cpu(), nrow=4)[0]
    color_grid = color.label2rgb(grid.numpy(), bg_label=0)
    plt.subplot(133)
    plt.imshow(color_grid)
    plt.title('Prediction')

    plt.suptitle(plt_title)
    plt.tight_layout()

    f.savefig(file_save_name, bbox_inches='tight')
    plt.close(f)
    plt.gcf().clear()


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, file_save_name="temp.pdf"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    :param np.ndarray cm: confusion matrix array
    :param list classes: list of class ids
    :param str title: title of the plot
    :param matplotlib.colors.Colormap cmap: color map
    :param file_save_name: the filename of the saved figure
    :return: None
    """
    f = plt.figure(figsize=(35, 35))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    f.savefig(file_save_name, bbox_inches='tight')
    plt.close(f)
    plt.gcf().clear()


##
# Training routine
##
class Solver(object):
    """
    Class for training neural networks
    """

    # gamma is the factor for lowering the lr and step_size is when it gets lowered
    default_lr_scheduler_args = {"gamma": 0.3,
                                 "step_size": 5}

    def __init__(self, num_classes, optimizer=torch.optim.Adam, optimizer_args={}, loss_func=CombinedLoss(),
                 lr_scheduler_args={}):

        # Merge and update the default arguments - optimizer
        self.optimizer_args = optimizer_args

        lr_scheduler_args_merged = Solver.default_lr_scheduler_args.copy()
        lr_scheduler_args_merged.update(lr_scheduler_args)

        # Merge and update the default arguments - lr scheduler
        self.lr_scheduler_args = lr_scheduler_args_merged

        self.optimizer = optimizer
        self.loss_func = loss_func
        self.num_classes = num_classes
        self.classes = list(range(self.num_classes))

    def train(self, model, train_loader, validation_loader, class_names, num_epochs,
              log_params, expdir, scheduler_type, torch_v11, resume=True):
        """
        Train Model with provided parameters for optimization
        Inputs:
        -- model - model to be trained
        -- train_loader - training DataLoader Object
        -- validation_loader - validation DataLoader Object
        -- num_epochs = total number of epochs
        -- log_params - parameters for logging the progress
        -- expdir --directory to save check points

        """
        create_exp_directory(expdir)  # Experimental directory
        create_exp_directory(log_params["logdir"])  # Logging Directory

        # Instantiate the optimizer class
        optimizer = self.optimizer(model.parameters(), **self.optimizer_args)

        # Instantiate the scheduler class
        if scheduler_type == "StepLR":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.lr_scheduler_args["step_size"],
                                            gamma=self.lr_scheduler_args["gamma"])

            log_params["logger"].info(
                "Scheduler: StepLR, step_size: {}, gamma: {}".format(self.lr_scheduler_args["step_size"],
                                                                     self.lr_scheduler_args["gamma"]))
        else:
            scheduler = None
            log_params["logger"].info("Scheduler: None")

        # Set up logger format
        a = "{}\t" * (self.num_classes - 2) + "{}"

        epoch = -1  # To allow for restoration
        print('-------> Starting to train')

        # Code for restoring model
        if resume:

            try:
                prior_model_paths = sorted(glob.glob(os.path.join(expdir, 'Epoch_*')), key=os.path.getmtime)

                if prior_model_paths:
                    current_model = prior_model_paths.pop()

                state = torch.load(current_model)

                # Restore model dictionary
                model.load_state_dict(state["model_state_dict"])
                optimizer.load_state_dict(state["optimizer_state_dict"])
                if scheduler is not None:
                    scheduler.load_state_dict(state["scheduler_state_dict"])
                epoch = state["epoch"]

                print("Successfully restored the model state. Resuming training from Epoch {}".format(epoch + 1))

            except Exception as e:
                print("No model to restore. Resuming training from Epoch 0. {}".format(e))

        log_params["logger"].info("{} parameters in total".format(sum(x.numel() for x in model.parameters())))

        while epoch < num_epochs:

            epoch = epoch + 1
            epoch_start = time.time()

            # Update learning rate based on epoch number (only for pytorch version <1.2)
            if torch_v11 and scheduler is not None:
                scheduler.step()

            loss_batch = np.zeros(1)

            for batch_idx, sample_batch in enumerate(train_loader):

                # Assign data
                images_batch, labels_batch, weights_batch = sample_batch['image'], sample_batch['label'], \
                                                            sample_batch['weight']

                # Map to variables
                images_batch = Variable(images_batch)
                labels_batch = Variable(labels_batch)
                weights_batch = Variable(weights_batch)

                if torch.cuda.is_available():
                    images_batch, labels_batch, weights_batch = images_batch.cuda(), labels_batch.cuda(), \
                                                                weights_batch.type(torch.FloatTensor).cuda()

                model.train()  # Set to training mode!
                optimizer.zero_grad()

                predictions = model(images_batch)

                loss_total, loss_dice, loss_ce = self.loss_func(predictions, labels_batch, weights_batch)

                loss_total.backward()

                optimizer.step()

                loss_batch += loss_total.item()

                if batch_idx % (len(train_loader) // 2) == 0 or batch_idx == len(train_loader) - 1:
                    log_params["logger"].info("Train Epoch: {} [{}/{}] ({:.0f}%)] "
                                              "with loss: {}".format(epoch, batch_idx,
                                                                     len(train_loader),
                                                                     100. * batch_idx / len(train_loader),
                                                                     loss_batch / (batch_idx + 1)))

                del images_batch, labels_batch, weights_batch, predictions, loss_total, loss_dice, loss_ce

            # Update learning rate at the end based on epoch number (only for pytorch version > 1.1)
            if not torch_v11 and scheduler is not None:
                scheduler.step()

            epoch_finish = time.time() - epoch_start

            log_params["logger"].info("Train Epoch {} finished in {:.04f} seconds.".format(epoch, epoch_finish))
            # End of Training, time to accumulate results

            # Testing Loop on Training Data
            # Set evaluation mode on the model
            model.eval()

            val_loss_total = 0
            val_loss_dice = 0
            val_loss_ce = 0

            ints_ = np.zeros(self.num_classes - 1)
            unis_ = np.zeros(self.num_classes - 1)
            per_cls_counts_gt = np.zeros(self.num_classes - 1)
            per_cls_counts_pred = np.zeros(self.num_classes - 1)
            accs = np.zeros(self.num_classes - 1)  # -1 to exclude background (still included in val loss)

            with torch.no_grad():

                if validation_loader is not None:

                    val_start = time.time()
                    cnf_matrix_validation = torch.zeros(self.num_classes, self.num_classes)

                    for batch_idx, sample_batch in enumerate(validation_loader):

                        images_batch, labels_batch, weights_batch = sample_batch['image'], sample_batch['label'], \
                                                                    sample_batch['weight']

                        # Map to variables (no longer necessary after pytorch 0.40)
                        images_batch = Variable(images_batch)
                        labels_batch = Variable(labels_batch)
                        weights_batch = Variable(weights_batch)

                        if torch.cuda.is_available():
                            images_batch, labels_batch, weights_batch = images_batch.cuda(), labels_batch.cuda(), \
                                                                        weights_batch.type(torch.FloatTensor).cuda()

                        # Get logits, sum up batch loss and get final predictions (argmax)
                        predictions = model(images_batch)
                        loss_total, loss_dice, loss_ce = self.loss_func(predictions, labels_batch, weights_batch)
                        val_loss_total += loss_total.item()
                        val_loss_dice += loss_dice.item()
                        val_loss_ce += loss_ce.item()

                        _, batch_output = torch.max(predictions, dim=1)

                        # Calculate iou_scores, accuracy and dice confusion matrix + sum over previous batches
                        int_, uni_ = iou_score(batch_output, labels_batch, self.num_classes)
                        ints_ += int_
                        unis_ += uni_

                        tpos, pcc_gt, pcc_pred = precision_recall(batch_output, labels_batch, self.num_classes)
                        accs += tpos
                        per_cls_counts_gt += pcc_gt
                        per_cls_counts_pred += pcc_pred

                        _, cm_batch = dice_confusion_matrix(batch_output, labels_batch, self.num_classes)
                        cnf_matrix_validation += cm_batch.cpu()

                        # Plot sample predictions
                        if batch_idx == 0:
                            plt_title = 'Validation Results Epoch ' + str(epoch)

                            file_save_name = os.path.join(log_params["logdir"],
                                                          'Epoch_' + str(epoch) + '_Validations_Predictions.pdf')

                            plot_predictions(images_batch, labels_batch, batch_output, plt_title, file_save_name)

                        del images_batch, labels_batch, weights_batch, predictions, batch_output, \
                            int_, uni_, tpos, pcc_gt, pcc_pred, loss_total, loss_dice, loss_ce  # cm_batch,

                    # Get final measures and log them
                    ious = ints_ / unis_
                    val_loss_total /= (batch_idx + 1)
                    val_loss_dice /= (batch_idx + 1)
                    val_loss_ce /= (batch_idx + 1)
                    cnf_matrix_validation = cnf_matrix_validation / (batch_idx + 1)
                    val_end = time.time() - val_start

                    print("Completed Validation Dataset in {:0.4f} s".format(val_end))

                    save_name = os.path.join(log_params["logdir"], 'Epoch_' + str(epoch) + '_Validation_Dice_CM.pdf')
                    plot_confusion_matrix(cnf_matrix_validation.cpu().numpy(), self.classes, file_save_name=save_name)

                    # Log metrics
                    log_params["logger"].info("[Epoch {} stats]: MIoU: {:.4f}; "
                                              "Mean Recall: {:.4f}; "
                                              "Mean Precision: {:.4f}; "
                                              "Avg loss total: {:.4f}; "
                                              "Avg loss dice: {:.4f}; "
                                              "Avg loss ce: {:.4f}".format(epoch, np.mean(ious),
                                                                           np.mean(accs / per_cls_counts_gt),
                                                                           np.mean(accs / per_cls_counts_pred),
                                                                           val_loss_total, val_loss_dice, val_loss_ce))

                    log_params["logger"].info(a.format(*class_names))
                    log_params["logger"].info(a.format(*ious))

            # Saving Models

            if epoch % log_params["log_iter"] == 0:
                save_name = os.path.join(expdir, 'Epoch_' + str(epoch).zfill(2) + '_training_state.pkl')
                checkpoint = {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                              "epoch": epoch}
                if scheduler is not None:
                    checkpoint["scheduler_state_dict"] = scheduler.state_dict()

                torch.save(checkpoint, save_name)

            model.train()
