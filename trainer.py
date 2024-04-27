import numpy as np
import torch
from higher import innerloop_ctx as metaloop

import torch.nn.functional as F


class trainer:
    def __init__(self, model, train_dataloader, test_dataloader, X_validation, y_validation, device):
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.X_val = X_validation
        self.y_val = y_validation

    def train_normal(self, epochs):
        '''trains the model using traditional approach'''
        # criterion= torch.nn.CrossEntropyLoss()
        criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        running_loss_per_epoch = []
        test_stats = []
        for epoch in range(epochs):
            self.model.train()
            total_num = 0
            running_loss = 0
            train_acc = 0
            running_loss_per_batch = []
            for i, data in enumerate(self.train_dataloader):
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)

                predicted_logits = self.model(images)
                pred_labels = (F.sigmoid(predicted_logits) > 0.5).int()
                loss = criterion(predicted_logits, labels.type_as(predicted_logits))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                train_acc += torch.sum(torch.eq(pred_labels, labels)).item()
                running_loss_per_batch.append(loss.item())
                total_num += len(images)

            running_loss_per_epoch.append(np.mean(running_loss_per_batch))
            print("[epoch: %d] loss: %.3f    train accuracy: %.3f" \
                  % (epoch + 1, running_loss_per_epoch[-1], (train_acc / total_num) * 100), end=' ')
            test_stat = self.test()
            test_stats.append(test_stat['accuracy'])

        return test_stats

    def test(self):
        criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.model.eval()  # setting the model in evaluation mode
        val_stat = {}
        all_accuracy = []
        all_loss = []
        all_predictions = []
        all_labels = []
        total_num = 0
        with torch.no_grad():
            running_loss_per_batch = []
            for images, labels in self.test_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                pred_logits = self.model(images).float()
                labels_tensor = labels.clone().detach()
                pred_labels = (F.sigmoid(pred_logits) > 0.5).int()
                loss = criterion(pred_logits, labels_tensor.type_as(pred_logits))
                accuracy = torch.sum(torch.eq(pred_labels, labels)).item()
                running_loss_per_batch.append(
                    loss.item())  # tracking the loss, accuracy, predicted labels, and true labels
                all_accuracy.append(accuracy)
                all_predictions.append(pred_labels)
                all_labels.append(labels)
                total_num += len(images)

        val_stat['loss'] = np.mean(running_loss_per_batch)
        val_stat['accuracy'] = sum(all_accuracy) / total_num
        val_stat['prediction'] = torch.cat(all_predictions, dim=0)
        val_stat['labels'] = torch.cat(all_labels, dim=0)
        print(
            f"Test: loss: {val_stat['loss']:.3f} acc: {100 * val_stat['accuracy']:.3f}%")
        return val_stat  # returning the tracked values in the form of a dictionary

    def train_reweighted(self, epochs):
        '''trains the model using reweighted sampling approach'''
        criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        running_loss_per_epoch = []
        test_stats = []
        for epoch in range(epochs):
            self.model.train()
            train_acc = 0
            total_num = 0
            running_loss = 0
            running_loss_per_batch = []

            for i, data in enumerate(self.train_dataloader):
                # L2-L3: get data
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                # Inner loop for reweighted training
                with metaloop(self.model, optimizer) as (inner_model, inner_optimizer):
                    # L4-L5: 1. forward pass to compute the initial weighted loss
                    inner_train_outputs = inner_model(images)
                    cost = criterion(inner_train_outputs, labels.type_as(inner_train_outputs))

                    eps = torch.zeros(cost.size(), requires_grad=True, device=self.device)  # sample weights
                    inner_trainloss = torch.sum(eps * cost)
                    # L6-L7: model parameter update
                    inner_optimizer.step(inner_trainloss)

                    # L8-L10: computes validation loss and gradients for reweighting
                    _criterion = torch.nn.BCEWithLogitsLoss(reduction='mean').to(self.device)
                    val_loss = 0
                    val_images, val_labels = self.X_val.to(device=self.device), self.y_val.to(device=self.device)
                    val_pred_logits = inner_model(val_images)
                    loss = _criterion(val_pred_logits, val_labels.type_as(val_pred_logits))
                    eps_grads = (torch.autograd.grad(loss, eps, allow_unused=True)[0].detach())

                # L11: Compute weights for reweighting based on validation gradients
                w_tilde = torch.clamp(-eps_grads, min=0)
                norm = torch.sum(w_tilde)
                w = w_tilde / norm if norm != 0 else w_tilde

                # L12-L14: Perform standard training step using reweighted loss
                predicted_logits = self.model(images)
                pred_labels = (F.sigmoid(predicted_logits) > 0.5).int()
                criterion.reduction = 'none'
                loss = criterion(predicted_logits, labels.type_as(predicted_logits))
                loss = torch.sum(w * loss)

                # Update variables
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                train_acc += torch.sum(torch.eq(pred_labels, labels)).item()
                running_loss_per_batch.append(loss.item())
                total_num += len(images)

            running_loss_per_epoch.append(np.mean(running_loss_per_batch))
            print("[epoch: %d] loss: %.3f    train accuracy: %.3f" % (
            epoch + 1, running_loss_per_epoch[-1], (train_acc / total_num) * 100), end=' ')

            test_stat = self.test()
            test_stats.append(test_stat['accuracy'])

        return test_stats
