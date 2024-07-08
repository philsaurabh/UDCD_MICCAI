import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim

from utils.losses import KLD, uncertainity_loss
from utils.CRA import CRALoss


def get_accuracy(y_pred, y_actual):
    """Calculates the accuracy (0 to 1)

    Args:
    + y_pred (tensor ): output from the model
    + y_actual (tensor): ground truth 

    Returns:
    + float: a value between 0 to 1
    """
    y_pred = torch.argmax(y_pred, axis=1)
    y_actual = torch.argmax(y_actual, axis=1)
    return (1/len(y_actual))*torch.sum(torch.round(y_pred) == y_actual)


def update_teacher(student, teacher, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(teacher.parameters(), student.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def fit(
        student,
        teacher,
        train_dl,
        test_dl,
        weights,
        class_index,
        logger,
        args,
        device="cpu",
):
    print()
    student = student.to(device)
    teacher = teacher.to(device)

    # Set optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(filter(
            lambda p: p.requires_grad, student.parameters()), lr=args.lr, weight_decay=1e-5)
    elif args.optimizer == "amsgrad":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr, weight_decay=1e-5, amsgrad=True
        )
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-5
        )

    # Set scheduler
    if args.scheduler == 'StepLR':
        sch = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.epochs // 3, gamma=0.1)
    elif args.scheduler == 'CosineAnnealingLR':
        sch = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    elif args.scheduler == 'OneCycleLR':
        sch = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, epochs=args.epochs,
                                            steps_per_epoch=len(train_dl))
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5)

    # Loss Functions
    weights = torch.Tensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    CRA = CRALoss(args).cuda()
    # kld = KLD()

    student.train()
    iter_num = 0

    # early_stop = 0
    # best_loss = 1000.0

    acc_all = np.zeros(10)
    pre_all = np.zeros(10)
    rec_all = np.zeros(10)
    f1_all = np.zeros(10)

    for epoch in range(args.epochs):
        train_running_loss = 0
        train_running_acc = 0

        tqdm_train_iterator = tqdm(enumerate(train_dl),
                                   desc=f"[train]{epoch+1}/{args.epochs}",
                                   ascii=True, leave=True,
                                   total=len(train_dl),
                                   colour="green", position=0,
                                   bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                                   mininterval=10)

        # Training loop
        for batch_idx, ((s_images, t_images), target, index, sample_idx) in tqdm_train_iterator:

            s_images = s_images.to(device)
            t_images = t_images.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            s_ftrs, s_logits = student(s_images)
            t_ftrs, t_logits = teacher(t_images)

            loss = criterion(s_logits, target)

            if epoch >= args.n_distill:
                consistency_weight = args.consistency * \
                    sigmoid_rampup(epoch, args.consistency_rampup)
                consistency_dist = uncertainity_loss(t_logits, s_logits)
                consistency_loss = consistency_weight * consistency_dist

                ccd_loss, relation_loss = CRA(s_ftrs, t_ftrs, index.cuda(
                ), target, class_index, args.nce_p, sample_idx.cuda())
                loss += consistency_loss
                loss += args.ccd_weight * ccd_loss
                loss += args.rel_weight * relation_loss

            loss.backward()

            optimizer.step()
            sch.step()
            update_teacher(student, teacher, args.t_decay, iter_num)
            iter_num += 1

            train_running_loss += loss.item()

            train_running_acc += get_accuracy(s_logits.detach(), target)

            tqdm_train_iterator.set_postfix(avg_train_acc=f"{train_running_acc/(batch_idx+1):0.4f}",
                                            avg_train_loss=f"{(train_running_loss/(batch_idx+1)):0.4f}")

        logger.info('')
        logger.info(f"Epoch: {epoch}")
        acc, pre, rec, f1 = test(
            student, test_dl, logger, verbose=True, device=device)

        if epoch >= args.epochs - 10:
            acc_all[epoch - (args.epochs - 10)] = acc
            pre_all[epoch - (args.epochs - 10)] = pre
            rec_all[epoch - (args.epochs - 10)] = rec
            f1_all[epoch - (args.epochs - 10)] = f1

    logger.info("\nAverage performance of the last 10 epochs:")
    logger.info("\nAccuracy: {:6f}, Precision: {:6f}, Balanced Accuracy: {:6f}, F1: {:6f}"
                .format(np.mean(acc_all), np.mean(pre_all), np.mean(rec_all), np.mean(f1_all)))
    student.eval()
    logger.info(" ** Training complete **")
    print(" ** Training complete **")

# Test the model


def test(
    net,
    test_dl,
    logger,
    verbose=True,
    device="cpu"
):
    net = net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    tqdm_train_iterator = tqdm(enumerate(test_dl),
                               desc="[TEST]",
                               total=len(test_dl),
                               ascii=True, leave=True,
                               colour="green", position=0,
                               bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
                               mininterval=10)

    test_running_loss = 0
    test_running_acc = 0

    actuals = []
    predictions = []

    for idx, (data, target) in tqdm_train_iterator:
        data = data.to(device)
        target = target.to(device)
        _, y_pred = net(data)
        loss = criterion(y_pred, target)

        test_running_loss += loss.item()

        actuals.extend(target.argmax(dim=1).cpu().numpy())
        predictions.extend(y_pred.argmax(dim=1).cpu().numpy())

        test_running_acc += get_accuracy(y_pred.detach(), target)

        tqdm_train_iterator.set_postfix(avg_test_acc=f"{test_running_acc/(idx+1):0.4f}",
                                        avg_test_loss=f"{(test_running_loss/(idx+1)):0.4f}")

    print("Test Loss: ", test_running_loss/len(test_dl))
    actuals = np.array(actuals)
    predictions = np.array(predictions)

    acc = accuracy_score(actuals, predictions)
    pre = precision_score(actuals, predictions, average='macro')
    rec = recall_score(actuals, predictions, average='macro')
    f1 = f1_score(actuals, predictions, average='macro')

    if verbose:
        logger.info("Accuracy: %6f, Precision: %6f, Recall: %6f, F1: %6f \n" %
                    (acc, pre, rec, f1))

        print("Accuracy: %6f, Precision: %6f, Recall: %6f, F1: %6f \n" %
              (acc, pre, rec, f1))

    return acc, pre, rec, f1
