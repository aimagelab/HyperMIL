"""Training and evaluation loop utilities for HyperMIL."""

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from lookhead import Lookahead
from models.channels_hypergraph import HypergraphFeatureExtractor
from models.hypermil import HyperMIL
from models.timemil import TimeMIL


def collate_fn(batch):
    """Pad variable-length sequences in a batch."""
    if len(batch[0]) == 2:
        sequences, labels = zip(*batch)
        lengths = None
    else:
        sequences, labels, _ = zip(*batch)
        lengths = [seq.size(0) for seq in sequences]
    sequences = [torch.as_tensor(seq) if isinstance(seq, np.ndarray) else seq for seq in sequences]
    return pad_sequence(sequences, batch_first=True), torch.stack(labels), lengths


def _build_optimizer(model: nn.Module, args):
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return Lookahead(optimizer)


def _build_scheduler(optimizer, args):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    if args.scheduler == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=10)
    return None


def train_loop(trainset, testset, num_classes, seq_len, args, logger, weights=None, wandb_logger=None):
    """Run training and return best validation metrics list."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.autocast and device.type == "cuda")

    criterion = (
        nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
        if args.loss == "cross_entropy"
        else nn.BCEWithLogitsLoss(weight=weights.to(device) if weights is not None else None)
    )

    hg_feat_extr = HypergraphFeatureExtractor(
        M=args.feats_size,
        out_dim=args.embed,
        d=args.intra_embed,
        K=args.k_prototypes,
        num_conv=args.num_convs,
        tau=args.tau,
        time_decay=args.time_decay,
        activity_gate=args.activity_gate,
        identity=args.intra_identity,
    ).to(device)
    timemil = TimeMIL(
        args.feats_size,
        mDim=args.embed,
        n_classes=args.num_classes,
        dropout=args.dropout_node,
        max_seq_len=seq_len,
        encoding=args.encoding,
        pooling=args.pooling,
        num_layers=args.num_layers,
    ).to(device)
    milnet = HyperMIL(intra_hg=hg_feat_extr, timemil=timemil).to(device)

    optimizer = _build_optimizer(milnet, args)
    scheduler = _build_scheduler(optimizer, args)

    trainloader = DataLoader(trainset, args.batchsize, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    testloader = DataLoader(testset, 128, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    save_path = os.path.join(args.save_dir, "weights")
    os.makedirs(save_path, exist_ok=True)
    best_score, results_best = -1.0, None

    for epoch in range(1, args.num_epochs + 1):
        train_loss = train(trainloader, milnet, criterion, optimizer, epoch, args, device, amp_enabled)
        test_loss, results = test(testloader, milnet, criterion, args, device, amp_enabled)
        avg_score = results[0]

        logger.info(f"Epoch [{epoch}/{args.num_epochs}] train={train_loss:.4f} test={test_loss:.4f} acc={avg_score:.4f}")
        if scheduler is not None:
            scheduler.step(avg_score) if args.scheduler == "plateau" else scheduler.step()

        if avg_score >= best_score:
            best_score, results_best = avg_score, results
            torch.save(milnet.state_dict(), os.path.join(save_path, "best_model.pth"))

    return results_best


def train(trainloader, milnet, criterion, optimizer, epoch, args, device, amp_enabled):
    scaler = GradScaler()
    milnet.train()
    total_loss = 0.0

    for feats, label, lengths in trainloader:
        bag_feats, bag_label = feats.to(device), label.to(device)
        if args.dropout_patch > 0:
            selected = random.sample(range(10), int(args.dropout_patch * 10))
            interval = max(1, bag_feats.shape[1] // 10)
            for idx in selected:
                st, en = idx * interval, min((idx + 1) * interval, bag_feats.shape[1])
                bag_feats[:, st:en, :] = torch.randn_like(bag_feats[:, st:en, :])

        optimizer.zero_grad()
        with autocast(enabled=amp_enabled):
            bag_prediction, _ = milnet(bag_feats, lengths, warmup=epoch < args.epoch_des)
            bag_loss = criterion(bag_prediction, bag_label)

        scaler.scale(bag_loss).backward()
        if args.unscale:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(milnet.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += bag_loss.item()

    return total_loss / max(1, len(trainloader))


def test(testloader, milnet, criterion, args, device, amp_enabled):
    milnet.eval()
    total_loss = 0.0
    labels_list, probs_list = [], []

    with torch.no_grad():
        for feats, label, lengths in testloader:
            bag_feats, bag_label = feats.to(device), label.to(device)
            with autocast(enabled=amp_enabled):
                bag_prediction, _ = milnet(bag_feats, lengths)
                loss = criterion(bag_prediction, bag_label)
            total_loss += loss.item()
            probs_list.append(torch.sigmoid(bag_prediction).cpu().numpy())
            labels_list.append(label.cpu().numpy())

    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    test_labels = labels.argmax(axis=1)
    test_predictions = probs.argmax(axis=1)

    avg_score = accuracy_score(test_labels, test_predictions)
    balanced_avg_score = balanced_accuracy_score(test_labels, test_predictions)
    f1_macro = f1_score(test_labels, test_predictions, average="macro")
    f1_micro = f1_score(test_labels, test_predictions, average="micro")
    p_macro = precision_score(test_labels, test_predictions, average="macro", zero_division=0)
    p_micro = precision_score(test_labels, test_predictions, average="micro", zero_division=0)
    r_macro = recall_score(test_labels, test_predictions, average="macro", zero_division=0)
    r_micro = recall_score(test_labels, test_predictions, average="micro", zero_division=0)
    war = recall_score(test_labels, test_predictions, average="weighted", zero_division=0)
    cm = confusion_matrix(test_labels, test_predictions)
    uar = float(np.mean([cm[i, i] / max(1, cm[i].sum()) for i in range(len(cm))]))

    def _safe_auc(*auc_args, **auc_kwargs):
        try:
            return roc_auc_score(*auc_args, **auc_kwargs)
        except ValueError:
            return float("nan")

    if args.num_classes == 2:
        pos_scores = probs[:, 1].ravel()
        auc_bin = _safe_auc(test_labels, pos_scores)
        roc_auc_ovo_macro = roc_auc_ovo_micro = roc_auc_ovr_macro = roc_auc_ovr_micro = auc_bin
    else:
        y_true_bin = label_binarize(test_labels, classes=list(range(args.num_classes)))
        roc_auc_ovo_macro = _safe_auc(y_true_bin, probs, average="macro", multi_class="ovo")
        roc_auc_ovo_micro = _safe_auc(y_true_bin, probs, average="micro", multi_class="ovo")
        roc_auc_ovr_macro = _safe_auc(y_true_bin, probs, average="macro", multi_class="ovr")
        roc_auc_ovr_micro = _safe_auc(y_true_bin, probs, average="micro", multi_class="ovr")

    results = [
        avg_score,
        balanced_avg_score,
        f1_macro,
        f1_micro,
        p_macro,
        p_micro,
        r_macro,
        r_micro,
        roc_auc_ovo_macro,
        roc_auc_ovo_micro,
        roc_auc_ovr_macro,
        roc_auc_ovr_micro,
        uar,
        war,
    ]
    return total_loss / max(1, len(testloader)), results
