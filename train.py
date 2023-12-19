"""
Project: Advancing Surgical VQA with Scene Graph Knowledge
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
import os
import argparse
import pandas as pd
from lib2to3.pytree import convert

from torch import nn
from torch import optim
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer
from torch.utils.data import DataLoader

from utils.utils import *
from utils.dataloaderClassification import *
from models.VisualBertClassification_ssgqa import VisualBertClassification
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


"""
Seed randoms
"""


def seed_everything(seed=27):
    """
    Set random seed for reproducible experiments
    Inputs: seed number
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(
    args, train_dataloader, model, criterion, optimizer, epoch, tokenizer, device
):
    model.train()

    total_loss = 0.0
    label_true = None
    label_pred = None
    label_score = None

    import time

    for i, (_, visual_features, q, labels) in enumerate(train_dataloader, 0):
        # prepare questions
        questions = []
        for question in q:
            questions.append(question)
        inputs = tokenizer(
            questions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.question_len,
        )

        # GPU / CPU
        visual_features = visual_features.to(device)
        labels = labels.to(device)
        if args.transformer_ver == "pure_language":
            outputs = model(inputs)
        else:
            outputs = model(inputs, visual_features)

        loss = criterion(outputs, labels)

        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        total_loss += loss.item()

        scores, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
        label_true = (
            labels.data.cpu()
            if label_true == None
            else torch.cat((label_true, labels.data.cpu()), 0)
        )
        label_pred = (
            predicted.data.cpu()
            if label_pred == None
            else torch.cat((label_pred, predicted.data.cpu()), 0)
        )
        label_score = (
            scores.data.cpu()
            if label_score == None
            else torch.cat((label_score, scores.data.cpu()), 0)
        )

    # loss and acc
    acc, c_acc = calc_acc(label_true, label_pred), calc_classwise_acc(
        label_true, label_pred
    )
    precision, recall, fscore = calc_precision_recall_fscore(label_true, label_pred)
    print(
        "Train: epoch: %d loss: %.6f | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f"
        % (epoch, total_loss, acc, precision, recall, fscore)
    )
    return acc


def validate(
    args, val_loader, model, criterion, epoch, tokenizer, device, save_output=False
):
    model.eval()

    total_loss = 0.0
    label_true = None
    label_pred = None
    label_score = None
    file_names = list()

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, (file_name, visual_features, q, labels) in enumerate(val_loader, 0):
            # prepare questions
            questions = []
            for question in q:
                questions.append(question)
            inputs = tokenizer(
                questions,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=args.question_len,
            )

            # GPU / CPU
            visual_features = visual_features.to(device)
            labels = labels.to(device)

            if args.transformer_ver == "pure_language":
                outputs = model(inputs)
            else:
                outputs = model(inputs, visual_features)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            scores, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
            label_true = (
                labels.data.cpu()
                if label_true == None
                else torch.cat((label_true, labels.data.cpu()), 0)
            )
            label_pred = (
                predicted.data.cpu()
                if label_pred == None
                else torch.cat((label_pred, predicted.data.cpu()), 0)
            )
            label_score = (
                scores.data.cpu()
                if label_score == None
                else torch.cat((label_score, scores.data.cpu()), 0)
            )
            for f in file_name:
                file_names.append(f)

    mAP, mAR, mAf1, wf1, acc = eval_for_f1_et_all(label_true, label_pred)
    c_acc = 0.0

    print(
        "Test: epoch: %d loss: %.6f | Acc: %.6f | mAP: %.6f | mAR: %.6f | mAf1: %.6f | wf1: %.6f"
        % (epoch, total_loss, acc, mAP, mAR, mAf1, wf1)
    )

    if save_output:
        """
        Saving predictions
        """
        if os.path.exists(args.checkpoint_dir + "text_files") == False:
            os.mkdir(args.checkpoint_dir + "text_files")
        file1 = open(args.checkpoint_dir + "text_files/labels.txt", "w")
        file1.write(str(label_true))
        file1.close()

        file1 = open(args.checkpoint_dir + "text_files/predictions.txt", "w")
        file1.write(str(label_pred))
        file1.close()

        if "ssg-qa" in args.dataset_type:
            convert_arr = [
                "0",
                "1",
                "10",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "False",
                "True",
                "abdominal_wall_cavity",
                "adhesion",
                "anatomy",
                "aspirate",
                "bipolar",
                "blood_vessel",
                "blue",
                "brown",
                "clip",
                "clipper",
                "coagulate",
                "cut",
                "cystic_artery",
                "cystic_duct",
                "cystic_pedicle",
                "cystic_plate",
                "dissect",
                "fluid",
                "gallbladder",
                "grasp",
                "grasper",
                "gut",
                "hook",
                "instrument",
                "irrigate",
                "irrigator",
                "liver",
                "omentum",
                "pack",
                "peritoneum",
                "red",
                "retract",
                "scissors",
                "silver",
                "specimen_bag",
                "specimenbag",
                "white",
                "yellow",
            ]
        else:
            raise NotImplementedError

        df = pd.DataFrame(columns=["Img", "Ground Truth", "Prediction"])
        for i in range(len(label_true)):
            df = df.append(
                {
                    "Img": file_names[i],
                    "Ground Truth": convert_arr[label_true[i]],
                    "Prediction": convert_arr[label_pred[i]],
                },
                ignore_index=True,
            )

        df.to_csv(
            args.checkpoint_dir
            + args.checkpoint_dir.split("/")[1]
            + "_"
            + args.checkpoint_dir.split("/")[2]
            + "_eval.csv"
        )
    return (acc, c_acc, mAP, mAR, mAf1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSGQA VQA Classification")

    # Model parameters
    parser.add_argument(
        "--emb_dim", type=int, default=300, help="dimension of word embeddings."
    )
    parser.add_argument("--n_heads", type=int, default=8, help="Multi-head attention.")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument(
        "--encoder_layers",
        type=int,
        default=6,
        help="the number of layers of encoder in Transformer.",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="number of epochs to train for (if early stopping is not triggered).",
    )  # 80, 26
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="for data-loading; right now, only 1 works with h5pys.",
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=100,
        help="print training/validation stats every __ batches.",
    )

    parser.add_argument(
        "--checkpoint", default=None, help="path to checkpoint, None if none."
    )
    # existing checkpoint

    parser.add_argument(
        "--lr", type=float, default=0.000005, help="0.000005, 0.00001, 0.000005"
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="./checkpoints/final_vb/",
        help="med_vqa_c$version$/m18/c80//m18_vid$temporal_size$/c80_vid$temporal_size$",
    )  # clf_v1_2_1x1/med_vqa_c3
    parser.add_argument(
        "--dataset_type",
        default="ssg-qa-roi_coord",
        help="med_vqa/m18/c80/m18_vid/c80_vid/ssg-qa-full/img/roi/rot_coord",
    )
    parser.add_argument("--dataset_cat", default="None", help="cat1/cat2/cat3")
    parser.add_argument(
        "--transformer_ver",
        default="vb",
        help="vb/vbrm/two/vbrm_visual/pure_visual/pure_language",
    )
    parser.add_argument("--tokenizer_ver", default="v2", help="v2/v3")
    parser.add_argument("--patch_size", default=1, help="1/2/3/4/5")
    parser.add_argument("--temporal_size", default=3, help="1/2/3/4/5")
    parser.add_argument("--question_len", default=77, help="25")
    parser.add_argument("--num_class", default=2, help="25")
    parser.add_argument(
        "--validate", default=False, help="When only validation required False/True"
    )
    parser.add_argument(
        "--analysis_type",
        default=["exist"],
        help="Which type of question are evaluated",
    )

    args = parser.parse_args()

    # load checkpoint, these parameters can't be modified
    final_args = {
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "encoder_layers": args.encoder_layers,
    }

    seed_everything()

    # GPU or CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    print("device =", device)

    # best model initialize
    start_epoch = 1
    best_epoch = [0]
    best_results = [0.0]
    epochs_since_improvement = 0

    # # dataset

    if args.dataset_type == "ssg-qa-roi_coord":
        """
        Train and test for cholec dataset
        """
        # tokenizer
        if args.tokenizer_ver == "v2":
            tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif args.tokenizer_ver == "v3":
            tokenizer = BertTokenizer.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT", do_lower_case=True
            )
        elif args.tokenizer_ver == "v4":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # dataloader
        train_seq = [
            "VID73",
            "VID40",
            "VID62",
            "VID42",
            "VID29",
            "VID56",
            "VID50",
            "VID78",
            "VID66",
            "VID13",
            "VID52",
            "VID06",
            "VID36",
            "VID05",
            "VID12",
            "VID26",
            "VID68",
            "VID32",
            "VID49",
            "VID65",
            "VID47",
            "VID04",
            "VID23",
            "VID79",
            "VID51",
            "VID10",
            "VID57",
            "VID75",
            "VID25",
            "VID14",
            "VID15",
            "VID08",
            "VID80",
            "VID27",
            "VID70",
        ]
        val_seq = ["VID18", "VID48", "VID01", "VID35", "VID31"]
        test_seq = ["VID22", "VID74", "VID60", "VID02", "VID43"]  #

        folder_head = "./data/"
        folder_tail = "/*.txt"

        # dataloader
        train_dataset = SSGVQAClassification_full_roi_coord(
            train_seq, folder_head, folder_tail, patch_size=args.patch_size
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        val_dataset = SSGVQAClassification_full_roi_coord(
            val_seq, folder_head, folder_tail, patch_size=args.patch_size
        )
        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )

        test_dataset = SSGVQAClassification_full_roi_coord(
            test_seq, folder_head, folder_tail, patch_size=args.patch_size
        )
        test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        # num_classes
        args.num_class = 51

    elif args.dataset_type == "ssg-qa-roi-analysis":
        """
        Train and test for cholec dataset
        """

        if args.tokenizer_ver == "v2":
            tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif args.tokenizer_ver == "v3":
            tokenizer = BertTokenizer.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT", do_lower_case=True
            )

        # dataloader
        test_seq = ["VID22", "VID74", "VID02", "VID60", "VID43"]

        folder_head = "./data/"
        folder_tail = "/*.txt"

        # dataloader
        test_sets = [
            SSGVQAClassification_full_roi_analysis(
                test_seq,
                folder_head,
                folder_tail,
                ana_type=["zero_hop.json"],
                patch_size=args.patch_size,
            ),
            SSGVQAClassification_full_roi_analysis(
                test_seq,
                folder_head,
                folder_tail,
                ana_type=["one_hop.json"],
                patch_size=args.patch_size,
            ),
            SSGVQAClassification_full_roi_analysis(
                test_seq,
                folder_head,
                folder_tail,
                ana_type=["single_and.json"],
                patch_size=args.patch_size,
            ),
            SSGVQAClassification_full_roi_analysis(
                test_seq,
                folder_head,
                folder_tail,
                ana_type=["query_color", "query_type", "query_location"],
                patch_size=args.patch_size,
            ),
            SSGVQAClassification_full_roi_analysis(
                test_seq,
                folder_head,
                folder_tail,
                ana_type=["query_component"],
                patch_size=args.patch_size,
            ),
            SSGVQAClassification_full_roi_analysis(
                test_seq,
                folder_head,
                folder_tail,
                ana_type=["exist"],
                patch_size=args.patch_size,
            ),
            SSGVQAClassification_full_roi_analysis(
                test_seq,
                folder_head,
                folder_tail,
                ana_type=["count"],
                patch_size=args.patch_size,
            ),
        ]
        test_dataloader = [
            DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
            for test_dataset in test_sets
        ]
        # num_classes
        args.num_class = 51

    # model
    if args.transformer_ver == "vb":
        print("loading VisualBert")
        model = VisualBertClassification(
            vocab_size=len(tokenizer),
            layers=args.encoder_layers,
            n_heads=args.n_heads,
            num_class=args.num_class,
        )
    else:
        raise NotImplementedError

    # Initialize / load checkpoint
    if args.checkpoint is None:
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        print("loading from checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location=str(device))
        start_epoch = checkpoint["epoch"]
        epochs_since_improvement = checkpoint["epochs_since_improvement"]
        best_Acc = checkpoint["Acc"]
        model_dict = checkpoint["model"]
        model.load_state_dict(model_dict)
        optimizer = checkpoint["optimizer"]
        final_args = checkpoint["final_args"]
        for key in final_args.keys():
            args.__setattr__(key, final_args[key])

    # Move to GPU, if available
    model = model.to(device)
    print(final_args)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("model params: ", pytorch_total_params)
    # print(model)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # validation
    if args.validate:
        if "analysis" in args.dataset_type:
            for i in test_dataloader:
                (
                    test_acc,
                    test_c_acc,
                    test_precision,
                    test_recall,
                    test_fscore,
                ) = validate(
                    args,
                    val_loader=i,
                    model=model,
                    criterion=criterion,
                    epoch=(args.epochs - 1),
                    tokenizer=tokenizer,
                    device=device,
                    save_output=True,
                )
        else:
            test_acc, test_c_acc, test_precision, test_recall, test_fscore = validate(
                args,
                val_loader=test_dataloader,
                model=model,
                criterion=criterion,
                epoch=(args.epochs - 1),
                tokenizer=tokenizer,
                device=device,
                save_output=True,
            )
    else:
        for epoch in range(start_epoch, args.epochs):
            print(epoch)

            if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
                adjust_learning_rate(optimizer, 0.8)

            # train
            train_acc = train(
                args,
                train_dataloader=train_dataloader,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                epoch=epoch,
                tokenizer=tokenizer,
                device=device,
            )

            # validation
            test_acc, test_c_acc, test_precision, test_recall, test_fscore = validate(
                args,
                val_loader=test_dataloader,
                model=model,
                criterion=criterion,
                epoch=epoch,
                tokenizer=tokenizer,
                device=device,
            )

            if test_acc >= best_results[0]:
                epochs_since_improvement = 0

                best_results[0] = test_acc
                best_epoch[0] = epoch
                print(
                    "Best epoch: %d | Best acc: %.6f" % (best_epoch[0], best_results[0])
                )
                save_clf_checkpoint(
                    args.checkpoint_dir,
                    epoch,
                    epochs_since_improvement,
                    model,
                    optimizer,
                    best_results[0],
                    final_args,
                )

            else:
                epochs_since_improvement += 1
                print(
                    "\nEpochs since last improvement: %d\n"
                    % (epochs_since_improvement,)
                )

            if train_acc >= 1.0:
                break
