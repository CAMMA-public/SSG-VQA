"""
Project: Advancing Surgical VQA with Scene Graph Knowledge
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
import os
import glob
import h5py
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SSGVQAClassification_full_roi_analysis(Dataset):
    """
    seq: train_seq  = ['1','2','3','4','6','7','8','9','10','13','14','15','16','18','20',
                      '21','22','23','24','25','28','29','30','32','33','34','35','36','37','38','39','40']
         val_seq    = ['5','11','12','17','19','26','27','31']
    folder_head     = 'dataset/Cholec80-VQA/Classification/'
        folder_tail     = '/*.txt'
        patch_size      = 1/2/3/4/5
    """

    def __init__(self, seq, folder_head, folder_tail, ana_type, patch_size=4):
        self.patch_size = patch_size
        self.folder_head = folder_head

        # files, question and answers
        filenames = []
        for curr_seq in seq:
            filenames = filenames + glob.glob(
                folder_head + "qa_txt/" + str(curr_seq) + folder_tail
            )

        new_filenames = filenames

        self.vqas = []
        for file in new_filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for idx, line in enumerate(lines):
                if idx >= 2 and line.count("|") >= 3:
                    ll = line.split("|")
                    t1 = ll[2]
                    t2 = ll[3]
                    if t1 in ana_type or t2 in ana_type:
                        self.vqas.append([file, line])
        print(
            "Total files: %d | Total question: %.d"
            % (len(new_filenames), len(self.vqas))
        )

        # labels
        self.labels = [
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

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        vid = self.vqas[idx][0].split("/")[3]
        fid = self.vqas[idx][0].split("/")[-1]

        visual_feature_pix = os.path.join(
            self.folder_head,
            "visual_feats",
            "cropped_images",
            vid,
            "vqa",
            "img_features",
            "1x1",
            "%06d" % int(fid.split(".txt")[0]) + ".hdf5",
        )
        frame_data_pix = h5py.File(visual_feature_pix, "r")
        visual_features_pix = torch.from_numpy(frame_data_pix["visual_features"][:])

        visual_feature_loc = os.path.join(
            self.folder_head,
            "visual_feats",
            "roi_yolo_coord",
            vid,
            "labels",
            "vqa",
            "img_features",
            "roi",
            "%06d" % int(fid.split(".txt")[0]) + ".hdf5",
        )

        frame_data = h5py.File(visual_feature_loc, "r")
        visual_features = torch.from_numpy(frame_data["visual_features"][:])

        visual_features[:, 18:] = visual_features_pix

        # question and answer
        question = self.vqas[idx][1].split("|")[0]
        label = self.labels.index(str(self.vqas[idx][1].split("|")[1]))
        return "", visual_features, question, label


class SSGVQAClassification_full_roi_coord(Dataset):
    """
    seq: train_seq  = ['1','2','3','4','6','7','8','9','10','13','14','15','16','18','20',
                      '21','22','23','24','25','28','29','30','32','33','34','35','36','37','38','39','40']
         val_seq    = ['5','11','12','17','19','26','27','31']
    folder_head     = './data/'
        folder_tail     = '/*.txt'
        patch_size      = 1/2/3/4/5
    """

    def __init__(self, seq, folder_head, folder_tail, patch_size=4):
        self.patch_size = patch_size
        self.folder_head = folder_head

        # files, question and answers
        filenames = []
        for curr_seq in seq:
            filenames = filenames + glob.glob(
                folder_head + "qa_txt/" + str(curr_seq) + folder_tail
            )

        new_filenames = filenames

        self.vqas = []
        for file in new_filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for idx, line in enumerate(lines):
                if idx >= 2:
                    self.vqas.append([file, line])
        print(
            "Total files: %d | Total question: %.d"
            % (len(new_filenames), len(self.vqas))
        )

        # labels
        self.labels = [
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

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        vid = self.vqas[idx][0].split("/")[3]
        fid = self.vqas[idx][0].split("/")[-1]

        visual_feature_pix = os.path.join(
            self.folder_head,
            "visual_feats",
            "cropped_images",
            vid,
            "vqa",
            "img_features",
            "1x1",
            "%06d" % int(fid.split(".txt")[0]) + ".hdf5",
        )

        frame_data_pix = h5py.File(visual_feature_pix, "r")
        visual_features_pix = torch.from_numpy(frame_data_pix["visual_features"][:])

        visual_feature_loc = os.path.join(
            self.folder_head,
            "visual_feats",
            "roi_yolo_coord",
            vid,
            "labels",
            "vqa",
            "img_features",
            "roi",
            "%06d" % int(fid.split(".txt")[0]) + ".hdf5",
        )

        frame_data = h5py.File(visual_feature_loc, "r")
        visual_features = torch.from_numpy(frame_data["visual_features"][:])

        visual_features[:, 18:] = visual_features_pix

        # question and answer
        question = self.vqas[idx][1].split("|")[0]
        label = self.labels.index(str(self.vqas[idx][1].split("|")[1]))

        return "", visual_features, question, label
