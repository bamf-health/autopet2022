import SimpleITK
import glob
import numpy as np
from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)

import SimpleITK as sitk
import time
import os

import subprocess
import shutil
from pathlib import Path


# from nnunet.inference.predict import predict_from_folder
# from predict import predict_from_folder
# # from nnunet.paths import default_plans_identifier, network_training_output_dir, default_cascade_trainer, default_trainer
# from batchgenerators.utilities.file_and_folder_operations import join, isdir
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
import torch

os.environ["nnUNet_raw_data_base"] = "nnUNet_raw_data_base/"
os.environ["RESULTS_FOLDER"] = "nnUNet_trained_models/"
os.environ["nnUNet_preprocessed"] = "nnUNet_preprocessed/"
os.environ["MKL_THREADING_LAYER"] = "GNU"

network_training_output_dir = "nnUNet_trained_models/"


class Autopet(SegmentationAlgorithm):
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        # set some paths and parameters
        self.input_path = (
            "/input/"  # according to the specified grand-challenge interfaces
        )
        self.output_path = "/output/images/automated-petct-lesion-segmentation/"  # according to the specified grand-challenge interfaces
        self.nii_path = "nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs/"
        self.result_path = "Task001_TCIA/"
        self.nii_seg_file = "TCIA_001.nii.gz"

        # make directories
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        Path(self.nii_path).mkdir(parents=True, exist_ok=True)
        Path(self.result_path).mkdir(parents=True, exist_ok=True)

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  # nnUNet specific
        self.ref_img = sitk.ReadImage(mha_input_path)
        sitk.WriteImage(self.ref_img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  # nnUNet specific
        img = sitk.ReadImage(nii_input_path)
        img.CopyInformation(self.ref_img)
        sitk.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print("Checking GPU availability")
        is_available = torch.cuda.is_available()
        print("Available: " + str(is_available))
        print(f"Device count: {torch.cuda.device_count()}")
        if is_available:
            print(f"Current device: {torch.cuda.current_device()}")
            print("Device name: " + torch.cuda.get_device_name(0))
            print(
                "Device memory: "
                + str(torch.cuda.get_device_properties(0).total_memory)
            )

    def load_inputs(self):
        """
        Read from /input/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        ct_mha = os.listdir(os.path.join(self.input_path, "images/ct/"))[0]
        pet_mha = os.listdir(os.path.join(self.input_path, "images/pet/"))[0]
        uuid = os.path.splitext(ct_mha)[0]

        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/ct/", ct_mha),
            os.path.join(self.nii_path, "TCIA_001_0000.nii.gz"),
        )
        self.convert_mha_to_nii(
            os.path.join(self.input_path, "images/pet/", pet_mha),
            os.path.join(self.nii_path, "TCIA_001_0001.nii.gz"),
        )
        return uuid

    def dice_coef(self, y_true, y_pred, smooth=1):

        y_true_f = np.ndarray.flatten(y_true)
        y_pred_f = np.ndarray.flatten(y_pred)
        intersection = np.sum(y_true_f * y_pred_f)
        dice_coef_ = (2.0 * intersection + smooth) / (
            np.sum(y_true_f) + np.sum(y_pred_f) + smooth
        )
        return dice_coef_

    def no_lesion(self, ensemble_2d):
        ensemble_2d[ensemble_2d < 0.6] = 0
        ensemble_2d[ensemble_2d >= 0.6] = 1
        return ensemble_2d

    def lesion(self, ensemble_lesion):
        # ensemble_lesion = (ensemble_2d + ensemble_3d + ensemble_residual)/3
        ensemble_lesion[ensemble_lesion >= 0.6] = 1
        ensemble_lesion[ensemble_lesion < 0.6] = 0
        return ensemble_lesion

    def adaptive_ensemble(self, output_list, final_ensemble):
        adaptive = []
        for index, i in enumerate(output_list):
            temp = i.copy()
            temp[temp >= 0.5] = 1
            temp[temp < 0.5] = 0
            dice_score = self.dice_coef(final_ensemble, temp)
            if dice_score >= 0.9:
                adaptive.append(i)
        ensemble = sum(adaptive) / len(adaptive)
        ensemble[ensemble >= 0.5] = 1
        ensemble[ensemble < 0.5] = 0
        return ensemble

    def ensemble(self):
        output_files = glob.glob(
            "/output/images/automated-petct-lesion-segmentation/*/*/*.nii.gz"
        )
        ensemble_residual = np.zeros(
            ((sitk.GetArrayFromImage(sitk.ReadImage(output_files[0]))).shape)
        )
        elist = []
        for outputs in output_files:
            img_data = sitk.GetArrayFromImage(sitk.ReadImage(outputs))
            elist.append(img_data)
            ensemble_residual += img_data
        ensemble_residual /= len(output_files)
        ensemble_temp = np.copy(ensemble_residual)
        ensemble_residual[ensemble_residual >= 0.8] = 1
        ensemble_residual[ensemble_residual < 0.8] = 0
        if ensemble_residual.max() != 0:
            ensemble_temp[ensemble_temp >= 0.5] = 1
            ensemble_temp[ensemble_temp < 0.5] = 0
            ensemble_temp = self.adaptive_ensemble(elist, ensemble_temp)
            ensemble_img = sitk.GetImageFromArray(ensemble_temp)
        else:
            ensemble_img = sitk.GetImageFromArray(ensemble_residual)

        sitk.WriteImage(
            ensemble_img, os.path.join(self.output_path, "TCIA_001.nii.gz"), True
        )

    def write_outputs(self, uuid):
        """
        Write to /output/
        Check https://grand-challenge.org/algorithms/interfaces/
        """
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.convert_nii_to_mha(
            os.path.join(self.output_path, self.nii_seg_file),
            os.path.join(self.output_path, uuid + ".mha"),
        )
        print("Output written to: " + os.path.join(self.output_path, uuid + ".mha"))

    def predict(self):
        models = ["residual"]
        for model in models:
            for folds in range(0, 5):
                if model != "residual":
                    if not os.path.isdir(
                        "/output/images/automated-petct-lesion-segmentation/{model}/fold{folds}"
                    ):
                        os.system(
                            f"mkdir -p /output/images/automated-petct-lesion-segmentation/{model}/fold{folds}"
                        )
                    os.system(
                        f"nnUNet_predict -i nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs/ -o /output/images/automated-petct-lesion-segmentation/{model}/fold{folds}/ -t Task001_TCIA -tr nnUNetTrainerV2 -m {model} -p nnUNetPlansv2.1 --overwrite_existing --save_npz -f {folds}"
                    )
                else:
                    if not os.path.isdir(
                        "/output/images/automated-petct-lesion-segmentation/{model}/fold{folds}"
                    ):
                        os.system(
                            f"mkdir -p /output/images/automated-petct-lesion-segmentation/{model}/fold{folds}"
                        )
                    os.system(
                        f"nnUNet_predict -i nnUNet_raw_data_base/nnUNet_raw_data/Task001_TCIA/imagesTs/ -o /output/images/automated-petct-lesion-segmentation/{model}/fold{folds}/ -t Task001_TCIA -tr nnUNetTrainerV2_ResencUNet -m 3d_fullres -p nnUNetPlans_FabiansResUNet_v2.1 --overwrite_existing --save_npz -f {folds}"
                    )
                print(f"Prediction finished for fold {folds} using {model}")

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()
        print("Start processing")
        uuid = self.load_inputs()
        print("Start prediction")
        self.predict()
        print("Start output writing")
        self.ensemble()
        self.write_outputs(uuid)


if __name__ == "__main__":
    Autopet().process()
