import glob
import pandas as pd
import statistics


def determine_corruption_type(corruption):
    standard = [
    'gaussian_noise', 'impulse_noise', 'shot_noise', 'defocus_blur', 'glass_blur', 'zoom_blur', 'motion_blur',
    'snow', 'frost', 'brightness', 'elastic_transform', 'jpeg_compression', 'contrast', 'pixelate', 'fog']

    holdout = ['gaussian_blur', 'saturate', 'speckle_noise', 'spatter']

    if corruption in standard:
        return "standard"
    elif corruption in holdout:
        return "holdout"
    else:
        print("Something went wrong. This corruption is not in the database")
        return "error"

def get_output_files(output_dir):

    #loc = output_dir + '*txt'
    loc = output_dir + '*.txt'
    files = glob.glob(loc)

    #print(files)

    return files

def generate_results_df_from_output_files(files):

    ood_datasets = ["CIFAR100", "CIFAR100C", "SVHN", "SVHNC"]

    results_list = [["Adaptation", "LearningRate", "PL_Threshold", "OOD_DetMethod", "OOD_Threshold"]]
    for dataset in ood_datasets:
        results_list[0].append(dataset)
    results_list[0].extend(["Corruption", "CorruptionType", "Severity", "Epoch", "Error"])

    for file in files:

        lines = []
        with open(file) as f:
            lines = [line.rstrip() for line in f]

        samples_per_dataset = dict()
        for dataset in ood_datasets:
                samples_per_dataset[dataset] = 0
        adaptation = learning_rate =  ""
        ood_method = "none"
        pl_threshold = "NA"
        ood_threshold =  "NA"
        learning_rate = "NA"

        for line in lines:

            if "ADAPTATION" in line:
                temp = line.split(' ')
                adaptation = temp[-1]

            if "LR:" in line and adaptation not in ["source","norm"]:
                temp = line.split(' ')
                learning_rate = temp[-1]

            if "PL_THRESHOLD" in line and adaptation in ["gce", "hardpl"]:
                temp = line.split(' ')
                pl_threshold = temp[-1]

            if "OOD_METHOD" in line:
                temp = line.split(' ')
                ood_method = temp[-1]

            if "OOD_THRESHOLD" in line and ood_method in ["threshold_confidence", "threshold_n_images"]:
                temp = line.split(' ')
                ood_threshold = temp[-1]

            for dataset in ood_datasets:
                if dataset + '_samples' in line:
                    temp = line.split(' ')
                    n_samples = int(temp[-1])
                    samples_per_dataset[dataset] = n_samples

            if "error %" in line:
                temp = line.split(' ')
                epoch = temp[-5]
                corruption = temp[-2][1:-3]
                severity = int(temp[-2][-3])
                error = float(temp[-1].replace("%",""))
                result = [adaptation, learning_rate, pl_threshold, ood_method, ood_threshold]
                for dataset in ood_datasets:
                    result.append(int(samples_per_dataset[dataset]))
                result.extend([corruption, determine_corruption_type(corruption), severity, epoch, error])
                results_list.append(result)

                if adaptation in ["source","norm"] and epoch=="1":
                    #print("test")
                    for temp_epoch in range(2,7):
                        temp_result = result.copy()
                        temp_result[-2] = str(temp_epoch)
                        results_list.append(temp_result)

    results_df = pd.DataFrame(results_list[1:], columns = results_list[0])

    return results_df


def transform_name_to_label(name):

    ood_datasets = ["CIFAR100", "CIFAR100C", "SVHN", "SVHNC"]

    label = list()
    label.append(name[0])
    label.append(name[1])
    label.append(name[9])

    temp = list()
    no_ood = True
    for i in range(5,9):
        if name[i] != 0:
            temp.append("{}: {}".format(ood_datasets[i-5], name[i]))
            no_ood = False
    if no_ood == True:
        label.append(["No OOD samples"])
    else:
        label.append(temp)
    if name[3] == "none":
        label.append("({})".format(name[3]))
    else:
        label.append("({}, {})".format(name[3], name[4]))

    return label

def generate_thresholding_df_from_output_files(files):

    ood_datasets = ["CIFAR100", "CIFAR100C", "SVHN", "SVHNC"]

    results_list = [["Adaptation", "LearningRate", "PL_Threshold", "OOD_DetMethod", "OOD_Threshold"]]
    for dataset in ood_datasets:
        results_list[0].append(dataset)
    results_list[0].extend(["Corruption", "CorruptionType", "Severity", "Epoch", "Error", "FilteredTotal", "FilteredCifar", "FilteredOOD"])

    for file in files:

        lines = []
        with open(file) as f:
            lines = [line.rstrip() for line in f]

        samples_per_dataset = dict()
        for dataset in ood_datasets:
                samples_per_dataset[dataset] = 0
        adaptation = learning_rate =  ""
        ood_method = "none"
        pl_threshold = "NA"
        ood_threshold =  "NA"
        learning_rate = "NA"
        epoch = 0
        filtered_out={"total": [], "cifar": [], "ood":[]}

        for line in lines:

            if "ADAPTATION" in line:
                temp = line.split(' ')
                adaptation = temp[-1]
                #print(adaptation)

            if "LR:" in line and adaptation not in ["source","norm"]:
                temp = line.split(' ')
                learning_rate = temp[-1]

            if "PL_THRESHOLD" in line and adaptation in ["gce", "hardpl"]:
                temp = line.split(' ')
                pl_threshold = temp[-1]

            if "OOD_METHOD" in line:
                temp = line.split(' ')
                ood_method = temp[-1]

            if "OOD_THRESHOLD" in line and ood_method in ["threshold_confidence", "threshold_n_images"]:
                temp = line.split(' ')
                ood_threshold = temp[-1]

            for dataset in ood_datasets:
                if dataset + '_samples' in line:
                    temp = line.split(' ')
                    n_samples = int(temp[-1])
                    samples_per_dataset[dataset] = n_samples

            if "N_Images OOD detection filtered out" in line:
                temp = line.split(' ')
                values = temp[-4].replace("[","").replace("]","").replace("%","").split(',')
                filtered_out["total"].append(float(values[0]))
                filtered_out["cifar"].append(float(values[1]))
                filtered_out["ood"].append(float(values[2]))


            if "error %" in line:
                temp = line.split(' ')
                epoch = temp[-5]
                filtered_total = statistics.mean(filtered_out["total"])
                filtered_cifar = statistics.mean(filtered_out["cifar"])
                filtered_ood = statistics.mean(filtered_out["ood"])
                corruption = temp[-2][1:-3]
                severity = int(temp[-2][-3])
                error = float(temp[-1].replace("%",""))

                result = [adaptation, learning_rate, pl_threshold, ood_method, ood_threshold]
                for dataset in ood_datasets:
                    result.append(int(samples_per_dataset[dataset]))
                result.extend([corruption, determine_corruption_type(corruption), severity, epoch, error, filtered_total, filtered_cifar, filtered_ood])
                results_list.append(result)

                if adaptation == "norm" and epoch=="1":
                    for temp_epoch in range(2,7):
                        temp_result = result.copy()
                        temp_result[-2] = str(temp_epoch)
                        results_list.append(temp_result)
                filtered_out={"total": [], "cifar": [], "ood":[]}

    results_df = pd.DataFrame(results_list[1:], columns = results_list[0])

    return results_df
