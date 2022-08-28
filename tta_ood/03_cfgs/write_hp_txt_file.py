normal_corruptions = [
    'gaussian_noise', 'impulse_noise', 'shot_noise', 'defocus_blur', 'glass_blur', 'zoom_blur', 'motion_blur',
    'snow', 'frost', 'brightness', 'elastic_transform', 'jpeg_compression', 'contrast', 'pixelate', 'fog'   
]

holdout_corruptions = ['gaussian_blur', 'saturate', 'speckle_noise', 'spatter']
holdout_corruptions_tsne = ['speckle_noise','saturate']

"""
corruption_par=${params[0]}
svhn_par=${params[1]}
svhnc_par=${params[2]}
cifar100_par=${params[3]}
cifar100c_par=${params[4]}
lr_par=${params[5]}
pl_t_par=${params[6]}
ood_m_par=${params[7]}
ood_t_par=${params[8]}
"""

lrs = [0.001] #[0.001, 0.003, 0.0003]

pl_thresholds = [0.0] #[0.0,0.5]

ood_datasets = ["SVHNC"]#["SVHN", "SVHNC", "CIFAR100", "CIFAR100C"] #["SVHN", "SVHNC", "CIFAR100", "CIFAR100C"] ["none"] 
include_no_ood = True
ood_dataset_dict = {"SVHN": 0, 
                     "SVHNC": 0,
                     "CIFAR100": 0,
                     "CIFAR100C": 0
                     }

n_ood_samples = [100]
include_no_ood = False

#ood_methods = ["none"] #["none", "threshold_n_images", "threshold_confidence"]
#ood_thresholds = [0.5, 0.8]

ood_detection_settings = [("threshold_n_images", 0.5), ("threshold_n_images", 0.667), ("threshold_n_images", 0.8)]#[("none",0.0)] #[("threshold_n_images", 0.5), ("threshold_n_images", 0.667), ("threshold_n_images", 0.8)] #[("none","0.")]#[("threshold_n_images", 0.5), ("threshold_confidence", 0.5), ("threshold_n_images", 0.8), ("threshold_confidence", 0.8)]

with open('03_cfgs/hp_settings_ooddetection_lr0.001.txt', 'w') as f:
    counter = 0
    for corruption in holdout_corruptions:
        for lr in lrs:
            for pl_threshold in pl_thresholds:
                for setting in ood_detection_settings:
                    ood_method, ood_threshold = setting
                    for dataset in ood_datasets: #[0]:#
                        for n_oods in n_ood_samples:
                            ood_dataset_dict[dataset] = n_oods
                            
                            f.write("'{}' {} {} {} {} {} {} {} {}\n".format(corruption, 
                                                                            ood_dataset_dict['SVHN'], 
                                                                            ood_dataset_dict['SVHNC'], 
                                                                            ood_dataset_dict['CIFAR100'], 
                                                                            ood_dataset_dict['CIFAR100C'], 
                                                                            lr,
                                                                            pl_threshold,
                                                                            ood_method,
                                                                            ood_threshold))
                            counter += 1
                            ood_dataset_dict[dataset] = 0

                    if include_no_ood:
                        f.write("'{}' {} {} {} {} {} {} {} {}\n".format(corruption, 
                                                                                ood_dataset_dict['SVHN'], 
                                                                                ood_dataset_dict['SVHNC'], 
                                                                                ood_dataset_dict['CIFAR100'], 
                                                                                ood_dataset_dict['CIFAR100C'], 
                                                                                lr,
                                                                                pl_threshold,
                                                                                ood_method,
                                                                                ood_threshold))
                        counter += 1

print("Succesfully created file. File contains {} rows".format(counter))