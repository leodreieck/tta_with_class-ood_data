corruptions = [
    'gaussian_noise', 'impulse_noise', 'shot_noise', 'defocus_blur', 'glass_blur', 'zoom_blur', 'motion_blur',
    'snow', 'frost', 'brightness', 'elastic_transform', 'jpeg_compression', 'contrast', 'pixelate', 'fog'   
]

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

ood_dataset_dict = {"SVHN": 0, 
                     "SVHNC": 0,
                     "CIFAR100": 0,
                     "CIFAR100C": 0
                     }
n_ood_samples = [100]
include_no_ood = False

### norm no detection ###
# [lr, pl_th, oodMeth, ood_th,epoch]
### norm no detection ###
"""hps = {"noOOD": [0.001,0.0,"none",0.0,1],
        "CIFAR100-100": [0.001,0.0,"none",0.0,1],
        "CIFAR100-200": [0.001,0.0,"none",0.0,1],
        "CIFAR100C-100": [0.001,0.0,"none",0.0,1],
        "CIFAR100C-200": [0.001,0.0,"none",0.0,1],
        "SVHN-100": [0.001,0.0,"none",0.0,1],
        "SVHN-200": [0.001,0.0,"none",0.0,1],
        "SVHNC-100": [0.001,0.0,"none",0.0,1],
        "SVHNC-200": [0.001,0.0,"none",0.0,1]}"""
### gce no detection ###
"""hps = {"noOOD": [0.0003,0.8,"none",0.0,5],
        "CIFAR100-100": [0.0003,0.8,"none",0.0,4],
        "CIFAR100-200": [0.0003,0.8,"none",0.0,4],
        "CIFAR100C-100": [0.0003,0.8,"none",0.0,4],
        "CIFAR100C-200": [0.0003,0.8,"none",0.0,4],
        "SVHN-100": [0.001,0.8,"none",0.0,4],
        "SVHN-200": [0.001,0.8,"none",0.0,5],
        "SVHNC-100": [0.0003,0.8,"none",0.0,5],
        "SVHNC-200": [0.001,0.8,"none",0.0,4]}"""
### tent no detection ###
"""hps = {"noOOD": [0.001,0.0,"none",0.0,4],
        "CIFAR100-100": [0.0003,0.0,"none",0.0,4],
        "CIFAR100-200": [0.001,0.0,"none",0.0,2],
        "CIFAR100C-100": [0.0003,0.0,"none",0.0,3],
        "CIFAR100C-200": [0.0003,0.0,"none",0.0,3],
        "SVHN-100": [0.001,0.0,"none",0.0,4],
        "SVHN-200": [0.001,0.0,"none",0.0,6],
        "SVHNC-100": [0.001,0.0,"none",0.0,4],
        "SVHNC-200": [0.0003,0.0,"none",0.0,6]}"""
### spl no detection ###
'''hps = {
        'noOOD': [0.0003,0.0,'none',0.0,1],
        'CIFAR100-100': [0.003,0.0,'none',0.0,4],
        'CIFAR100-200': [0.003,0.0,'none',0.0,4],
        'CIFAR100C-100': [0.0003,0.0,'none',0.0,1],
        'CIFAR100C-200': [0.0003,0.0,'none',0.0,1],
        'SVHN-100': [0.0003,0.0,'none',0.0,5],
        'SVHN-200': [0.001,0.0,'none',0.0,3],
        'SVHNC-100': [0.0003,0.0,'none',0.0,1],
        'SVHNC-200': [0.0003,0.0,'none',0.0,1]
}'''
### hpl no detection ###
'''
hps = {
    'noOOD': [0.0003,0.0,'none',0.0,6],
    'CIFAR100-100': [0.0003,0.0,'none',0.0,4],
    'CIFAR100-200': [0.0003,0.0,'none',0.0,4],
    'CIFAR100C-100': [0.0003,0.0,'none',0.0,4],
    'CIFAR100C-200': [0.0003,0.0,'none',0.0,4],
    'SVHN-100': [0.0003,0.0,'none',0.0,6],
    'SVHN-200': [0.001,0.0,'none',0.0,5],
    'SVHNC-100': [0.0003,0.0,'none',0.0,6],
    'SVHNC-200': [0.001,0.0,'none',0.0,4]
}
'''

### ood detection
# spl
'''hps = {
    'CIFAR100-100': [0.003,0.0,'none',0.0,6,0.8],
    'CIFAR100C-100': [0.0003,0.0,'none',0.0,1,0.8],
    'SVHN-100': [0.0003,0.0,'none',0.0,5,0.8],
    'SVHNC-100': [0.0003,0.0,'none',0.0,1,0.8]
}'''
#hpl
'''hps = {
    'CIFAR100-100': [0.0003,0.0,'none',0.0,5,0.5],
    'CIFAR100C-100': [0.0003,0.0,'none',0.0,5,0.667],
    'SVHN-100': [0.0003,0.0,'none',0.0,5,0.5],
    'SVHNC-100': [0.0003,0.0,'none',0.0,5,0.667]
}'''
#gce
'''hps = {
    'CIFAR100-100': [0.0003,0.0,'none',0.0,5,0.5],
    'CIFAR100C-100': [0.0003,0.0,'none',0.0,5,0.667],
    'SVHN-100': [0.001,0.0,'none',0.0,5,0.5],
    'SVHNC-100': [0.0003,0.0,'none',0.0,5,0.667]
}'''
#tent
hps = {
    'CIFAR100-100': [0.0003,0.0,'none',0.0,5,0.5],
    'CIFAR100C-100': [0.0003,0.0,'none',0.0,5,0.667],
    'SVHN-100': [0.001,0.0,'none',0.0,5,0.5],
    'SVHNC-100': [0.001,0.0,'none',0.0,1,0.5]
}

with open('03_cfgs/hp_final_tent_ooddetection.txt', 'w') as f:
    counter = 0
    for corruption in corruptions:
        for dataset in ood_dataset_dict:
            for n_oods in n_ood_samples:
                ood_dataset_dict[dataset] = n_oods

                lr = hps[dataset+"-"+str(n_oods)][0]
                pl_threshold = hps[dataset+"-"+str(n_oods)][1]
                ood_method = "threshold_n_images"
                ood_threshold = hps[dataset+"-"+str(n_oods)][5]
                n_epochs = hps[dataset+"-"+str(n_oods)][4]

                f.write("'{}' {} {} {} {} {} {} {} {} {}\n".format(corruption, 
                                                                ood_dataset_dict['SVHN'], 
                                                                ood_dataset_dict['SVHNC'], 
                                                                ood_dataset_dict['CIFAR100'], 
                                                                ood_dataset_dict['CIFAR100C'], 
                                                                lr,
                                                                pl_threshold,
                                                                ood_method,
                                                                ood_threshold,
                                                                n_epochs))

                counter += 1
                ood_dataset_dict[dataset] = 0

        if include_no_ood:
            lr = hps["noOOD"][0]
            pl_threshold = hps["noOOD"][1]
            ood_method = hps["noOOD"][2]
            ood_threshold = hps["noOOD"][3]
            n_epochs = hps["noOOD"][4]
            f.write("'{}' {} {} {} {} {} {} {} {} {}\n".format(corruption, 
                                                                ood_dataset_dict['SVHN'], 
                                                                ood_dataset_dict['SVHNC'], 
                                                                ood_dataset_dict['CIFAR100'], 
                                                                ood_dataset_dict['CIFAR100C'], 
                                                                lr,
                                                                pl_threshold,
                                                                ood_method,
                                                                ood_threshold,
                                                                n_epochs))
            counter += 1

print("Succesfully created file. File contains {} rows".format(counter))