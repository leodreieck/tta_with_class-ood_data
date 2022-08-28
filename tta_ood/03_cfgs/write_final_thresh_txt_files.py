corruptions = [
    'gaussian_noise', 'impulse_noise', 'shot_noise', 'defocus_blur', 'glass_blur', 'zoom_blur', 'motion_blur',
    'snow', 'frost', 'brightness', 'elastic_transform', 'jpeg_compression', 'contrast', 'pixelate', 'fog'   
]

counter = 0
with open('03_cfgs/hp_final_breakepochs.txt', 'w') as f:
#for setup in [("tent",0.001,0.8,5)]: #("tent",0.001,0.8,6) ("spl",0.0003,0.8,1), ("hpl",0.0003,0.8,4),("rpl",0.0003,0.8,4)
    for setup in [("none", 0.0), ("threshold_n_images", 0.5)]:
    
    #with open('03_cfgs/hp_final_{}_thresholding.txt'.format(setup[0]), 'w') as f:
    
        for corruption in corruptions:
            lr = 0.001 #setup[1]
            ood_threshold = setup[1] #0.5 #setup[2]
            n_epochs = 6#setup[3]
            pl_threshold = 0.0
            ood_method = setup[0] #"threshold_n_images"

            f.write("'{}' {} {} {} {} {} {} {} {} {}\n".format(corruption, 
                                                            200,0,0,0,
                                                            lr,
                                                            pl_threshold,
                                                            ood_method,
                                                            ood_threshold,
                                                            n_epochs))
            counter += 1

print("Succesfully created file. File contains {} rows".format(counter))