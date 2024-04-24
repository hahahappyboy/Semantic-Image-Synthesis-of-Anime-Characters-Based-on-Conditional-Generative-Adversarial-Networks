
import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config

# python test.py --class_num 9  --name quintuplets --class_dir ./datasets/Quintuplets/class.txt  --ckpt_iter 140000 --dataset_mode custom --dataroot ./datasets/Quintuplets  --batch_size 1 --gpu_ids 0
# python test.py --class_num 15  --name zero_two --class_dir ./datasets/ZeroTwo/class.txt  --ckpt_iter 140000 --dataset_mode custom --dataroot ./datasets/ZeroTwo  --batch_size 1 --gpu_ids 0

if __name__ == "__main__":
    #--- read options ---#
    opt = config.read_arguments(train=False)

    #--- create dataloader ---#
    _, dataloader_val = dataloaders.get_dataloaders(opt)

    #--- create utils ---#
    image_saver = utils.results_saver(opt)

    #--- create models ---#
    model = models.OASIS_model(opt)
    model = models.put_on_multi_gpus(model, opt)
    # model.eval()

    #--- iterate over validation set ---#
    for i, data_i in enumerate(dataloader_val):
        _, label,edge,img_class = models.preprocess_input(opt, data_i)
        generated = model(None, label, "generate", None,edge,img_class)
        image_saver(label, generated, data_i["name"])
