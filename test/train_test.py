from gan_utils.lib.train import Trainer

dataroot = '/home/clayton/workspace/prj/data_keep/data/gan/dataset'
trainer = Trainer(dataroot=dataroot, video_save='vis.avi')
trainer.train()