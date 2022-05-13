from gan_utils.lib.train import Trainer

dataroot = '/home/clayton/workspace/prj/data/gan/dataset'
trainer = Trainer(dataroot=dataroot, video_save='vis.avi', num_epochs=100)
trainer.train()
