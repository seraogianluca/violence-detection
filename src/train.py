import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from models.violence import ViolenceDetection
from data.trainset import ViolenceDataset, ViolenceDatasetOF
from data.testset import ViolenceTest, ViolenceTestOF

if __name__ == '__main__':

    logger = CSVLogger(save_dir='../log', name='busd_r3d', version=0)

    # RGB Input
    # fight_detection /mnt/d/serao/real_life_violence bus_dataset/train bus_data_cp
    data = ViolenceDataset('/mnt/d/serao/bus_data_cp/train', num_clips=16, batch_size=5)

    # Optical Flow Input
    # data = ViolenceDatasetOF('/mnt/d/serao/real_life_violence_cp/train', num_clips=8, batch_size=5)

    # model = ViolenceDetection.load_from_checkpoint('/home/serao/violence_detection/violence-detection/log/rl_violence_1/version_0/epoch=28-val_loss=0.073-val_acc=0.980.ckpt')
    model = ViolenceDetection(save_dir='../log/busd_r3d/version_0')
    trainer = pl.Trainer(gpus=1, max_epochs=60, num_sanity_val_steps=0, logger=logger)
    trainer.fit(model, data)