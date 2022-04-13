import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from models.violence import ViolenceDetection
from data.trainset import ViolenceDataset, ViolenceDatasetOF
from data.testset import ViolenceTest, ViolenceTestOF

if __name__ == '__main__':

    logger = CSVLogger(save_dir='../log', name='contro_test', version=3)

    # RGB Input
    # fight_detection /mnt/d/serao/real_life_violence bus_dataset/train bus_data_cp
    data = ViolenceDataset('/mnt/d/serao/real_life_violence', num_clips=16, batch_size=5)

    # Optical Flow Input
    # data = ViolenceDatasetOF('/mnt/d/serao/busd_cp/train', num_clips=8, batch_size=5)

    # model = ViolenceDetection.load_from_checkpoint('/home/serao/violence_detection/violence-detection/log/rlvs_r2plus1d/version_0/epoch=43-val_loss=0.091-val_acc=0.981.ckpt', save_dir='../log/busd_r2plus1d_rlvs/version_0')
    model = ViolenceDetection(save_dir='../log/contro_test/version_3')
    trainer = pl.Trainer(gpus=1, max_epochs=60, num_sanity_val_steps=0, logger=logger)
    trainer.fit(model, data)