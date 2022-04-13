import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from models.violence import ViolenceDetection
from data.trainset import ViolenceDataset, ViolenceDatasetOF
from data.testset import ViolenceTest, ViolenceTestOF

if __name__ == '__main__':

    logger = CSVLogger(save_dir='../log', name='temp', version=1)
    model_test = ViolenceDetection.load_from_checkpoint('/home/serao/violence_detection/violence-detection/log/fight_r3d/version_3/epoch=26-val_loss=0.407-val_acc=0.854.ckpt')

    # Optical Flow Input
    # test_set = ViolenceTestOF('/mnt/d/serao/busd_cp/test', num_clips=8, batch_size=1)

    # RGB Input
    test_set = ViolenceTest('/mnt/d/serao/fight_detection_cp/test', num_clips=10, batch_size=1)

    test_trainer = pl.Trainer(gpus=1, logger=logger)
    test_trainer.test(model=model_test, dataloaders=test_set, verbose=True)