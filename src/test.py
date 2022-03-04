import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from models.violence import ViolenceDetection
from data.trainset import ViolenceDataset, ViolenceDatasetOF
from data.testset import ViolenceTest, ViolenceTestOF

if __name__ == '__main__':

    logger = CSVLogger(save_dir='../log', name='bus_r3d_test', version=0)
    model_test = ViolenceDetection.load_from_checkpoint('/home/serao/violence_detection/violence-detection/log/busd_r3d/version_0/epoch=42-val_loss=0.456-val_acc=0.859.ckpt')

    # Optical Flow Input
    # test_set = ViolenceTestOF('/mnt/d/serao/bus_data_cp/test', num_clips=16, batch_size=1)

    # RGB Input
    test_set = ViolenceTest('/mnt/d/serao/bus_data_cp/test', num_clips=16, batch_size=1)

    test_trainer = pl.Trainer(gpus=1, logger=logger)
    test_trainer.test(model=model_test, dataloaders=test_set, verbose=True)