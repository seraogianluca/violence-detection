import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from models.violence import ViolenceDetection
from data.trainset import ViolenceDataset, ViolenceDatasetOF
from data.testset import ViolenceTest, ViolenceTestOF

if __name__ == '__main__':

    logger = CSVLogger(save_dir='../log', name='rl_optflow', version=0)
    # fight_detection /mnt/d/serao/real_life_violence bus_dataset/train
    # data = ViolenceDataset('/mnt/d/serao/bus_dataset/train', num_clips=16, batch_size=5)
    data = ViolenceDatasetOF('/mnt/d/serao/real_life_violence', num_clips=8, batch_size=5)
    # model = ViolenceDetection.load_from_checkpoint('/home/serao/violence_detection/violence-detection/log/rl_violence_1/version_0/epoch=28-val_loss=0.073-val_acc=0.980.ckpt')
    model = ViolenceDetection(save_dir='../log/rl_optflow/version_1')
    trainer = pl.Trainer(gpus=1, max_epochs=60, num_sanity_val_steps=0, logger=logger)
    trainer.fit(model, data)

    # model_test = ViolenceDetection.load_from_checkpoint('/home/serao/violence_detection/violence-detection/log/optflow/version_0/epoch=7-val_loss=0.565-val_acc=0.728.ckpt')
    # test_set = ViolenceTestOF('/mnt/d/serao/bus_dataset/test', num_clips=8, batch_size=5)
    # test_trainer = pl.Trainer(gpus=1, logger=logger)
    # test_trainer.test(model=model_test, dataloaders=test_set, verbose=True)