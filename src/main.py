import pytorch_lightning as pl
from models.violence import ViolenceDetection
from data.trainset import ViolenceDataset
from data.testset import ViolenceTest

if __name__ == '__main__':
    #  fight_detection /mnt/d/serao/real_life_violence
    data = ViolenceDataset('/mnt/d/serao/bus_data_30fps/train', num_clips=16, batch_size=5)
    model = ViolenceDetection()

    trainer = pl.Trainer(gpus=1, max_epochs=60, num_sanity_val_steps=0)
    trainer.fit(model, data)

    #test_set = ViolenceTest('/mnt/d/serao/bus_dataset/test', num_clips=16, batch_size=5)
    #trainer = pl.Trainer(gpus=1)
    #trainer.test(model=model, dataloaders=test_set, ckpt_path='/home/serao/violence_detection/violence-detection/bus_model.ckpt', verbose=True)