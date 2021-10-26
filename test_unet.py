from trainers.unet_trainer import UnetTrainer
import os
import yaml

def main():

    file = open("hyperparameters/hyperparameters.yaml", 'r')
    parameter_dict = yaml.safe_load(file)

    if parameter_dict['mlflow']:
        import mlflow
        
        mlflow.set_experiment(parameter_dict["net"])

        with mlflow.start_run():
            unet_trainer = UnetTrainer("hyperparameters.yaml")
            unet_trainer.test()
    else:
        unet_trainer = UnetTrainer("hyperparameters.yaml")
        unet_trainer.test()


if __name__ == '__main__':

    try:
        main()
        os.system('chmod -R 777 .')
    except KeyboardInterrupt:
        os.system('chmod -R 777 .')
        exit()
