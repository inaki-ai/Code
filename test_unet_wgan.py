from trainers.wgan_trainer import WGanTrainer
import os
import mlflow
import yaml

def main():

    file = open("hyperparameters/hyperparameters.yaml", 'r')
    parameter_dict = yaml.safe_load(file)

    mlflow.set_experiment(parameter_dict["net"])

    with mlflow.start_run():
        wgan_trainer = WGanTrainer("hyperparameters.yaml")
        wgan_trainer.test()


if __name__ == '__main__':

    try:
        main()
        os.system('chmod -R 777 .')
    except KeyboardInterrupt:
        os.system('chmod -R 777 .')
        exit()
