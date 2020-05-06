# -*- coding: utf-8 -*-
import getpass
import logging
import os
from pathlib import Path

import click
try:
    import kaggle
except OSError:
    print("Kaggle API's credential required")
    username = input("kaggle username : ")
    key = getpass.getpass("kaggle API key : ")
    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
    import kaggle

DATASET_NAME = "olistbr/brazilian-ecommerce"


def download_dataset(name, output_filepath, **kwargs):
    """Download the dataset."""
    logger = logging.getLogger(__name__)
    logger.info("Start download, this can take a while...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET_NAME,
                                      path=output_filepath,
                                      unzip=True)


def init_data_dir(project_dir):
    """Create empty dir if they not exists"""
    logger = logging.getLogger(__name__)
    logger.info("Init data directories")
    filepath = Path(project_dir).resolve()
    data_dirs = ['external',
                 'interim',
                 'processed',
                 'raw']
    data_path = filepath.joinpath('data')
    if not os.path.exists(data_path):
        os.mkdir(data_path)
        for directory in data_dirs:
            os.mkdir(data_path.joinpath(directory))
    else:
        logger.info('already exists')


def fetch_data(input_filepath):
    logger = logging.getLogger(__name__)
    logger.info('Making the dataset.')
    download_dataset(DATASET_NAME, output_filepath=input_filepath)


@click.command()
@click.argument('input_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    make_data(input_filepath)


def make_data(input_filepath):
    logger = logging.getLogger(__name__)
    logger.info('Making the dataset.')
    download_dataset(DATASET_NAME, output_filepath=input_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())
    init_data_dir(project_dir)
    main()
