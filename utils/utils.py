import os
import logging
import datetime
import pandas as pd


def set_logger(args):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    args.log_file_name = (
        f"{args.dataset}_log-%s"
        % (datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    )
    log_path = args.log_file_name + ".log"
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        level=logging.DEBUG,
        filemode="w",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger


def get_labels_frequency(data_csv, col_target, col_single_id, verbose=False):
    """
    This function returns the frequency of each label in the dataset
    :param data_csv (string or pd.DataFrame): the path for a csv or a dataframe already loaded
    :param col_target (string): the name of the target/label column
    :param col_single_id (string): the name any column that is present for all rows in the dataframe
    :param verbose (boolean): a boolean to print or not the frequencies
    return (pd.DataFrame): a dataframe containing the frequency of each label
    """

    # Loading the data_csv
    if isinstance(data_csv, str):
        data_csv = pd.read_csv(data_csv)

    data_ = data_csv.groupby([col_target])[col_single_id].count()
    if (verbose):
        print('### Data summary: ###')
        print(data_)
        print(">> Total samples: {} <<".format(data_.sum()))

    return data_
