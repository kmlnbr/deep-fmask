"""Function used for managing csv files containing various metrics"""
import csv
import os
from collections import OrderedDict
import logging
import tempfile

logger = logging.getLogger(__name__)

# Define the header for the various csv files used in this project

classname = ['clear', 'cloud', 'shadow', 'snow', 'water']
header_valid = ['Epoch'] + classname
header_pred = ['metric'] + classname

CSV_HEADERS = OrderedDict()

CSV_HEADERS['overall_statistics.csv'] = ['Epoch', 'Train_Loss', 'Train_Accuracy',
                                         'Valid_Loss', 'Valid_Accuracy',
                                         'Valid_MIOU']
CSV_HEADERS['valid_precision_class.csv'] = header_valid
CSV_HEADERS['valid_recall_class.csv'] = header_valid
CSV_HEADERS['valid_f1_class.csv'] = header_valid
CSV_HEADERS['valid_iou_class.csv'] = header_valid
CSV_HEADERS['pred_class.csv'] = header_valid

CSV_HEADERS['hollstein_dataset.csv'] = ['Lat', 'Longitude', 'Class']


def _create_csv(csv_path, header):
    """Creates csv file with header"""
    with open(csv_path, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)


def _update_csv(csv_path, row_entry):
    """Updates csv file with row entry"""
    with open(csv_path, mode='a') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row_entry)


def add_to_csv(row_entry, output_data_path, filename='overall_statistics.csv'):
    """Add given row to csv file. If the file doesn't exist, then it is created
    with the appropriate header and then the row is added to the file."""
    csv_path = os.path.join(output_data_path, filename)

    if not os.path.exists(csv_path):
        header = CSV_HEADERS[filename]
        _create_csv(csv_path, header)

    _update_csv(csv_path, row_entry)


def read_row(row_number, output_data_path, filename, skip_0=True):
    """Reads a given row from csv file"""
    csv_path = os.path.join(output_data_path, filename)
    with open(csv_path, mode='r') as file:
        csv_row = file.readlines()[row_number].strip()
    start = int(skip_0)
    return csv_row.split(',')[start:]


def print_val_csv_metrics(row_number, output_data_path):
    """Prints validation metrics of best epoch in tabular form at the end of
    training """
    l = '{}\t{:>9} \t{:>9} \t{:>9} \t{:>9} \t{:>9}'.format('metrics'.ljust(15),
                                                           *classname)
    print(l)
    logger.info(l)
    for filename in list(CSV_HEADERS.keys()):
        if not filename.startswith('valid_'):
            continue
        metric_name = filename.split('_')[1]
        csv_row = read_row(row_number, output_data_path, filename)
        l = '{}\t{:>9} \t{:>9} \t{:>9} \t{:>9} \t{:>9} '.format(
            metric_name.ljust(15), *csv_row)
        print(l)
        logger.info(l)


def print_pred_csv_metrics(output_data_path, filename):
    """Prints prediction metrics in tabular form"""
    l = '{}\t{:>9} \t{:>9} \t{:>9} \t{:>9} \t{:>9}'.format(
        'metrics'.ljust(15), *classname)
    print(l)
    logger.info(l)
    for i in range(1, 5):
        csv_row = read_row(i, output_data_path, filename, skip_0=False)
        metric_name = csv_row[0]
        l = '{}\t{:>9} \t{:>9} \t{:>9} \t{:>9} \t{:>9} '.format(
            metric_name.ljust(15), *csv_row[1:])
        print(l)
        logger.info(l)


def pred_csv(metrics_values, do_print=True):
    """Aggregates prediction metrics data for printing"""
    model_metrics = metrics_values[2]
    class_metrics_sorted = sorted(model_metrics.keys())

    with tempfile.TemporaryDirectory() as td:
        for metric in ['Precision', 'Recall', 'f1', 'iou']:
            row_entry = [metric]
            for i in class_metrics_sorted:
                # i is in the format classname_metric
                if metric in i: row_entry.append('{:7.4}'.format(model_metrics[i]))
            filename = 'pred_class.csv'
            add_to_csv(row_entry, td, filename)
        if do_print:
            print('Accuracy'.ljust(
                10) + '\t\t FMask {:6.4}\t Sen2Cor {:6.4}\t OurModel {:6.4}'.format(
                metrics_values[0]['acc'], metrics_values[1]['acc'], metrics_values[2]['acc']))
            print('mIOU'.ljust(
                10) + '\t\t FMask {:6.4}\t Sen2Cor {:6.4}\t OurModel {:6.4}'.format(
                metrics_values[0]['mIOU'], metrics_values[1]['mIOU'], metrics_values[2]['mIOU']))
            print('+' * 75)

            print_pred_csv_metrics(td, filename)


def make_overall_statistics_csv(train_metrics, valid_metrics, class_metrics_dict,
                                epoch, log_path):
    """Generate all csv files from the validation metrics at the end of each
    training epoch. """
    overall_metrics_str = ['{:.4}'.format(i) for i in
                           train_metrics.tolist() + valid_metrics.tolist()]
    row_entry = [str(epoch + 1)] + overall_metrics_str

    add_to_csv(row_entry, log_path, filename='overall_statistics.csv')

    class_metrics_sorted = sorted(class_metrics_dict.keys())
    for metric in ['Precision', 'Recall', 'f1', 'iou']:
        row_entry = [str(epoch + 1)]
        for i in class_metrics_sorted:
            # i is in the format classname_metric
            if metric in i:
                row_entry.append('{:7.4}'.format(class_metrics_dict[i]))

        add_to_csv(row_entry, log_path,
                   filename='valid_{}_class.csv'.format(metric.lower()))
