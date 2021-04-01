import csv
import os

# Define the header for the various csv files used in this project
CSV_HEADERS = {
    'train_statistics.csv': ['Epoch', 'TrainLoss','TrainAccuracy','TrainMIOU', 'ValidLoss','ValidAccuracy','ValidMIOU',],
}


def _create_csv(csv_path, header):
    with open(csv_path, mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)


def _update_csv(csv_path, row_entry):
    with open(csv_path, mode='a') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row_entry)


def add_to_csv(row_entry,output_data_path,filename='train_statistics.csv'):
    csv_path = os.path.join(output_data_path, filename)

    if not os.path.exists(csv_path):
        header = CSV_HEADERS[filename]
        _create_csv(csv_path, header)

    _update_csv(csv_path, row_entry)




