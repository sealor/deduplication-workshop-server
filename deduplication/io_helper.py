import csv


def read_csv_rows(file):
    reader = csv.reader(file, delimiter=';')
    next(reader)  # ignore header
    for row in reader:
        yield row


def load_full_data_ids(filename):
    with open(filename, "r", encoding="iso-8859-1") as file:
        return {row[0] for row in read_csv_rows(file)}


def load_gold_standard_id_duplicates(filename):
    with open(filename, "r", encoding="iso-8859-1") as file:
        return read_id_duplicates(file)


def read_id_duplicates(file):
    return {frozenset({row[0], row[1]}) for row in read_csv_rows(file)}
