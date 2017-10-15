import cgi
from http.server import SimpleHTTPRequestHandler, HTTPServer
from io import StringIO

from deduplication.classifier_evaluator import ClassifierEvaluator
from deduplication.io_helper import load_full_data_ids, load_gold_standard_id_duplicates, read_id_duplicates

DATA_FILENAME = "data.csv"
GOLD_STANDARD_DATA_FILENAME = "gold_standard_data.csv"


class RequestHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        environ = {
            'REQUEST_METHOD': 'POST',
            'CONTENT_TYPE': self.headers['Content-Type'],
        }
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ=environ)
        file_content = form["file"].file.read().decode("utf-8")

        classifier_id_duplicates = read_id_duplicates(StringIO(file_content))
        result = self.server.classifier_evaluator.evaluate_classifier_data(classifier_id_duplicates)

        self.send_head()
        self.wfile.write(str(result).encode("utf-8"))


class EvaluationServer(HTTPServer):
    def __init__(self, classifier_evaluator, host_port=("127.0.0.1", 8080), request_handler_class=RequestHandler):
        HTTPServer.__init__(self, host_port, request_handler_class)
        self.classifier_evaluator = classifier_evaluator


if __name__ == '__main__':
    full_data_ids = load_full_data_ids(DATA_FILENAME)
    id_duplicates = load_gold_standard_id_duplicates(GOLD_STANDARD_DATA_FILENAME)

    classifier_evaluator = ClassifierEvaluator()
    classifier_evaluator.prepare(full_data_ids, id_duplicates)

    EvaluationServer(classifier_evaluator).serve_forever()
