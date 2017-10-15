from io import StringIO
from threading import Thread
from unittest import TestCase

import requests

from deduplication.classifier_evaluator import ClassifierEvaluator
from deduplication.evaluation_server import EvaluationServer


class EvaluationServerTest(TestCase):
    def setUp(self):
        self.server = EvaluationServer(ClassifierEvaluator(), host_port=("127.0.0.1", 0))
        Thread(target=self.server.serve_forever).start()

    def tearDown(self):
        self.server.shutdown()

    def test(self):
        url = "http://127.0.0.1:" + str(self.server.server_port)
        full_data_ids = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"}
        id_duplicates = {frozenset({"1", "2"}), frozenset({"3", "4"}), frozenset({"5", "6"})}
        self.server.classifier_evaluator.prepare(full_data_ids, id_duplicates)

        files = {'file': StringIO('"id1";"id2"\n1;2\n4;5\n')}
        response = requests.post(url, files=files)

        self.assertEqual(200, response.status_code)

        self.assertTrue("precision" in response.text)
        self.assertTrue("0.5" in response.text)
        self.assertTrue("recall" in response.text)
        self.assertTrue("0.33" in response.text)
