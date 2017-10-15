from unittest import TestCase

from deduplication.classifier_evaluator import ClassifierEvaluator


class ClassifierEvaluatorTest(TestCase):
    def test_assert_ids_match(self):
        full_data_ids = {1, 2, 3, 4}
        id_duplicates = {frozenset({1, 2}), frozenset({3, 4})}

        ClassifierEvaluator._assert_ids_match(full_data_ids, id_duplicates)

    def test_assert_ids_match_raises_exception(self):
        full_data_ids = {1, 2, 3, 4}
        id_duplicates = {frozenset({1, 2}), frozenset({3, 5})}

        with self.assertRaises(Exception) as context:
            ClassifierEvaluator._assert_ids_match(full_data_ids, id_duplicates)

        self.assertEqual("Unknown ID in gold standard file: 5", str(context.exception))

    def test_build_duplicate_closures_without_closure(self):
        id_duplicates = {frozenset({1, 2}), frozenset({3, 5})}

        duplicate_closures = ClassifierEvaluator._build_duplicate_closures(id_duplicates)

        self.assertEqual({1: {1, 2}, 2: {1, 2}, 3: {3, 5}, 5: {3, 5}}, duplicate_closures)

    def test_build_duplicate_closures_with_closure(self):
        id_duplicates = {frozenset({1, 2}), frozenset({2, 5})}

        duplicate_closures = ClassifierEvaluator._build_duplicate_closures(id_duplicates)

        self.assertEqual({1: {1, 2, 5}, 2: {1, 2, 5}, 5: {1, 2, 5}}, duplicate_closures)

    def count_duplicates_in_duplicate_closures_without_closure(self):
        duplicate_closures = {1: {1, 2}, 2: {1, 2}, 3: {3, 5}, 5: {3, 5}}

        self.assertEqual(2, ClassifierEvaluator._count_duplicates_in_duplicate_closures(duplicate_closures))

    def count_duplicates_in_duplicate_closures_with_closure(self):
        duplicate_closures = {1: {1, 2, 5}, 2: {1, 2, 5}, 5: {1, 2, 5}}

        self.assertEqual(3, ClassifierEvaluator._count_duplicates_in_duplicate_closures(duplicate_closures))

    def test_evaluate_test_data1(self):
        classifier_evaluator = ClassifierEvaluator()
        classifier_evaluator.prepare({1, 2, 3, 4, 5, 6, 7, 8, 9, 0}, [{1, 2}, {3, 4}])
        classifier_id_duplicates = {frozenset({1, 2}), frozenset({4, 3})}

        result = classifier_evaluator.evaluate_classifier_data(classifier_id_duplicates)

        self.assertEqual({
            "true_positive_count": 2,
            "false_positive_count": 0,
            "precision": 1,
            "recall": 1
        }, result)

    def test_evaluate_test_data2(self):
        classifier_evaluator = ClassifierEvaluator()
        classifier_evaluator.prepare({1, 2, 3, 4, 5, 6, 7, 8, 9, 0}, [{1, 2}, {3, 4}])
        classifier_id_duplicates = {frozenset({1, 2}), frozenset({3, 5})}

        result = classifier_evaluator.evaluate_classifier_data(classifier_id_duplicates)

        self.assertEqual({
            "true_positive_count": 1,
            "false_positive_count": 1,
            "precision": 0.5,
            "recall": 0.5
        }, result)
