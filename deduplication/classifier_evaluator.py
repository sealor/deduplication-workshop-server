class ClassifierEvaluator:
    def __init__(self):
        self.duplicate_count = 0
        self.duplicate_closures = {}

    def prepare(self, full_data_ids, id_duplicates):
        self._assert_ids_match(full_data_ids, id_duplicates)

        self.duplicate_closures = self._build_duplicate_closures(id_duplicates)
        self.duplicate_count = self._count_duplicates_in_duplicate_closures(self.duplicate_closures)

    @staticmethod
    def _assert_ids_match(full_data_ids, id_duplicates):
        for id_duplicate in id_duplicates:
            for id_duplicate_part in id_duplicate:
                if id_duplicate_part not in full_data_ids:
                    raise Exception("Unknown ID in gold standard file: " + str(id_duplicate_part))

    @staticmethod
    def _build_duplicate_closures(id_duplicates):
        duplicate_closures = dict()

        for id1, id2 in id_duplicates:
            id1_duplicates = duplicate_closures.get(id1, set())
            id2_duplicates = duplicate_closures.get(id2, set())

            id1_and_id2_duplicate_closure = id1_duplicates | id2_duplicates | {id1, id2}

            for part_id in id1_and_id2_duplicate_closure:
                duplicate_closures[part_id] = id1_and_id2_duplicate_closure

        return duplicate_closures

    @staticmethod
    def _count_duplicates_in_duplicate_closures(duplicate_closures):
        unique_duplicates = set()
        for duplicate_id1, duplicate_closure in duplicate_closures.items():
            for duplicate_id2 in duplicate_closure:
                if duplicate_id1 != duplicate_id2:
                    unique_duplicates.add(frozenset((duplicate_id1, duplicate_id2)))
        return len(unique_duplicates)

    def evaluate_classifier_data(self, classifier_id_duplicates):
        true_positive_count = 0
        false_positive_count = 0

        for classifier_id1, classifier_id2 in classifier_id_duplicates:
            if classifier_id1 in self.duplicate_closures.get(classifier_id2, []):
                true_positive_count += 1
            else:
                false_positive_count += 1

        return {
            "true_positive_count": true_positive_count,
            "false_positive_count": false_positive_count,
            "precision": true_positive_count / len(classifier_id_duplicates),
            "recall": true_positive_count / self.duplicate_count
        }
