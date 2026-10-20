"""Testes da avaliação final e da interpretação do modelo selecionado."""

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from src.data_preparation import FEATURE_COLUMNS
from src.model_evaluation import (
    SELECTED_MODEL,
    coefficient_importance,
    evaluate_final_model,
    main,
    train_selected_model,
    validation_permutation_importance,
)
from src.model_training import calculate_metrics, create_models


class ModelEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.X_train = pd.DataFrame({
            "radius_mean": [-4., -3., -2., -1., 1., 2., 3., 4.],
            "area_mean": [8., 7., np.nan, 5., 4., 3., 2., 1.],
        }, index=range(10, 18))
        self.y_train = pd.Series([0, 0, 0, 0, 1, 1, 1, 1], index=self.X_train.index)
        self.X_validation = pd.DataFrame({
            "radius_mean": [-3., -2., 2., 3.],
            "area_mean": [7., np.nan, 3., 2.],
        }, index=range(30, 34))
        self.y_validation = pd.Series([0, 0, 1, 1], index=self.X_validation.index)
        self.X_test = pd.DataFrame({
            "radius_mean": [-5., -1., 1., 5.],
            "area_mean": [9., 5., np.nan, 0.],
        }, index=range(50, 54))
        self.y_test = pd.Series([0, 0, 1, 1], index=self.X_test.index)

    def test_existing_configuration_and_no_random_forest_training(self):
        with (
            patch("src.model_evaluation.create_models", wraps=create_models) as factory,
            patch.object(RandomForestClassifier, "fit", side_effect=AssertionError("RF ajustada")),
            patch.object(RandomForestClassifier, "predict", side_effect=AssertionError("RF avaliada")),
        ):
            model = train_selected_model(self.X_train, self.y_train)
            evaluate_final_model(model, self.X_test, self.y_test)
        factory.assert_called_once_with()
        classifier = model.named_steps["classifier"]
        self.assertIsInstance(classifier, LogisticRegression)
        self.assertEqual(
            classifier.get_params(),
            create_models()[SELECTED_MODEL].named_steps["classifier"].get_params(),
        )
        self.assertEqual(classifier.max_iter, 1000)
        self.assertEqual(classifier.random_state, 42)

    def test_final_metrics_and_confusion_matrix(self):
        model = Mock()
        model.predict.return_value = np.array([0, 0, 1, 0, 0, 1, 1, 1])
        y = pd.Series([0, 0, 0, 1, 1, 1, 1, 1])
        X_test = pd.concat([self.X_test, self.X_test], ignore_index=True)
        metrics = evaluate_final_model(model, X_test, y)
        model.predict.assert_called_once_with(X_test)
        self.assertAlmostEqual(metrics["Accuracy"], 5 / 8)
        self.assertAlmostEqual(metrics["Precision"], 3 / 4)
        self.assertAlmostEqual(metrics["Recall"], 3 / 5)
        self.assertAlmostEqual(metrics["F1-score"], 2 / 3)
        np.testing.assert_array_equal(metrics["confusion_matrix"], [[2, 1], [2, 3]])

    def test_names_match_all_30_coefficients_in_original_order(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(60, 30)), columns=list(reversed(FEATURE_COLUMNS)))
        y = (X.iloc[:, 0] > 0).astype(int)
        model = train_selected_model(X, y)
        table = coefficient_importance(model)
        self.assertEqual(len(table), 30)
        self.assertEqual(set(table.index), set(FEATURE_COLUMNS))
        np.testing.assert_array_equal(
            table.loc[X.columns, "Coeficiente"],
            model.named_steps["classifier"].coef_[0],
        )
        np.testing.assert_array_equal(table["Valor absoluto"], table["Coeficiente"].abs())
        self.assertTrue(table["Valor absoluto"].is_monotonic_decreasing)
        for coefficient, sign in table[["Coeficiente", "Sinal"]].itertuples(index=False):
            expected = "positivo" if coefficient > 0 else "negativo" if coefficient < 0 else "zero"
            self.assertEqual(sign, expected)

    def test_permutation_importance_is_reproducible(self):
        model = train_selected_model(self.X_train, self.y_train)
        first = validation_permutation_importance(model, self.X_validation, self.y_validation)
        second = validation_permutation_importance(model, self.X_validation, self.y_validation)
        assert_frame_equal(first, second)
        self.assertEqual(set(first.index), set(self.X_validation.columns))
        self.assertTrue(np.isfinite(first.to_numpy()).all())
        self.assertTrue((first["Desvio padrão"] >= 0).all())
        self.assertTrue(first["Importância média"].is_monotonic_decreasing)

    def test_permutation_preserves_training_statistics_and_coefficients(self):
        model = train_selected_model(self.X_train, self.y_train)
        preprocessing = model.named_steps["preprocessing"]
        imputer, scaler = preprocessing.named_steps.values()
        before = [a.copy() for a in (
            imputer.statistics_, scaler.mean_, scaler.var_,
            model.named_steps["classifier"].coef_,
        )]
        with patch.object(model, "fit", side_effect=AssertionError("Novo ajuste")):
            validation_permutation_importance(model, self.X_validation, self.y_validation)
            evaluate_final_model(model, self.X_test, self.y_test)
        for previous, current in zip(before, (
            imputer.statistics_, scaler.mean_, scaler.var_,
            model.named_steps["classifier"].coef_,
        )):
            np.testing.assert_array_equal(previous, current)
        np.testing.assert_allclose(imputer.statistics_, self.X_train.median())
        np.testing.assert_allclose(scaler.mean_, self.X_train.fillna(self.X_train.median()).mean())
        self.assertEqual(scaler.n_samples_seen_, len(self.X_train))

    def test_training_and_final_predictions_are_reproducible(self):
        first = train_selected_model(self.X_train, self.y_train)
        second = train_selected_model(self.X_train, self.y_train)
        assert_frame_equal(coefficient_importance(first), coefficient_importance(second))
        np.testing.assert_array_equal(first.predict(self.X_test), second.predict(self.X_test))
        a = evaluate_final_model(first, self.X_test, self.y_test)
        b = evaluate_final_model(second, self.X_test, self.y_test)
        for metric in ("Accuracy", "Precision", "Recall", "F1-score"):
            self.assertEqual(a[metric], b[metric])
        np.testing.assert_array_equal(a["confusion_matrix"], b["confusion_matrix"])

    def test_main_routes_each_dataset_to_its_assigned_role(self):
        models = create_models()
        model = models[SELECTED_MODEL]
        splits = (
            self.X_train, self.X_validation, self.X_test,
            self.y_train, self.y_validation, self.y_test,
        )
        raw, features, target = object(), self.X_train, self.y_train
        with (
            patch("src.model_evaluation.load_dataset", return_value=raw) as load,
            patch("src.model_evaluation.prepare_data", return_value=(features, target)) as prepare,
            patch("src.model_evaluation.split_data", return_value=splits) as split,
            patch("src.model_evaluation.create_models", return_value=models),
            patch.object(model, "fit", wraps=model.fit) as fit,
            patch.object(model, "predict", wraps=model.predict) as predict,
            patch.object(RandomForestClassifier, "fit", side_effect=AssertionError("RF ajustada")),
            patch.object(RandomForestClassifier, "predict", side_effect=AssertionError("RF avaliada")),
            patch("src.model_evaluation.permutation_importance", wraps=permutation_importance) as permute,
            patch("src.model_evaluation.calculate_metrics", wraps=calculate_metrics) as metrics,
            redirect_stdout(StringIO()) as output,
        ):
            predict.__name__ = "predict"
            main()
        load.assert_called_once_with()
        prepare.assert_called_once_with(raw)
        split.assert_called_once_with(features, target)
        fit.assert_called_once()
        self.assertIs(fit.call_args.args[0], self.X_train)
        self.assertIs(fit.call_args.args[1], self.y_train)
        permute.assert_called_once()
        self.assertIs(permute.call_args.args[0], model)
        self.assertIs(permute.call_args.args[1], self.X_validation)
        self.assertIs(permute.call_args.args[2], self.y_validation)
        self.assertEqual(permute.call_args.kwargs["random_state"], 42)
        self.assertEqual(permute.call_args.kwargs["n_repeats"], 10)
        scorer = permute.call_args.kwargs["scoring"]
        fake_model = LogisticRegression()
        fake_model.classes_ = np.array([0, 1])
        fake_model.predict = lambda X: np.array([0, 0, 1, 0])
        fake_model.predict.__name__ = "predict"
        self.assertAlmostEqual(scorer(fake_model, self.X_validation, self.y_validation), 0.5)
        metrics.assert_called_once()
        self.assertIs(metrics.call_args.args[0], self.y_test)
        self.assertEqual(len(metrics.call_args.args[1]), len(self.y_test))
        final_calls = [c for c in predict.call_args_list if c.args[0] is self.X_test]
        self.assertEqual(len(final_calls), 1)
        for call in predict.call_args_list:
            if call.args[0] is not self.X_test:
                self.assertTrue(call.args[0].index.equals(self.X_validation.index))
        for text in (SELECTED_MODEL, "Accuracy", "Recall", "VN, FP", "FN, VP", "causalidade"):
            self.assertIn(text, output.getvalue())


if __name__ == "__main__":
    unittest.main()
