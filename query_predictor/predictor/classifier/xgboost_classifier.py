# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the wrapper for an XGBoost classifier.
"""
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

from ..exceptions import ClassifierException
from ..logging_utils import get_module_logger
from .classifier import Classifier

_logger = get_module_logger(__name__)


class XGBoostClassifier(Classifier):
    """
    The wrapper for an XGBoost classifier. XGBoost is an implementation of gradient
    boosting decision trees.

    :param params: Parameters for the classifier.
    """

    def __init__(self, params: Optional[Dict] = None) -> None:
        super().__init__(params)
        self.type = "XGBoost"
        self.init_classifier()

    def init_classifier(self):
        """
        Initializes the XGBoost classifier.

        :return: ``None``
        """
        self.classifier = xgb.XGBClassifier(**self.params)

    def fit(
        self, train_data: np.ndarray, train_labels: np.array
    ) -> xgb.XGBClassifier:
        """
        Builds a XGBoost model from the training dataset. After the training,
        the ``self.classifier`` variable holds the best model instance. The method
        calls the XGBoost sklearn wrapper ``fit`` method for the model training.

        :param train_data: Training input samples.
        :param train_labels: Training labels.
        :return: A classifier instance of ``xgb.XGBClassifier``.
        """
        return self.classifier.fit(train_data, train_labels)

    def tune(
        self,
        train_data: np.ndarray,
        train_labels: np.ndarray,
        param_grid: Union[Dict[str, List], List[Dict[str, List]]],
        scoring: Union[
            str, Callable, List[str], Tuple[str], Dict[str, Callable]
        ] = "accuracy",
        n_jobs: int = -1,
        cv: int = 3,
    ) -> Dict:
        """
        Tunes the parameters of the XGBoost classification algorithm with an
        exhaustive search over parameter values provided. After the method, the
        ``Classifier.classifier`` variable is set to hold the best estimator.

        :param train_data: Training input samples.
        :param train_labels: Training labels.
        :param param_grid: Dictionary (or a list of such dictionaries) with parameters
        names as keys and lists of parameter settings for model tuning.
        :param scoring: A string of metric or a callable to evaluate the predictions
        on the validation dataset. The mean accuracy is used by default.
        :param n_jobs: Number of jobs to run in parallel. ``-1`` is set by default,
        meaning using all processors.
        :param cv: The number of folds used for the cross validation. ``3`` is set
        by default (instead of ``5`` in the ``sklearn.model_selection.GridSearchCV``).
        :return: A dictionary of the best score and corresponding parameters.
        """
        tune_search = GridSearchCV(
            self.classifier,
            param_grid,
            cv=cv,
            n_jobs=n_jobs,
            scoring=scoring,
            return_train_score=True,
        )
        tune_search.fit(train_data, train_labels)

        # The classifier holds the best estimator founded. So the classifier can
        # be used later for other purposes such as testing on another dataset.
        self.classifier = tune_search.best_estimator_

        # Logging the information of tuning results of all estimators.
        cv_results = tune_search.cv_results_
        for i, param in enumerate(cv_results["params"]):
            param_str = ", ".join(
                f"{key}={val}" for (key, val) in param.items()
            )
            _logger.info(
                "(%s): {mean_train_score: %s, mean_test_score: %s}",
                param_str,
                cv_results["mean_train_score"][i],
                cv_results["mean_test_score"][i],
            )

        _logger.info(
            "best_params: %s, best_score: %s",
            tune_search.best_params_,
            tune_search.best_score_,
        )

        return {
            "best_params": tune_search.best_params_,
            "best_score": tune_search.best_score_,
        }

    def predict(self, test_data: np.ndarray) -> np.ndarray:
        """
        Predicts classes of the testing inputs. It calls the XGBoost sklearn
        wrapper ``predict`` method for the model prediction.

        :param test_data: Testing data input samples.
        :return: Array of the predicted labels.
        """
        return self.classifier.predict(test_data)

    def score(self, test_data: np.ndarray, test_labels: np.ndarray) -> float:
        """
        Gets the mean accuracy of the given test data and labels.

        :param test_data: Testing data input samples.
        :param test_labels: Testing labels.
        :return: The mean accuracy.
        """
        return self.classifier.score(test_data, test_labels)

    def report(self, test_data: np.ndarray, test_labels: np.ndarray) -> Dict:
        """
        Reports additional metrics of the classifier. It contains
        1) The confusion matrix. By definition a confusion matrix :math:`C` is
        such that :math:`C_{i, j}` is equal to the number of observations known
        to be in group :math:`i` and predicted to be in group :math:`j`.
        2) Additional classification report including precision, recall, f1-score,
        and support of each class.
        3) Overall classification accuracy score.

        :param test_data: Testing data input samples.
        :param test_labels: Testing labels.
        :return: A dictionary containing the additional classification metrics.
        """
        pred_labels = self.predict(test_data)

        return {
            "confusion_matrix": confusion_matrix(test_labels, pred_labels),
            "classification_report": classification_report(
                test_labels, pred_labels
            ),
            "accuracy_score": accuracy_score(test_labels, pred_labels),
        }

    def save(self, path: str) -> None:
        """
        Saves the trained model.

        :param path: Target saved model path.
        :return: ``None``
        :raise ClassifierException: If the model hasn't been trained.
        """
        if self.classifier is None:
            raise ClassifierException("Model does not exist")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # We use the XGBoost ``save_model`` here instead of ``pickle`` or
        # ``joblib`` to maintain backward compatibility of models
        # Reference:
        # https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
        self.classifier.save_model(path)

    def load(self, path: str) -> xgb.XGBClassifier:
        """
        Loads a trained model.

        :param path: Loaded model path.
        :return: An XGBoost classifier.
        :raise ClassifierException: If the model path does not exist.
        """
        if not Path(path).exists():
            raise ClassifierException(f"File {path} does not exist")

        if not self.classifier:
            self.init_classifier()

        # We use the XGBoost ``load_model`` here instead of ``pickle`` or
        # ``joblib`` to maintain backward compatibility of models
        # Reference:
        # https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html
        self.classifier.load_model(path)
        return self.classifier
