import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh


class RegressorConfig:
    """Manages the configuration space for regressors in supervised learning contexts

    RegressorConfig oversees the configuration space of regressors used for a
    supervised machine learning task. This class facilitates the addition of
    new regressors and the modification of existing ones in the configuration space
    via standardized methods. The ConfigSpace library is utilized to designate the
    configuration space, enabling the creation of complex and adaptable
    configuration setups. For additional information on using constraints and
    various types of hyperparameters with ConfigSpace, refer to
    the ConfigSpace documentation, available at:
    https://automl.github.io/ConfigSpace/main/quickstart.html

    Add a regressor 💡
    ----------------

    1️⃣ To add a new regressor, define its setup method following the naming convention
    `setup_regressorName`. This method should:
        * Define hyperparameters specific to the regressor.
        * Use `_add_hyperparameters_and_equals_conditions` to add these
        hyperparameters to the config space with appropriate conditions.

    2️⃣ Next, your setup method needs to be added to the `regressors_setup_map` in
    the `__init__` method, where the key should be the Sci-kit learn name of your
    regressor, and the value should be pointing to your newly setup method.

    voila! 🎉 You are done! Your regressor is now added to the config space.

    How to use the shared hyperparameters 🪢
    -------------------------------------

    The shared hyperparameters are hyperparameters that are shared across multiple
    regressors. These hyperparameters are defined in the `shared_hyperparameters`
    property. To use these hyperparameters, simply add them to the setup method of
    the regressor you are adding. For example, to add the `n_estimators` hyperparameter
    to the `RandomForestRegressor`, add the following line to the
    `setup_random_forest_regressor` method:

    >>>    n_estimators = csh.CategoricalHyperparameter(
    >>>        "n_estimators__random_forest",
    >>>        self.shared_hyperparameters["n_estimators"]
    >>>    )

    voila! 🎉 The `n_estimators` hyperparameter is now added to the
    RandomForestRegressor with the shared value available in the
    `shared_hyperparameters` property.

    How to name my hyperparameters ✍️
    ------------------------------

    The hyperparameters you add to the config space should be named in the following
    format if similar hyperparameter names can be found in other regressors:

    >>>    <hyperparameter_name>__<regressor_name>

    For example, the `max_depth` hyperparameter for the `DecisionTreeRegressor` should
    be named `max_depth__decision_tree` given that the `max_depth` hyperparameter is
    also available in the `RandomForestRegressor`. This naming convention is used to
    ensure that the hyperparameters are added to the correct regressor in the config
    space.

    If your hyperparameter name is unique to your regressor, you can name it as you
    please without the need to have `__<regressor_name>` at the end of the name.
    Nonetheless, following the naming convention would in any way not cause any issues.

    Modify an existing regressor 💅
    -------------------

    To modify an existing regressor, adjust its respective setup method and the
    shared hyperparameters property as needed by modifying the values of the
    hyperparameters. For example, to change the value of the `n_estimators`
    hyperparameter for the `RandomForestRegressor`, change the value of the
    `n_estimators` hyperparameter in the `shared_hyperparameters` property by:

    >>>    "n_estimators": [50, 100, 200],

    The `n_estimators` hyperparameter will then be added to the config space with the
    appropriate value. However, be cautious, if you change values in the shared
    hyperparameters property, it will be changed for all regressors that use that
    hyperparameter. If you want this change to only apply to a specific regressor, you
    should add the hyperparameter to the setup method of that regressor. E.g. if you
    want to change the value of the `n_estimators` hyperparameter for the
    `RandomForestRegressor`, and only want this change to apply to the
    `RandomForestRegressor`, add the following line to the
    `setup_random_forest_regressor` method:

    >>>    n_estimators = csh.CategoricalHyperparameter(
    >>>        "n_estimators__random_forest", [50, 100, 200]
    >>>    )

    The `n_estimators` hyperparameter will be added as-is for the
    `RandomForestRegressor` and the value of the `n_estimators` hyperparameter for other
    regressors will be as available in the `shared_hyperparameters` property – iff they
    use the `n_estimators` hyperparameter of the `shared_hyperparameters` property.

    Parameters
    ----------
    config_space : cs.ConfigurationSpace
        The ConfigSpace object that defines the hyperparameters and their ranges for
        the regressors.

    """

    def __init__(
        self,
        config_space: cs.ConfigurationSpace,
    ):
        if "estimators" not in config_space.meta:
            raise ValueError("Expected 'estimators' key in meta of config_space")
        self.config_space = config_space
        self.regressors_setup_map = {
            "ElasticNetCV": self.setup_elastic_net_cv,
            "ExtraTreesRegressor": self.setup_extra_trees_regressor,
            "GradientBoostingRegressor": self.setup_gradient_boosting_regressor,
            "AdaBoostRegressor": self.setup_ada_boost_regressor,
            "DecisionTreeRegressor": self.setup_decision_tree_regressor,
            "KNeighborsRegressor": self.setup_k_neighbors_regressor,
            "LassoLarsCV": self.setup_lasso_lars_cv,
            "LinearSVR": self.setup_linear_svr,
            "RandomForestRegressor": self.setup_random_forest_regressor,
        }
        self.cs_estimators_name = self.config_space.meta["estimators"]

    @property
    def shared_hyperparameters(self):
        return {
            "n_estimators": [100],
            "max_features": {"lower": 0.05, "upper": 1.01, "default_value": 1.0},
            "min_samples_split": {"lower": 2, "upper": 21},
            "min_samples_leaf": {"lower": 1, "upper": 21},
            "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
            "loss": [
                "squared_error",
                "absolute_error",
                "huber",
                "quantile",
                "linear",
                "square",
                "exponential",
            ],
            "tol": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            "bootstrap": [True, False],
            "max_depth": {"lower": 1, "upper": 11},
        }

    def setup_regressors(self):
        regressors_choices = list(self.regressors_setup_map.keys())

        if not regressors_choices:
            raise ValueError("No regressors to add to config space")

        regressors = csh.CategoricalHyperparameter(
            name=self.cs_estimators_name,
            choices=regressors_choices,
        )
        self.config_space.add_hyperparameter(regressors)

        for regressor_name in regressors_choices:
            if setup_func := self.regressors_setup_map.get(regressor_name):
                setup_func(regressors)

    def _add_hyperparameters_and_equals_conditions(
        self, local_vars: dict, estimator_name: str
    ):
        if "regressors" not in local_vars or not isinstance(
            local_vars["regressors"], csh.CategoricalHyperparameter
        ):
            raise ValueError(
                "Expected 'regressors' key with a CategoricalHyperparameter in local"
                "vars"
            )

        hyperparameters_to_add = [
            hyperparameter
            for hyperparameter in local_vars.values()
            if isinstance(hyperparameter, csh.Hyperparameter)
            and hyperparameter != local_vars["regressors"]
        ]

        conditions_to_add = [
            cs.EqualsCondition(hyperparameter, local_vars["regressors"], estimator_name)
            for hyperparameter in hyperparameters_to_add
        ]

        self.config_space.add_hyperparameters(hyperparameters_to_add)
        self.config_space.add_conditions(conditions_to_add)

    def setup_elastic_net_cv(self, regressors: csh.CategoricalHyperparameter):
        l1_ratio = csh.UniformFloatHyperparameter(
            "l1_ratio__elastic_net_cv", lower=0.0, upper=1.01, default_value=0.05
        )
        tol = csh.CategoricalHyperparameter(
            "tol__elastic_net_cv", self.shared_hyperparameters["tol"]
        )

        self._add_hyperparameters_and_equals_conditions(locals(), "ElasticNetCV")

    def setup_extra_trees_regressor(self, regressors: csh.CategoricalHyperparameter):
        n_estimators = csh.Constant(
            "n_estimators__extra_trees_regressor",
            value=self.shared_hyperparameters["n_estimators"][0],
        )
        max_features = csh.UniformFloatHyperparameter(
            "max_features__extra_trees_regressor",
            **self.shared_hyperparameters["max_features"],
        )
        min_samples_split = csh.UniformIntegerHyperparameter(
            "min_samples_split__extra_trees_regressor",
            **self.shared_hyperparameters["min_samples_split"],
        )
        min_samples_leaf = csh.UniformIntegerHyperparameter(
            "min_samples_leaf__extra_trees_regressor",
            **self.shared_hyperparameters["min_samples_leaf"],
        )
        bootstrap = csh.CategoricalHyperparameter(
            "bootstrap__extra_trees_regressor", self.shared_hyperparameters["bootstrap"]
        )

        self._add_hyperparameters_and_equals_conditions(locals(), "ExtraTreesRegressor")

    def setup_gradient_boosting_regressor(
        self, regressors: csh.CategoricalHyperparameter
    ):
        n_estimators = csh.Constant(
            "n_estimators__gradient_boosting_regressor",
            value=self.shared_hyperparameters["n_estimators"][0],
        )
        loss = csh.CategoricalHyperparameter(
            "loss__gradient_boosting_regressor", self.shared_hyperparameters["loss"]
        )
        learning_rate = csh.CategoricalHyperparameter(
            "learning_rate__gradient_boosting_regressor",
            self.shared_hyperparameters["learning_rate"],
        )
        max_depth = csh.UniformIntegerHyperparameter(
            "max_depth__gradient_boosting_regressor",
            **self.shared_hyperparameters["max_depth"],
        )
        min_samples_split = csh.UniformIntegerHyperparameter(
            "min_samples_split__gradient_boosting_regressor",
            **self.shared_hyperparameters["min_samples_split"],
        )
        min_samples_leaf = csh.UniformIntegerHyperparameter(
            "min_samples_leaf__gradient_boosting_regressor",
            **self.shared_hyperparameters["min_samples_leaf"],
        )
        subsample = csh.UniformFloatHyperparameter(
            "subsample__gradient_boosting_regressor",
            lower=0.05,
            upper=1.01,
            default_value=1.0,
        )
        max_features = csh.UniformFloatHyperparameter(
            "max_features__gradient_boosting_regressor",
            **self.shared_hyperparameters["max_features"],
        )
        alpha = csh.CategoricalHyperparameter(
            "alpha__gradient_boosting_regressor", [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
        )

        self._add_hyperparameters_and_equals_conditions(
            locals(), "GradientBoostingRegressor"
        )

    def setup_ada_boost_regressor(self, regressors: csh.CategoricalHyperparameter):
        n_estimators = csh.Constant(
            "n_estimators__ada_boost_regressor",
            value=self.shared_hyperparameters["n_estimators"][0],
        )
        learning_rate = csh.CategoricalHyperparameter(
            "learning_rate__ada_boost_regressor",
            self.shared_hyperparameters["learning_rate"],
        )
        loss = csh.CategoricalHyperparameter(
            "loss__ada_boost_regressor", ["linear", "square", "exponential"]
        )

        self._add_hyperparameters_and_equals_conditions(locals(), "AdaBoostRegressor")

    def setup_decision_tree_regressor(self, regressors: csh.CategoricalHyperparameter):
        max_depth = csh.UniformIntegerHyperparameter(
            "max_depth__decision_tree_regressor",
            **self.shared_hyperparameters["max_depth"],
        )
        min_samples_split = csh.UniformIntegerHyperparameter(
            "min_samples_split__decision_tree_regressor",
            **self.shared_hyperparameters["min_samples_split"],
        )
        min_samples_leaf = csh.UniformIntegerHyperparameter(
            "min_samples_leaf__decision_tree_regressor",
            **self.shared_hyperparameters["min_samples_leaf"],
        )

        self._add_hyperparameters_and_equals_conditions(
            locals(), "DecisionTreeRegressor"
        )

    def setup_k_neighbors_regressor(self, regressors: csh.CategoricalHyperparameter):
        n_neighbors = csh.UniformIntegerHyperparameter(
            "n_neighbors__k_neighbors_regressor", lower=1, upper=101, default_value=5
        )
        weights = csh.CategoricalHyperparameter(
            "weights__k_neighbors_regressor", ["uniform", "distance"]
        )
        p = csh.CategoricalHyperparameter("p__k_neighbors_regressor", [1, 2])

        self._add_hyperparameters_and_equals_conditions(locals(), "KNeighborsRegressor")

    def setup_lasso_lars_cv(self, regressors: csh.CategoricalHyperparameter):
        normalize = csh.CategoricalHyperparameter(
            "normalize__lasso_lars_cv", [True, False]
        )

        self._add_hyperparameters_and_equals_conditions(locals(), "LassoLarsCV")

    def setup_linear_svr(self, regressors: csh.CategoricalHyperparameter):
        loss = csh.CategoricalHyperparameter(
            "loss__linear_svr", ["epsilon_insensitive", "squared_epsilon_insensitive"]
        )
        dual = csh.CategoricalHyperparameter("dual__linear_svr", [True, False])
        tol = csh.CategoricalHyperparameter(
            "tol__linear_svr", self.shared_hyperparameters["tol"]
        )
        C = csh.CategoricalHyperparameter(
            "C__linear_svr",
            [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        )
        epsilon = csh.CategoricalHyperparameter(
            "epsilon__linear_svr", [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        )

        self._add_hyperparameters_and_equals_conditions(locals(), "LinearSVR")

    def setup_random_forest_regressor(self, regressors: csh.CategoricalHyperparameter):
        n_estimators = csh.Constant(
            "n_estimators__random_forest_regressor",
            value=self.shared_hyperparameters["n_estimators"][0],
        )
        max_features = csh.UniformFloatHyperparameter(
            "max_features__random_forest_regressor",
            **self.shared_hyperparameters["max_features"],
        )
        min_samples_split = csh.UniformIntegerHyperparameter(
            "min_samples_split__random_forest_regressor",
            **self.shared_hyperparameters["min_samples_split"],
        )
        min_samples_leaf = csh.UniformIntegerHyperparameter(
            "min_samples_leaf__random_forest_regressor",
            **self.shared_hyperparameters["min_samples_leaf"],
        )
        bootstrap = csh.CategoricalHyperparameter(
            "bootstrap__random_forest_regressor",
            self.shared_hyperparameters["bootstrap"],
        )

        self._add_hyperparameters_and_equals_conditions(
            locals(), "RandomForestRegressor"
        )
