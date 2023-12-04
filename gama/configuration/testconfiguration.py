import ConfigSpace as cs

from gama.configuration.test_configuration_task import (
    TestClassifierConfig,
    TestPreprocessorConfig,
)

# A configuration with limited operators for unit tests 🧪

config_space = cs.ConfigurationSpace(
    meta={
        # "gama_system_name": "current_configuration_name",
        "estimators": "classifiers",
        "preprocessors": "preprocessors",
    }
)

classifier_config = TestClassifierConfig(config_space)
classifier_config.setup_classifiers()

preprocessor_config = TestPreprocessorConfig(config_space)
preprocessor_config.setup_preprocessors()
