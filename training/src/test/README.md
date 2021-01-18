# Testing

See below for an explanation around how we test the [Semantic Tagging TFX](https://github.com/sky-uk/disco-semantic-tagging-tfx) code.

---
**NOTE**: `functional/generate_test_data.py` must be run prior to any other functional tests!

---
The approach we take towards testing is outlined at a high-level in the path-to-production file in the [disco-machine-learning-operations repo](https://github.com/sky-uk/disco-machine-learning-operations/pull/62/files). Here we will detail how this approach towards testing is implemented for this particular training pipeline.

Currently, we have three "levels" of testing:
 1. Unit: Contains assertions about the **code** to ensure correct implementation
 2. Functional: Contains assertions about the **results** of particular components and their **functionality** to ensure that each component is behaving as intended.
 3. Testing built into the training pipeline itself (via custom validation TFX Components like our ThresholdValidator)

The training pipeline is a software system and should be tested as any other software system. We should note, however, that there is a subtle difference compared to a typical software system since we what we ultimately really care about is the correctness of the output of this system (i.e. the model). While in a typical software system (1) and (2) could give us pretty good indication that our system will behave as expected in production, for a machine learning system they tell us **essentially nothing** about how our model will behave in production. Therefore there is less of an emphsis on (1) and (2), and instead a greater emphasis should be placed on (3).

---
## Coverage of current tests

 + components/
    + component_utils.py
        - Covered by unit tests and indirectly tested via functional tests (see `functional/common_test_setup.py`)
    + model.py
        - Model definition is covered by unit tests
        - Tested for proper "machine learning behavior" in functional tests (e.g. is loss going down when training on a small amount of data?)
        - `get_server_tf_example_fn` is **not** currently being tested, will be tested indirectly by the InfrastructureValidator
        - `run_fn` is **not** currently being unit tested. It is tested indirectly by the pipeline runs in functional tests. This is also tested in the pipeline itself via the custom ThresholdValidator component.
    + ThresholdValidator.py
        - **Needs unit testing**
        - Indirectly tested via functional tests in `functional/test_pipeline`
    + transform.py
        - Unit tests on all funcitons
        - Indirectly tested via functional tests in `functional/test_pipeline`
 + pipelines/
    +  base_pipeline.py
        + base_pipeline is TFX pipeline definition. It consists of a series of standard and custom TFX components strung together. This is tested indirectly via the functional tests, particularly in `functional/test_pipeline.py` and `generate_test_data.py`
 + kubeflow_pipeline.py
    + This is the final pipeline. It could be compiled and run as a type of "integration/NFT" test on the KFP cluster. We should discuss whether it is worth running such a test before running the pipeline "for real".


---
## Notes on Training Pipeline "Tests"

In a TFX pipeline comprising standard components we will typically have three "gateways" before a model is pushed into production (data validation, model validation, and infrastructure validation). These steps are built into the pipeline itself. Currently we have only implemented a very simple version of model validation. Before more comprehensive data validation and model validation steps can be implemented more data science work needs to be done to better understand the data and the model.

We will likely want to expand beyond these three standard components to include more tests. This aligns with the point above that our main goal is to produce a high-quality model and therefore we should spend more time on these training pipeline tests.

Some examples of thigns we may want to include:

 + Human-in-the-loop for training pipeline. We may want to give the opportunity for human to manually inspect model outputs on certain examples before pushing a model. While this doesn't sound ideal, it may unfortunately be necessary due to the difficulty of quantitatively assessing this particular model.
  + More custom tests around data validation and model performance that don't fit neatly into existing TFX standard components.


---
## Notes on Testing `preprocessing_fn`

The `preprocessing_fn` used by the Transform component presents a double-edged sword; it provides a key benefit of TFX by incorporating the preprocessing steps into the model artifact, but it also has the disadvantage of being difficult to debug and to test. The main difficulty lies in the manner in which this function is called by the framework under the hood. In short, we cannot simply do something along the lines of the following:

```python
expected_output = ...
sample_input = ...
self.assertEquals(expected_output,
                  preprocessing_fn(sample_input))
```

Since the `preprocessing_fn` cannot be called in a straightforward manner as any other python function it is also difficult to make assertions about any method calls within the body of the `preprocessing_fn`.

[This github issue](https://github.com/tensorflow/transform/issues/167) gives some guidance on how to test the `preprocessing_fn`. However, the examples really only apply to using the `preprocessing_fn` in a standalone manner with tensorflow_transform as opposed to within the context of a full TFX pipeline.

The way we went about testing this was to refactor `transform.py` such that the `preprocessing_fn` itself only contained explicit calls to `tft` (tensorflow transform) methods and was therefore mostly a pass-through method. This allowed us to unit-test all of our processing we wrote and also then made it easy to mock out all the calls to `tft` methods such that we could treat the `preprocessing_fn` as a normal function.

While this seems to work well for this particular case, one thing to note is that this approach is potentially difficult to scale. If we had a lot of feature columns with calls to very many different `tft` methods we might end up having to patch a lot of methods.

---
## Notes on functional tests

`functional/generate_test_data.py` must be run prior to any other functional tests! The primary purpose of this file is to generate test data used by subsequent tests. However, since we are running the full pipeline in order to generate this data, we will get the added benefit of simulataneously checking that a full pipeline run works before diving in to some of the more specific tests contained in the `functional/` directory. While this may seem a bit backwards, it has two major benefits:
 + Test data is always up-to date with the code
 + Test data does not need to be checked in to source control since it is generated on-the-fly

---
## TODO

 + Training Pipeline Tests : This will be a main area of focus for phase 2 of the engagement
 + Unit tests for custom components.
