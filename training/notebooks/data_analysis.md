# Data Analysis
---

The original research notebooks contain some data analysis. For a summary of the data analysis already conducted, please see notes found in `disco-ml-operations` repo. The purpose of this document is to outline our general approach to data analysis, identify what pieces of data analysis are relevant for model productionization and thus require discussion and collaboration between data scientists and ml engineers, and enumerate particular pieces of analysis we'd like to do for this model in particular. 

Data analysis is *the* foundational step when developing a new ML use case. If we don't understand the data we have little chance of creating a good model, much less understanding it—any attempt to debug and understand the output of the model must be grounded on the patterns in the data. Furthermore, if we don't understand our data we will have no idea how to effecitvely monitor our ML system in practice. It is critical that we catch data errors early, before they propagate through our model and taint downstream models. 

When we initially perform data analysis for a new use-case, we should have two high-level goals in mind (we include some example questions we might ask in relation to each goal): 
    
 1. Gain understanding for model development
    + How should I clean the data?
    + Are there any interesting statistical patterns I can take advantage of?
    + What useful features should I engineer? 
    + What models might be appropriate? 
 2. Gain understanding for training and monitoring in production
    + What metrics are important to track? 
    + How often do I need to retrain my model? 
    + How can I detect shifts between the training and serving data? 

When we typically think of exploratory data analysis (EDA), we're usually thinking about the first goal: gaining an understanding of the data in such a way as to allow for the development of a good model. However, this analysis might be slightly different from that which needs to be done for model productionization. For now, we will place less of an emphasis on the first goal and instead focus on the types of data analysis that are relevant for productionizing our ML system. Of course, there will still be a lot of overlap between these two goals; many insights that we gain when doing feature engineering and developing our model will be relevant for monitoring the model in production. For example, during the course of preliminary data analysis for modeling we would hope to develop some notion of what "clean" data would mean for this particular use-case, as well as expectations around the type or distribution of each feature. We should take note of our expectations and communicate them to ML engineers!

---
---
## **Data Analysis for Productionizing Machine Learning Models**

Machine Learning systems are unique compared to many other software systems in that they tend to fail silently. Seemingly subtle changes in data distributions (the type of changes that would not usually trigger an alert in a normal software system) can destroy the performance of a machine learning model. This is due to the  assumption of independent and identically distributed (IID) data. In layman's terms, what this means is that when we train our machine learning models there is an assumption that the data our model sees "in the real world" will come from the same underlying distribution used to train the model. If this assumption is violated, our model could exhibit unexpected behavior. Unexpected behavior is obviously undesireable in any software system, and perhaps even moreso in machine learning due to the fact that we might already have limited understanding of the behavior of our "black box" deep learning model.

When dealing with a production machine learning model, we will have two major concerns when it comes to data: 

 + Ensuring that we are training on high-quality data that matches our expectations, otherwise the assumptions we initially made during modelling may not hold. 
    + As part of our checks during training, we also need to ensure that our evaluation data and training data come from the same distribution. This will give us more confidence in our evaluation metrics. 
 + Ensuring that the serving data is still from the same distribution as the training data. If this is not the case, we will need to retrain. 

Addressing these two concerns should help us to ensure model performance, which is ostensibly the most important aspect of monitoring machine learning systems. Ideally, we would be able to monitor a model's performance based on its accuracy in production. This would allow us to set up alerts in a relatively straightforward manner (e.g. if accuracy falls below X%, then trigger an alert). This also has the added benefit that our alerts might have a clear implications for the business. For example, we might be able to approximately quantify the monetary effect of our fraud detection model dropping a few points in accuracy. However, we will often find ourselves in the situation in which ground-truth labels are difficult to obtain due to operational and financial constraints. In these scenarios, we will need to base performance solely on the features of incoming data. If we can observe that our data distribution is shifting, it is a strong signal that our model might not behave in a consistent manner. 

Even in the case when we do have ground truth labels (and can thus track model performance in production), we should understand and monitor the data regardless. While we could solely rely on model-quality validation as a failsafe for data errors, resilience of ML algorithms to noisy data means that errors may result in small drops in model quality, which could go unnoticed (**especially** if these drops only apply to a small slice of data). Furthermore, if we place too much emphsis on model-quality validation at the cost of overlooking data validation, we will have limited understanding of our system. This will inhibit our ability to take the appropriate actions when things do go wrong (which they probably will). 

In summary, data validation is important **regardless** of whether we are able to track model performance metrics in production or not. Since data is so important in an ML workflow, we should probably be as rigorous about our data as we are with our code. This points to a data-centric approach to machine learning in which we develop a series of tests for our data, just as we would create tests for the code we write.


### **Practical Considerations**
While distribution shifts could cause severe degradation in model performance, they could also be perfectly benign (as noted above, many models will be robust to small changes in the data). We therefore need to be careful to distinguish malignant shifts that damage predictive performance from benign shifts that negligbly impact performance.

A point of discussion, then, is to find the appropriate trade-off between the following: 

+ **False alarm rate** (false positives). What are the implications if an alert goes off erroneously? 
    + Waste the valuable time of engineers 
    + Kick off new, unnessary training procedures
    + Lose trust in the system, begin to ignore alerts 
+ **Misdetection rate** (false negatives). What happens if an alert doesn't go off and it should have? 
    + Model performance might degrade (the "price" of this is highly dependent on the business use of the model)
    + Discover errors after its way too late, which might lead to constant fire-fighting and confusion over the source of poor results
    + Lose trust in the system
+ **Detection Delay**. What is the optimal delay between when distributions start to shift and firing an alert? 
    + Wait a small amount of time to catch shifts as they happen 
        + Would need to constantly run evaluation procedures, which might be costly OR
        + Would need to monitor every metric live, which might be difficult or impractical
        + Possibly increase the incidence of false positives 
        + In general, use a lot of resources for continuous monitoring and retraining
    + Wait a longer time to be more confident that the data distribution has in fact shifted
        + Sub-optimal model may have been in operation for some time
        + Less operational overhead
        + Fewer retraining runs means less expense on compute 

We will also need to consider how our model actually operates in production. The specific setup of our system may impact what types of validation checks are possible. For example, if we are doing batch scoring then we can afford to do all of our data quality and data shift tests as part of the scoring pipeline itself. On the other hand, if we are serving our model online we may need to collect all of the requests and then periodically run some type of evaluation pipeline to see if the serving distribution has shifted.  

Finally, we will need to discuss the specific methods we will use in order to perform data quality and data shift tests. Some of our data quality checks can be based on simple rules (e.g. Feature A should never be null), whereas our dataset-shift tests are fundamentally statistical in nature. An implication of this, therefore, is that we may in fact require separate machine learning models in order to detect dataset shifts for our ML system! 

### **Specific Methods for Detecting Drift**

In some cases, we may want to do outlier detection to flag anomalies whose model predictions we cannot trust and should not use in a production setting. This refers to catching **individual instances** that look untrustworthy. As a more general case, we need to focus on data drift detection to ensure that our training and test samples are drawn from the same underlying distribution. 

Different ways of approaching this problem have been suggested (see resources below). Many methods are based on statistical tests of feature values to ensure that they come from the same distribution. This type of method is explained very well in the [Failing Loudly](https://arxiv.org/abs/1810.11953?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529) paper and is implemented Seldon in their open-source package [Alibi Detect](https://github.com/SeldonIO/alibi-detect). At a high-level, the overall workflow involves taking the feature values (or a lower dimensional representation) and then performing some type of statistical test. Alibi Detect uses univariate Kolmogorov-Smirnov tests on each dimension (with Bonferoni correction) or a permutation test based on the values of the Maximum Mean Discrepancy. Other options seen in the literature include chi-square tests or thresholds based on Kullback-Leibler divergence, Jensen-Shannon divergence, or cosine similarity. The false positive rate can be tuned by the confidence level (in the case of a statistical test) or by setting the threshold (in the case of something like Kullback-Leibler divergence). 

Another option is called black-box shift detection. This method involves comparing the distribution of output predictions as a proxy for measuring dataset drift. The big advantage of this method is that we can reuse our existing model as a our "black-box predictor". After we get the predictions from each of our datasets, we can again perform statistical tests to check the difference between distributions. As such, this method is basically the same as that already described above with the distinction that our model itself will serve as our dimensionality reduction algorithm.

Another method worth considering is the use of domain classifiers. Here we can look directly at the input features from two sample datasets (e.g. training data and serving data). The goal is to train a classifier to predict which dataset a particular example comes from. Obviously it would be a bad sign if we are able to successfully discriminate between the two datasets. The main drawback to this method is that we need to train another model. 

---
## **Using Tensorflow Data Validation**

Within the Tensorflow Extended (TFX) ecosystem, we can use the Tensorflow Data Validation (TFDV) library for exploring and validating machine learning data. There are some key functionalities we can leverage:

 + Compution of summary statistics
 + Visualization of statistics 
 + Schema Inference
 + Data Validation
 + "Anomaly" detection

 In this section, we will discuss the TFDV library in more detail. The discussion here is heavily based on the [Data Validation for Machine Learning](https://research.google/pubs/pub47967/) paper.

 TFDV can be used in the very early stages of exploratory data analysis for visualizing, analyzing, and understanding the data. However, the key advantage of TFDV from a production-ML point of view is the data schema, which codifies our expectations for correct data. The schema follows a logical data model where each training or serving example is a collection of features, with each feature having several constraints attached to it. The schema can help us with the following: 
 
 1. Describing the expected data types, format, and distribution of values
 2. Validating new data that's received for the training model.
 3. Detecting training-serving skew 
 
 According to the TFDV authors, pipeline owners (i.e. us) are expected to treat the schema as a **production asset** at par with source code and to adapt best practices for reviewing, versioning, and mainting the schema. Furthermore, they advocate for the co-evoluation of data and schema in the lifecycle of these pipelines. 

 The main workflow for curating the schema should look something like the following: 
 
 1. Obtain an initial schema through automatic inference
 2. Review and manually edit the schema to codify expectations
 3. Commit the updated schema to a version control system and then start the training part of the pipeline
 4. Keep the schema up to date with the latest data changes

The implication is that this is seen as a human-in-the-loop process in which a person with extensive knowledge about the dataset will add their data expectations to the schema. 

TFDV follows a very different philosophy from the papers and methods described in the previous section. The TFDV approach is based on the need to surface **high-precision**, **actionable** alerts to a human operator. As such, it is much more geared towards minimizing false alarms as opposed to catching every possible data error or data shift. In contrast to the papers described above, the TFDV authors found that statistical tests for detecting changes in the data distribution—such as chi square tests—were **too sensitive** and also uninformative. As a result, they have looked at alternative, interpretable methods of quantifying changes between data distributions which are more resilient to observed differences in practice and whose sensitivity is easily tuned. To detect distribution skew, for example, they make use of the L-infinity norm (though it now looks like one can also define thresholds based on the jensen-shannon divergence for numerical features). 

In general, it seems like TFDV is a more rules-based approach to detecting dataset skew as opposed to the statistical approaches described above. They also claim that,

```
"while there are reasonable data properties that cannot currently be encoded by TFDV, our experience so far has shown that the current schema is powerful enough to capture the vast majority of use cases in production pipelines" 
```

**Our main goal will be to confirm whether or not this is the case**. Since we are using TFX, it makes sense to build our analysis around TFDV. That being said, there is still quite a bit of analysis that should be done outside of the capabilities of TFDV. We still need to do some preliminary data analysis to get an understanding of the data, and there may be additional checks/tests for either dataset quality or dataset skew to implement that are beyond the current capabilities of TFDV. In the next section, we describe a more general workflow centered around TFDV. 

**NOTE:** TFDV uses a slightly confusing terminology in which an "anomaly" is anything which deviates from the expected data characteristics as described in the schema. This includes the following: 
 + Data type mismatch 
 + Missing or unexpected features
 + Data domain violation 
 + Distribution anomalies 
    + Feature skew 
    + Distribution skew 


---
## **General Workflow**

In this section we will discuss how a data scientist can go about doing data analysis with an eye towards productionizing their model, i.e. data analysis in a manner that is helpful to the engineers who will be responsible for the model's operation in production. 

Before doing any data analysis at all, it is worth discussing the use case more generally. An example of the type of question we should ask ourselves is what the effect of having a suboptimal model operating in production could be? This simple question can have a profound impact on our approach to monitoring. 

Next, but still before any analysis really begins, it is important to get an understanding of the data source. We need to understand how often this data source is refreshed as well as establish whether anything could go wrong with this source. Is there a risk that the format of the upstream data could change? 

After we've laid some groundwork we should still start with the typical steps of initial and exploratory data analysis. We will still go through the process of exploring data, generating insights, testing hypotheses, checking assumptions, and revealing underlying hidden patterns in the data to exploit during modelling. Though a data scientist is probably more inclined towards finding patterns for modelling, even at the earliest stages of working with a dataset, establishing quality checks around the data and documenting these checks can immensely speed up operations in the long run. It would be highly beneficial if data scientsits did the following: 

 1. Identify which findings/insights from data analysis completed during model development are relevant for model monitoring
 2. Codify the expectations they have developed for the data (Feature A should be an integer in range 0-100, Feature B is a string that should only take on certain values) 
 3. Think about what additional pieces of analysis could be useful for model monitoring (even if they might not be immediately relevant for model development)

### **How does TFDV fit into the picture?**

 We can use TFDV to generate and visualize some statistics of our dataset. While this will certainly be helpful, there will inevitably be more analyses to do as more questions are raised and more hypothesis are formulated. TFDV is good to use relatively early in the analysis process. 

 After we feel we have a good understanding of our dataset, we should then use TFDV to generate a schema. We can use the knowledge we've gained from our analyses and discussions about the use-case to set the appropriate constraints on each feature in the schema. Finally, we can commit the updated schema to our github project where we will version it and go through the usual process for updates as we would with our code. 

### **Communicating Results** 

This process should involve extensive communication with the ML Engineers. Typically when we do exploratory data analysis we want to "tell a story" with the data. When we do data analysis with the goal of productionizing our model, we will have the benefit of knowing who our audience is. It is important that all relevant findings are presented in an understandable way. Critical insights should be surfaced in such a way as to be actionable. Furthermore, data documentation is essential for all stakeholders to communicate about the data and establish data contracts: "Here is what we know to be true about the data, and we want to ensure that continues to be the case." This will help to increase transparency and increase trust in the ML pipeline. 

Having established these contracts, the goal should be for data scientists and ml engineers to eventually come up with a set of data "unit-tests" together. In this process they will define constraints (or checks) on the data, convert constraints to computable metrics, and define a reporting metric for when constraints fail. They will need to codify expectations for correct data. Part of this discussion should revolve around whether the schema capabilities in TFDV will be enough. 



# Resources 
---
Papers:
 + Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift
 + A Primer on Data Drift & Drift Detection Techniques: A Technical White Paper 
 + Automatic Large-Scale Data Quality Verification 
 + Automatic Model Monitoring for Data Streams
 + Monitoring and Explainability of Models in Production 
 + Data Validation for Machine Learning 
 + The ML Test Score: A rubric for ml production readiness and technical debt reduction
 + [Detecting and Correcting for Label Shift with Black Box Predictors](https://arxiv.org/abs/1802.03916)

Products: 
 + greatexpectations 
    + greatexpectations.io/blog/ml-ops-data-quality/
    + greatexpectations.io/blog/ml-ops-great-expectations/
 + seldon alibi-detect
    + docs.seldon.io/projects/alibi-detect/en/latest/methods/ksdrift.html
 + dataiku
    + https://pages.dataiku.com/data-drift-detection-techniques

Books: 
 + [Building ML Pipelines (Chapter 4)](https://www.oreilly.com/library/view/building-machine-learning/9781492053187/)

Blogs
 + https://blog.fastforwardlabs.com/2019/08/28/two-approaches-for-data-validation-in-ml-production.html
 + https://www.depends-on-the-definition.com/data-validation-for-nlp/ 

More on EDA for NLP 
 + https://towardsdatascience.com/nlp-part-3-exploratory-data-analysis-of-text-data-1caa8ab3f79d
 + https://neptune.ai/blog/exploratory-data-analysis-natural-language-processing-tools
 + https://www.analyticsvidhya.com/blog/2020/04/beginners-guide-exploratory-data-analysis-text-data/

