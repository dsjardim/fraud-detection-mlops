# MLOps: A Complete and Hands-on Introduction

<img src="./images/MLOps.png" alt="MLOps-Cycle" width="600"/>

*Fig. 1: The MLOps cycle. Font: Neal Analytics.*

<!-- ![MLOps-Cycle](./images/MLOps.png) -->

Machine Learning Operations (MLOps) refers to the tools, techniques and practical experiences required to train  Machine Learning (ML) models and deploy and monitor them in production.

The concept of MLOps comes from DevOps, which is the existing practice of more efficiently write, deploy, and manage enterprise applications. It was designed to unite software developers (Devs) and IT operations teams (Ops) to enable better collaboration between them. MLOps shares these aims but adds the Data Scientists and also ML Engineers to the team. 

MLOps became popular due to the rise of the Data Science and ML areas in the past years. The reason for that is that Data Scientists have a strong background in statistics and programming, but they are not used to put ML models into production. This is where ML Engineers comes into picture, they have the knowledge of databases, REST APIs, and a collection of other IT skills that are needed to accomplish this task.

## The benefits of MLOps

### Simplified Deployment

The Data Science stack nowadays provides a lot of programming languages and frameworks to Data Scientists run their experiments. Therefore, MLOps enables ML Engineers to more rapidly deploy models from those many frameworks and languages.

### ML Monitoring

Many MLOps pipelines provide model-specific metrics, detection of data drift for important features, and other core functionality.

### Lifecycle Management

It can help the team to easily test the model and its updates without disrupting business applications.

### Experiment Tracking

In ML, experiment tracking is the process of saving all experiment-related information that you care about for every experiment you run. MLOps helps Data Scientists to better organize these kinds of data using best practices.

### Reproducibility

Reproducibility in ML means that you can repeatedly run your algorithm on certain datasets and obtain the same results no matter what environment (locally or remotely) you execute this experiment. It adds value to any CI/CD pipeline and allows these activities to proceed smoothly, so in-house changes and deployments for clients become routine and not a nightmare. Building a MLOps pipeline helps Data Science teams to reduce errors and ambiguity when the projects move from development to production for example.


## How to implement MLOps:

In any Data Science project, the process of delivering an ML model to production involves steps such as: data extraction, data analysis, data preparation, model training, validation and testing, and finally, model serving and monitoring. The level of automation of these steps defines the maturity of the ML pipeline, which reflects the velocity of training new models given new data or training new models given new implementations. Google recognizes three levels of MLOps, starting from the most common level, which involves no automation, up to automating both ML and CI/CD pipelines.

### Level 0 - Manual Process

At this level, the team may have a state-of-the-art ML model created by a data scientist, but the build and deployment process for ML models is completely manual. So, here we have:

- Script-driven and interactive process
- Disconnect between ML and operations
- No Continuous Integration (CI) and Continuous Deployment (CD)
- Lack of active performance monitoring

ML processes at this level are more related to experiments that are still in early stages, like proof of concepts (POCs) to evaluate an idea of product.

### Level 1 - ML Pipeline Automation

The improvement here in this level is to perform what's called Continuous Training (CT), which is the automated retraining ML models. To do this using new data in production, you need to introduce automated data and model validation steps to the pipeline, as well as pipeline triggers and metadata management. Some key ideas here:

- Rapid experiment
- CT of the model in production
- Modularized code
- CD of models

### Level 2 - CI/CD Pipeline Automation

A MLOps pipeline in level 2 reflects a robust, fully automated CI/CD pipeline system that can deliver reliable, rapid updates on the pipelines in production. With this automated CI/CD system, your data scientists rapidly explore new ideas around feature engineering, model architecture, and hyperparameters. This MLOps setup includes the following components:

- Source control
- Test and build services
- Deployment services
- Model registry
- Feature store
- ML metadata store
- ML pipeline orchestrator

## Tools for MLOps
<img src="./images/MLOps-Stack.png" alt="MLOps-Stack" width="497"/>

*Fig. 2: The MLOps stack. Font: The author.*

<!-- ![MLOps-Stack](./images/MLOps-Stack.png) -->

There are a lot of frameworks that a data scientist can use to build and train a ML model, as well as deploying and monitoring it in production stage using a MLOps pipeline. We present some of these options in Fig. 2.

We can separate them into some specific segments:
- Model Experimenting: Pytorch and Tensorflow
- Experiment Tracking and Report: DVC and CML
- Automation: Jenkins, GitHub Actions, MLflow and Kubeflow

You can use frameworks such as Pytorch or Tensorflow in order to build and train any ML model. In order to better organize your data (datasets and model artifacts), you can use the DVC tool to start versioning it. Finally, you can put this stages all together using some automation framework, such as any of the listed above. Today, this pipeline can be whether set on an on-premise environment or in the cloud, the result will be the same.

We hope to have demystified a little regarding MLOps concept and what a data scientist or a ML engineer does in each step of the MLOps pipeline. In the following blogposts, we will get into more details on how to build a very simple pipeline using a public dataset, and some tools we mentioned before.