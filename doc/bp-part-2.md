# MLOps: A Complete and Hands-on Introduction - Part 2

In the first part of this series we could explore the idea of MLOps and also some frameworks and other tools that can be useful when we want to apply this concept in our daily bases. Although we saw some tools there, I think it deserves some hands-on examples on how it can be used on a real data science project.

Here we will be using GitHub for hosting the repository, Data Version Control (DVC) for versioning our artifacts, such as our models and other files like metrics information and, of course, our dataset. DVC is also helpful to create the Data Science pipeline we need to go through in order to get our dataset, perform some preprocessing in it, train a model and, finally, validate it.

Following the MLOps principles, we are going to use GitHub Actions to automate all our development workflow, and we will integrate the repository with Heroku in order to automatically deploy or API and serve our model.
