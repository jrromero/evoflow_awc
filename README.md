# Automated workflow composition using G3P with domain-specific operators and ensemble diversity
_Supplementary material_ (November 29, 2023)

## Authors
- Rafael Barbudo (a,b)
- Aurora Ramírez (a,b)
- José Raúl Romero (a,b - corresponding author)

(a) Department of Computer Science and Numerical Analysis, University of Córdoba, 14071, Córdoba, Spain

(b) Andalusian Research Institute in Data Science and Computational Intelligence (DaSCI), Córdoba, Spain


## Abstract 
The process of extracting valuable and novel insights from raw data involves a series of complex steps. In the realm of Automated Machine Learning (AutoML), a significant research focus is on automating aspects of this process, specifically tasks like selecting algorithms and optimising their hyper-parameters. A particularly challenging task in AutoML is automatic workflow composition (AWC). AWC aims to identify the most effective sequence of data preprocessing and machine learning algorithms, coupled with their best hyper-parameters, for a specific dataset. However, existing AWC methods are limited in how many and in what ways they can combine algorithms within a workflow.

Addressing this gap, this paper introduces EvoFlow, a grammar-based evolutionary approach for AWC. EvoFlow enhances the flexibility in designing workflow structures, empowering practitioners to select algorithms that best fit their specific requirements. EvoFlow stands out by integrating two innovative features. First, it employs a suite of genetic operators, designed specifically for AWC, to optimise both the structure of workflows and their hyper-parameters. Second, it implements a novel updating mechanism that enriches the variety of predictions made by different workflows. Promoting this diversity helps prevent the algorithm from overfitting. With this aim, EvoFlow builds an ensemble whose workflows differ in their misclassified instances.

To evaluate EvoFlow's effectiveness, we carried out empirical validation using a set of classification benchmarks. We begin with an ablation study to demonstrate the enhanced performance attributable to EvoFlow’s unique components. Then, we compare EvoFlow with other AWC approaches, encompassing both evolutionary and non-evolutionary techniques. Our findings show that EvoFlow's specialised genetic operators and updating mechanism substantially outperform current leading methods in predictive performance. Additionally, EvoFlow is capable of discovering workflow structures that other approaches in the literature have not considered.


## Supplementary material

### Datasets and partitions

Collection of datasets used in the experimentation, together with their train/test partitions

### Evoflow

This folder includes the source code, configurations, grammar, execution scripts and raw results of our proposal, EvoFlow.

### Baselines

The following folders and files contain requirements files and raw results obtained with the baseline

#### AutoSklearn

Raw results obtained in AutoSklearn, including the “requirements.txt” file

- [Download](https://github.com/automl/auto-sklearn)
- _version_: 0.12.0

#### TPOT

Raw results obtained in TPOT, including the “requirements.txt” file

- [Download](https://github.com/EpistasisLab/tpot)
- _version_: 0.11.6.post3

#### RECIPE

Raw results obtained in RECIPE, including the “requirements.txt” file

- [Download](https://github.com/laic-ufmg/Recipe)
- _version_: n/a (version not updated as of July 2023)

#### MLPlan

Raw results obtained in RECIPE, including the “pom.xml” file (Java language)

- [Download](https://starlibs.github.io/AILibs/projects/mlplan/#installation)
- _version_: 0.2.7

### Statistical analysis

This folder includes a complete report with raw results of the conducted statistical analysis, including both the unadjusted and adjusted p-values.

