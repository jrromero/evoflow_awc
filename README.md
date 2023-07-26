# EvoFlow: Grammar-guided evolutionary automated workflow composition with domain-specific operators and ensemble diversity
_Supplementary material_ 

## Authors
- Rafael Barbudo (a,b)
- Aurora Ramírez (a,b)
- José Raúl Romero (a,b - corresponding author)

(a) Department of Computer Science and Numerical Analysis, University of Córdoba, 14071, Córdoba, Spain
(b) Andalusian Research Institute in Data Science and Computational Intelligence (DaSCI), Córdoba, Spain


## Abstract 
Extracting valuable and novel insights from raw data is a complex process that encompasses various intricate steps. Some of them are repetitive and time-consuming, lending themselves to (partially) automation. AutoML (Automated Machine Learning), a research field focused on methods that (semi-)automate the learning process, includes diverse tasks, such as algorithm selection and hyper-parameter optimisation for a given dataset. However, these methods predominantly concentrate on a single phase, typically model building. In this context, some authors have proposed methods for composing workflows that encompass preprocessing steps, thereby leading to more comprehensive support to the knowledge discovery process. This paper introduces EvoFlow, a grammar-based evolutionary approach for automating workflow composition. By employing a grammar-based approach, our method achieves flexibility, allowing practitioners to adapt EvoFlow to their specific requirements. Furthermore, EvoFlow incorporates two novel characteristics. Firstly, it defines a set of genetic operators explicitly tailored for workflow composition, considering both the optimisation of workflow structure and hyper-parameters. Secondly, it introduces an update mechanism that promotes diversity among individuals and constructs an ensemble of workflows capable of generating diverse predictions, consequently mitigating overfitting. To assess the performance of EvoFlow, we conduct empirical validation using a collection of classification benchmarks. Initially, we compare different variations of EvoFlow to confirm that the proposed enhancements yield improved results. Subsequently, we compare the results obtained with those achieved by other approaches in the literature, including diverse techniques. These comparisons are supported by statistical analyses, revealing that the use of specific genetic operators and the update mechanism significantly outperform the state-of-the-art in terms of predictive performance.

## Supplementary material

### Datasets and partitions

Collection of datasets used in the experimentation, together with their train/test partitions

### Evoflow

This folder includes the source code, configurations, grammar, execution scripts and raw results of our proposal, EvoFlow.

### Baselines

The following folders and files contain requirements files and raw results obtained with the baseline

#### AutoSklearn

Raw results obtained in AutoSklearn, including the “requirements.txt” file

url: https://github.com/automl/auto-sklearn
version: 0.12.0

#### TPOT

Raw results obtained in TPOT, including the “requirements.txt” file

url: https://github.com/EpistasisLab/tpot
version: 0.11.6.post3

#### RECIPE

Raw results obtained in RECIPE, including the “requirements.txt” file

url: https://github.com/laic-ufmg/Recipe
version: n/a (version not updated as of July 2023)

#### MLPlan

Raw results obtained in RECIPE, including the “pom.xml” file (Java language)

url: https://starlibs.github.io/AILibs/projects/mlplan/#installation
version: 0.2.7


