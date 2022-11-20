# Emotion prediction as inference over a generative theory of mind

![Generative Model of Inferred Appraisals](website/iaa_model_diagram.png)

**ABSTRACT** From sparse descriptions of events, observers can make systematic nuanced predictions of the emotions of people involved. We propose a formal model of emotion predictions, in the context of a public high-stakes social dilemma. This model uses inverse planning to infer a person's beliefs and preferences based on their action, including social preferences for equity and for maintaining a good reputation. The model then combines these inferred preferences with the outcome to compute 'appraisals': whether the outcome conformed to the expectations and fulfilled the preferences. We learn functions mapping inferred appraisals to emotion labels, allowing the model to match human observers' quantitative predictions of twenty emotions, including joy, relief, guilt, and envy. Model comparison indicates that inferred monetary preferences are not sufficient to explain observers' emotion predictions; inferred social preferences are factored into predictions for nearly every emotion. The model can also account for the flexibility of emotion attributions: both human observers and the model consistently use minimal individualizing information to adjust predictions for how different players will respond to the same event. This research provides a foundation for characterizing the shared conceptual structure of people's intuitive causal theories of emotion.

## Project repository

This repository provides all of the raw behavioral data, analyses, and models related to the submitted manuscript.

## Contents of the project

- `code` - models and analyses
- `datain` - the raw behavioral data collected from human observers
- `experiments` - mTurk experiments used to collect the behavioral data in `datain/`
- `manuscript` - LaTeX code that generates the submitted manuscript (not yet uploaded)
- `website` - the project website (not yet uploaded)



## Running the project

You can install the dependencies necessary to regenerate the results and figures using **(1)** [a Docker container](#1-as-a-docker-container) **(2)** [a conda environment](#2-as-a-conda-environment) **(3)** [a pip `requirements` specification](#3-as-a-pip-requirements-specification).

NB Running this model from scratch is prohibitively compute-heavy outside of a High Performing Computing cluster. We have cached the model at various checkpoints to make it easy to explore the results on a personal computer. To make use of the cached model, download and uncompress [inferredappraisals-cacheddata.zip](https://www.dropbox.com/s/var7vtxkdpzwcki/inferredappraisals-cacheddata.zip?dl=0). Then place the `dataOut` directory containing the cached `*.pkl` files in the local project folder that you clone/fork from this repository (e.g. `inferredappraisals/dataOut/`) .

### 1. As a Docker container 

Requires [Docker](https://www.docker.com/). The image includes [WebPPL](https://github.com/probmods/webppl) and [TeX Live](https://www.tug.org/texlive/). 

NB Docker is finicky about the cpu architecture. The example below builds an image optimized for `arm64` processors. For an example of building an image for `amd64` processors, see `.devcontainer/Dockerfile`.

```bash
### Clone git repo to the current working directory
git clone --branch main https://github.com/daeh/inferred-appraisals.git inferred-appraisals

### Enter the new directory
cd inferred-appraisals

### (optional but recommended)
### Add the dataOut/ directory that you downloaded in order to use the cached model

### Build Docker Image
docker build --tag iaaimage .

### Run Container (optional to specify resources like memory and cpus)
docker run --rm --name=iaamodel \
    --memory 12GB --cpus 4 --platform=linux/arm64 \
    --volume $(pwd)/:/projhost/ \
    iaaimage /projhost/code/react_main.py --projectdir /projhost/
```

The container tag is arbitrary (you can replace `iaaimage` with a different label).

### 2. As a conda environment

Requires [conda](https://docs.conda.io/en/latest/), [conda-lock](https://github.com/conda-incubator/conda-lock), and a local installation of [TeX Live](https://www.tug.org/texlive/). If you want to run the inverse planning models, you need to have the [WebPPL](https://github.com/probmods/webppl) executable in your `PATH` with the [webppl-json](https://github.com/stuhlmueller/webppl-json) add-on.

The example below uses the `conda-lock.yml` file to create an environment where the package versions are pinned to this project's specifications, which is recommend for reproducibility. If the lock file cannot resolve the dependencies for your system, you can use the `environment.yml` file to create an environment with the latest package versions. Simply replace the `conda-lock install ...` line with `conda env create -f environment.yml`.

```bash
### Clone git repo to the current working directory
git clone --branch main https://github.com/daeh/inferred-appraisals.git inferred-appraisals

### Enter the new directory
cd inferred-appraisals

### (optional but recommended)
### Add the dataOut/ directory that you downloaded in order to use the cached model

### Create the conda environment
conda-lock install --name enviaa conda-lock.yml

### Activate the conda environment
conda activate enviaa

### Run the python code
python ./code/react_main.py --projectdir $(pwd)
```

The conda environment name is arbitrary (you can replace `enviaa` with a different label).

### 3. As a `pip` `requirements` specification

If you use a strategy other than [conda](https://docs.conda.io/en/latest/) or [Docker](https://www.docker.com/) to manage python environments, you can install the dependencies using the `requirements.txt` file located in the root directory of the project. You need to have [TeX Live](https://www.tug.org/texlive/) installed locally. If you want to run the inverse planning models, you need to have the [WebPPL](https://github.com/probmods/webppl) executable in your `PATH` with the [webppl-json](https://github.com/stuhlmueller/webppl-json) add-on.

