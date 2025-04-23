# PAI: Learnable Prompt as Pseudo-Imputation

Welcome to the official GitHub repository for PAI (Learnable Prompt as Pseudo-Imputation)!

This is the official code for paper: **Learnable Prompt as Pseudo-Imputation: Rethinking the Necessity of Traditional EHR Data Imputation in Downstream Clinical Prediction**

**📢 News: this work has been accepted at the KDD 2025 !**

**If you find our project interesting or helpful, we would appreciate it if you could give us a star! Your support is a tremendous encouragement to us!**


## Overview

Analyzing the health status of patients based on Electronic Health Records (EHR) is a fundamental research problem in medical informatics. The presence of extensive missing values in EHR makes it challenging for deep neural networks (DNNs) to directly model the patient’s health status. Existing DNNs training protocols, including Impute-then-Regress Procedure and Jointly Optimizing of Impute-n-Regress Procedure, require the additional imputation models to reconstruction missing values. However, Impute-then-Regress Procedure introduces the risk of injecting imputed, non-real data into downstream clinical prediction tasks, resulting in power loss, biased estimation, and poorly performing models, while Jointly Optimizing of Impute-n-Regress Procedure is also difficult to generalize due to the complex optimization space and demanding data requirements. Inspired by the recent advanced literature of learnable prompt in the fields of NLP and CV, in this work, we rethought the necessity of the imputation model in downstream clinical tasks, and proposed Learnable Prompt as Pseudo-Imputation (PAI) as a new training protocol to assist EHR analysis. PAI no longer introduces any imputed data but constructs a learnable prompt to model the implicit preferences of the downstream model for missing values, resulting in a significant performance improvement for all state-of-the-arts EHR analysis models on four real-world datasets across two clinical prediction tasks. Further experimental analysis indicates that PAI exhibits higher robustness in situations of data insufficiency and high missing rates. More importantly, as a plug-and-play protocol, PAI can be easily integrated into any existing or even imperceptible future EHR analysis models.

## Install Environment

We use conda to manage the environment.
Please refer to the following steps to install the environment:

```sh
conda create -n PAI python=3.11 -y
conda activate PAI
pip install -r requirements.txt
```

## Download Datasets

- [x] [mimic-iii](https://physionet.org/content/mimiciii/1.4/)
- [x] [mimic-iv](https://www.physionet.org/content/mimiciv/2.2/)
- [x] [eICU](https://physionet.org/content/eicu-crd/2.0/)
- [x] [sepsis](https://physionet.org/content/challenge-2019/1.0.0/)
- [x] [cdsl](https://www.hmhospitales.com/prensa/notas-de-prensa/comunicado-covid-data-save-lives) 

## Running

To run the code, simply execute the following command:

```sh
python train_model.py
```

And we have provided some arguments to run the code with different EHR analysis models, different datasets, and different prediction tasks. You can enable all these augments with the following command:

```sh
python train_model.py --task "your_task_name" --model "your_model_name" --data "your_dataset_name" --fill
```

The following table lists all the available arguments, their default values and options for each argument:

| Argument | Options |
|---|---|
| `--task` | `outcome` (mortality), `los`, `readmission` |
| `--model` | `rnn`, `lstm`, `gru`, `transformer`, `retain`, `concare`, `m3care`, `safari` |
| `--data` | `mimic` (mimic-iv), `cdsl`, `sepsis`, `eicu`, `mimiciii` (mimic-iii) |

You can choose to remove the `--fill` from the command to close PAI

## Publication

```
@inproceedings{Liao2025PAI,
  title={Learnable Prompt as Pseudo-Imputation: Rethinking the Necessity of Traditional EHR Data Imputation in Downstream Clinical Prediction},
  author={Liao, Weibin and Zhu, Yinghao and Zhang, Zhongji and Wang, Yuhang and Wang, Zixiang and Chu, Xu and Wang, Yasha and Ma, Liantao},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1},
  pages={765--776},
  year={2025}
}
```



