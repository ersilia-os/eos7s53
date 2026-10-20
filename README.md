# MtbAIM Antitubercular Activity

Predicts activity against Mycobacterium tuberculosis H37Rv, modelling drug-susceptible and drug-resistant isolates separately: probability of activity (ChEMBL MIC below 5 uM labelled active, above 15 uM inactive) plus pMIC for MIC, MIC90 and MIC99, each with Tanimoto similarity to its training set. Eight tree-based models use RDKit descriptors and Morgan fingerprints, trained on ChEMBL 37 data (25,645 susceptible and 3,749 resistant compounds); scaffold-held-out AUROC is 0.86 and 0.84. Validation is retrospective only.



## Information
### Identifiers
- **Ersilia Identifier:** `eos7s53`
- **Slug:** `mtbaim`

### Domain
- **Task:** `Annotation`
- **Subtask:** `Activity prediction`
- **Biomedical Area:** `Tuberculosis`, `Antimicrobial resistance`
- **Target Organism:** `Mycobacterium tuberculosis`
- **Tags:** `Antimicrobial activity`, `MIC90`, `ChEMBL`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `16`
- **Output Consistency:** `Fixed`
- **Interpretation:** Probabilities of activity and pMIC (-log10 molar MIC) for susceptible and resistant M. tuberculosis, each paired with training-set Tanimoto similarity.

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| ds_prob_active | float | high | Probability of activity against drug-susceptible M. tuberculosis H37Rv from an ExtraTrees classifier (active is MIC below 5 uM and inactive above 15 uM in training; the authors' calibrated decision threshold is 0.4375) |
| dr_prob_active | float | high | Probability of activity against drug-resistant M. tuberculosis isolates from a histogram gradient boosting classifier (active is MIC below 5 uM and inactive above 15 uM in training; the authors' calibrated decision threshold is 0.6511) |
| ds_pmic | float | high | Predicted pMIC (minus log10 of the MIC in mol/L; higher means more potent) against drug-susceptible M. tuberculosis from an ExtraTrees regressor trained on 21683 compounds (pooled out-of-fold RMSE 0.714) |
| ds_pmic90 | float | high | Predicted pMIC (minus log10 of the MIC90 in mol/L; higher means more potent) against drug-susceptible M. tuberculosis from a LightGBM regressor trained on 3754 compounds (pooled out-of-fold RMSE 0.682) |
| ds_pmic99 | float | high | Predicted pMIC (minus log10 of the MIC99 in mol/L; higher means more potent) against drug-susceptible M. tuberculosis from an XGBoost regressor trained on 1846 compounds (pooled out-of-fold RMSE 0.670) |
| dr_pmic | float | high | Predicted pMIC (minus log10 of the MIC in mol/L; higher means more potent) against drug-resistant M. tuberculosis from an ExtraTrees regressor trained on 3220 compounds (pooled out-of-fold RMSE 0.729) |
| dr_pmic90 | float | high | Predicted pMIC (minus log10 of the MIC90 in mol/L; higher means more potent) against drug-resistant M. tuberculosis from an ExtraTrees regressor trained on 561 compounds (pooled out-of-fold RMSE 0.784) |
| dr_pmic99 | float | high | Predicted pMIC (minus log10 of the MIC99 in mol/L; higher means more potent) against drug-resistant M. tuberculosis from an ExtraTrees regressor trained on 180 compounds (pooled out-of-fold RMSE 0.604); few training compounds so treat with caution |
| ds_clf_sim | float | high | Maximum Tanimoto similarity (Morgan radius 2 with 2048 bits and chirality) between the query and the 25645 compounds used to train the susceptible activity classifier; low values mean the query is far from the training chemistry |
| dr_clf_sim | float | high | Maximum Tanimoto similarity (Morgan radius 2 with 2048 bits and chirality) between the query and the 3749 compounds used to train the resistant activity classifier; low values mean the query is far from the training chemistry |

_10 of 16 columns are shown_
### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`

### Resource Consumption


### References
- **Source Code**: [https://github.com/jasdeep002/MtbAIM](https://github.com/jasdeep002/MtbAIM)
- **Publication**: [https://doi.org/10.64898/2026.09.28.754881](https://doi.org/10.64898/2026.09.28.754881)
- **Publication Type:** `Preprint`
- **Publication Year:** `2026`
- **Ersilia Contributor:** [arnaucoma24](https://github.com/arnaucoma24)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [MIT](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos7s53
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos7s53
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
