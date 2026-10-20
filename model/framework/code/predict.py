"""MtbAIM inference for the Ersilia Model Hub.

Adapted from the production script (``Step30_unseen_prediction.py``) released by
the MtbAIM authors, Singh and Singh, 2026, https://doi.org/10.64898/2026.09.28.754881.
Weights and engine are distributed under the MIT licence at
https://huggingface.co/datasets/jasdeep002/MtbAIM (commit c686d24d54bccbae7fd4be36e5108e9ecb1536a1).

Only inference is kept. The eight final models are restored from ``model/checkpoints``
(fetched at build time with eosvc) and applied to the locked feature set of 217 RDKit
descriptors plus a 2048-bit binary Morgan fingerprint (radius 2, chirality on).
For every prediction we also report the maximum Tanimoto similarity between the query
and the training set of the model that made it.
"""

import csv
import gc
import gzip
import json
import os
import warnings

import joblib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

# Needed so joblib can restore the LightGBM and XGBoost models.
import lightgbm  # noqa: F401
import xgboost  # noqa: F401

CHECKPOINTS = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints")
)

N_RDKIT_DESCRIPTORS = 217
MORGAN_BITS = 2048
N_FEATURES = N_RDKIT_DESCRIPTORS + MORGAN_BITS
# Molecules featurised and scored together. Models are loaded one at a time for each
# chunk (the susceptible MIC regressor alone needs about 3 GB in memory), so a large
# chunk keeps the number of model reloads small.
CHUNK_SIZE = 20000

# Same parameters as the historical GetMorganFingerprintAsBitVect(radius=2, nBits=2048,
# useChirality=True) used to build the training features.
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=MORGAN_BITS, includeChirality=True
)

CLASSIFIERS = ("DS", "DR")
ENDPOINTS = ("MIC", "MIC90", "MIC99")

# Prediction order. Similarity columns follow the same order.
TASKS = (
    ["%s_CLASSIFICATION" % ds for ds in CLASSIFIERS]
    + ["%s_%s_REGRESSION" % (ds, ep) for ds in CLASSIFIERS for ep in ENDPOINTS]
)

COLUMNS = (
    ["ds_prob_active", "dr_prob_active"]
    + ["%s_%s" % (ds.lower(), ep.lower().replace("mic", "pmic", 1)) for ds in CLASSIFIERS for ep in ENDPOINTS]
    + ["ds_clf_sim", "dr_clf_sim"]
    + ["%s_%s_sim" % (ds.lower(), ep.lower().replace("mic", "pmic", 1)) for ds in CLASSIFIERS for ep in ENDPOINTS]
)


def _model_path(name):
    return os.path.join(CHECKPOINTS, "models", name)


def _model_file(task):
    ds, rest = task.split("_", 1)
    if rest == "CLASSIFICATION":
        return "29_%s_active_classifier.joblib" % ds
    return "29_%s_%s_pMIC_regressor.joblib" % (ds, rest.replace("_REGRESSION", ""))


def _parse_feature_contract(feature_columns):
    """Map the frozen feature names to RDKit descriptor functions and Morgan bits."""
    lookup = {str(name): fn for name, fn in Descriptors._descList}
    rdkit_specs, morgan_specs = [], []
    for pos, col in enumerate(feature_columns):
        low = col.lower()
        if low.startswith("rdkit_"):
            name = col.split("_", 1)[1]
            if name not in lookup:
                hits = [n for n in lookup if n.lower() == name.lower()]
                if len(hits) != 1:
                    raise RuntimeError("Cannot map frozen feature %r to an RDKit descriptor." % col)
                name = hits[0]
            rdkit_specs.append((pos, lookup[name]))
        elif low.startswith("morgan_"):
            morgan_specs.append((pos, int(col.split("_")[-1])))
        else:
            raise RuntimeError("Unsupported frozen feature name %r." % col)
    if len(rdkit_specs) != N_RDKIT_DESCRIPTORS or len(morgan_specs) != MORGAN_BITS:
        raise RuntimeError("Frozen feature schema does not match 217 descriptors + 2048 bits.")
    return rdkit_specs, morgan_specs


def _sanitize(x):
    """Inf and values outside the float32 range become NaN; never clip."""
    x = np.asarray(x, dtype=np.float64).copy()
    finite = np.isfinite(x)
    x[~finite] = np.nan
    x[finite & (np.abs(x) > float(np.finfo(np.float32).max))] = np.nan
    return x.astype(np.float32)


def _mol_from_smiles(smiles):
    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is not None:
            Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


class MtbAimPredictor:
    def __init__(self):
        with open(os.path.join(CHECKPOINTS, "29_feature_recipe.json")) as f:
            self.feature_columns = [str(c) for c in json.load(f)["ordered_features"]]
        if len(self.feature_columns) != N_FEATURES:
            raise RuntimeError("Expected %d ordered features." % N_FEATURES)
        self.rdkit_specs, self.morgan_specs = _parse_feature_contract(self.feature_columns)

        self.training_fps = self._load_training_fingerprints()

    def _load_bundle(self, task):
        bundle = joblib.load(_model_path(_model_file(task)))
        if [str(c) for c in bundle["feature_columns"]] != self.feature_columns:
            raise RuntimeError("%s was trained on a different feature order." % task)
        # Threaded ExtraTrees prediction adds the trees up in a scheduling-dependent order,
        # which changes the last digits between runs. One thread keeps the output identical.
        if isinstance(bundle["model"], (ExtraTreesClassifier, ExtraTreesRegressor)):
            bundle["model"].n_jobs = 1
        return bundle

    def _load_training_fingerprints(self):
        fps = {task: [] for task in TASKS}
        seen = {task: set() for task in TASKS}
        path = os.path.join(CHECKPOINTS, "29_training_reference_smiles.csv.gz")
        with gzip.open(path, "rt", newline="") as f:
            for row in csv.DictReader(f):
                task, smi = row["task"], row["canonical_smiles"]
                if task not in fps or not smi or smi in seen[task]:
                    continue
                seen[task].add(smi)
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fps[task].append(_MORGAN.GetFingerprint(mol))
        return fps

    def _featurize(self, mol):
        x = np.full(N_FEATURES, np.nan, dtype=np.float64)
        for pos, fn in self.rdkit_specs:
            try:
                x[pos] = float(fn(mol))
            except Exception:
                x[pos] = np.nan
        fp = _MORGAN.GetFingerprint(mol)
        bits = np.zeros(MORGAN_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, bits)
        for pos, bit in self.morgan_specs:
            x[pos] = float(bits[bit])
        return _sanitize(x), fp

    @staticmethod
    def _score(task, bundle, X):
        Xt = bundle["imputer"].transform(X).astype(np.float32, copy=False)
        model = bundle["model"]
        if task.endswith("_CLASSIFICATION"):
            classes = list(model.classes_)
            positive = bundle.get("positive_class", 1)
            return np.asarray(model.predict_proba(Xt)[:, classes.index(positive)], dtype=float)
        return np.asarray(model.predict(Xt), dtype=float)

    def predict(self, smiles_list):
        """Return an (n, 16) float array in COLUMNS order. Invalid SMILES give NaN rows."""
        n = len(smiles_list)
        out = np.full((n, len(COLUMNS)), np.nan, dtype=np.float64)
        n_tasks = len(TASKS)
        for start in range(0, n, CHUNK_SIZE):
            idx, rows, fps = [], [], []
            for i in range(start, min(start + CHUNK_SIZE, n)):
                mol = _mol_from_smiles(smiles_list[i])
                if mol is None:
                    continue
                x, fp = self._featurize(mol)
                idx.append(i)
                rows.append(x)
                fps.append(fp)
            if not idx:
                continue
            X = np.vstack(rows)
            for t, task in enumerate(TASKS):
                bundle = self._load_bundle(task)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out[idx, t] = self._score(task, bundle, X)
                del bundle
                gc.collect()
                reference = self.training_fps[task]
                out[idx, n_tasks + t] = [
                    max(DataStructs.BulkTanimotoSimilarity(fp, reference)) for fp in fps
                ]
        return out
