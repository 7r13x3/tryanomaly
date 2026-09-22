"""
TryAnomaly — Config loader
"""
import sys
from dataclasses import dataclass, field
from typing import List

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class DataCfg:
    dataset: str = "nsl-kdd"
    path: str = "data/"
    test_size: float = 0.2
    random_state: int = 42
    download: bool = True


@dataclass
class PreprocessCfg:
    scaler: str = "standard"
    encode_categorical: str = "onehot"
    drop_duplicates: bool = True
    handle_missing: str = "drop"


@dataclass
class IsoForestCfg:
    n_estimators: int = 100
    contamination: float = 0.1
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class SvmCfg:
    kernel: str = "rbf"
    nu: float = 0.1
    gamma: str = "scale"


@dataclass
class AutoencoderCfg:
    hidden_dims: List[int] = field(default_factory=lambda: [32, 16, 8])
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 0.001
    threshold_pct: int = 95


@dataclass
class EvaluationCfg:
    metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "precision",
                                  "recall", "f1", "roc_auc"]
    )


@dataclass
class VisualizationCfg:
    pca: bool = True
    tsne: bool = True
    confusion_matrix: bool = True
    roc_curve: bool = True
    feature_importance: bool = True


@dataclass
class ReportCfg:
    output: str = "report.html"


@dataclass
class Config:
    data: DataCfg = field(default_factory=DataCfg)
    preprocess: PreprocessCfg = field(default_factory=PreprocessCfg)
    isolation_forest: IsoForestCfg = field(default_factory=IsoForestCfg)
    oneclass_svm: SvmCfg = field(default_factory=SvmCfg)
    autoencoder: AutoencoderCfg = field(default_factory=AutoencoderCfg)
    evaluation: EvaluationCfg = field(default_factory=EvaluationCfg)
    visualization: VisualizationCfg = field(default_factory=VisualizationCfg)
    report: ReportCfg = field(default_factory=ReportCfg)


def load(path: str) -> Config:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    def g(section, key, default):
        return data.get(section, {}).get(key, default)

    return Config(
        data=DataCfg(
            dataset=g("data", "dataset", "nsl-kdd"),
            path=g("data", "path", "data/"),
            test_size=float(g("data", "test_size", 0.2)),
            random_state=int(g("data", "random_state", 42)),
            download=bool(g("data", "download", True)),
        ),
        preprocess=PreprocessCfg(
            scaler=g("preprocess", "scaler", "standard"),
            encode_categorical=g("preprocess", "encode_categorical", "onehot"),
            drop_duplicates=bool(g("preprocess", "drop_duplicates", True)),
            handle_missing=g("preprocess", "handle_missing", "drop"),
        ),
        isolation_forest=IsoForestCfg(
            n_estimators=int(g("isolation_forest", "n_estimators", 100)),
            contamination=float(g("isolation_forest", "contamination", 0.1)),
            random_state=int(g("isolation_forest", "random_state", 42)),
            n_jobs=int(g("isolation_forest", "n_jobs", -1)),
        ),
        oneclass_svm=SvmCfg(
            kernel=g("oneclass_svm", "kernel", "rbf"),
            nu=float(g("oneclass_svm", "nu", 0.1)),
            gamma=g("oneclass_svm", "gamma", "scale"),
        ),
        autoencoder=AutoencoderCfg(
            hidden_dims=list(g("autoencoder", "hidden_dims", [32, 16, 8])),
            epochs=int(g("autoencoder", "epochs", 50)),
            batch_size=int(g("autoencoder", "batch_size", 256)),
            learning_rate=float(g("autoencoder", "learning_rate", 0.001)),
            threshold_pct=int(g("autoencoder", "threshold_pct", 95)),
        ),
        evaluation=EvaluationCfg(
            metrics=list(g("evaluation", "metrics",
                           ["accuracy", "precision", "recall",
                            "f1", "roc_auc"])),
        ),
        visualization=VisualizationCfg(
            pca=bool(g("visualization", "pca", True)),
            tsne=bool(g("visualization", "tsne", True)),
            confusion_matrix=bool(g("visualization", "confusion_matrix", True)),
            roc_curve=bool(g("visualization", "roc_curve", True)),
            feature_importance=bool(g("visualization",
                                       "feature_importance", True)),
        ),
        report=ReportCfg(
            output=g("report", "output", "report.html"),
        ),
    )
