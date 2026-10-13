import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import joblib
from sqlalchemy.orm import Session
from skynet.config import config
from skynet.db.schema import ModelVersion

class ModelRegistry:
    """Manages model artifact persistence, SHA-256 verification, and database version registration."""

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or config.models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save_and_register_model(
        self,
        session: Session,
        model_obj: Any,
        version_tag: str,
        model_type: str,
        training_cutoff: datetime,
        hyperparameters: Dict[str, Any],
        metadata_info: Optional[Dict[str, Any]] = None
    ) -> ModelVersion:
        """Serializes model, computes hash, and logs to database."""
        timestamp_str = training_cutoff.strftime("%Y%m%d_%H%M%S")
        version_id = f"{model_type}_{version_tag}_{timestamp_str}"
        artifact_filename = f"{version_id}.joblib"
        artifact_path = self.models_dir / artifact_filename

        # Save artifact
        joblib.dump(model_obj, artifact_path)

        # Compute SHA256 checksum
        hasher = hashlib.sha256()
        with open(artifact_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        model_hash = hasher.hexdigest()

        # Database record
        db_model = session.query(ModelVersion).filter_by(id=version_id).first()
        if not db_model:
            db_model = ModelVersion(
                id=version_id,
                version_tag=version_tag,
                model_type=model_type,
                artifact_path=str(artifact_path),
                model_hash=model_hash,
                training_cutoff=training_cutoff,
                hyperparameters=hyperparameters,
                metadata_info=metadata_info or {},
            )
            session.add(db_model)
            session.flush()

        return db_model

    def load_model(self, artifact_path: str) -> Any:
        """Loads serialized model object."""
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
        return joblib.load(path)
