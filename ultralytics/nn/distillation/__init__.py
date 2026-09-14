"""Optical-to-SAR Prototype-Referenced Cross-Modal Distillation."""

from .crc import CRCOutput, CrossModalResponseCalibration
from .distiller import DistillOutput, PrototypeReferencedDistiller
from .icd import ICDOutput, InstanceConfigurationDistillation
from .object_support import ObjectSupportSampler, SupportGeometry
from .prototype_bank import TeacherPrototypeBank
from .spd import SPDOutput, SemanticPrototypeDistillation

__all__ = (
    "CRCOutput",
    "CrossModalResponseCalibration",
    "DistillOutput",
    "ICDOutput",
    "InstanceConfigurationDistillation",
    "ObjectSupportSampler",
    "PrototypeReferencedDistiller",
    "SPDOutput",
    "SemanticPrototypeDistillation",
    "SupportGeometry",
    "TeacherPrototypeBank",
)
