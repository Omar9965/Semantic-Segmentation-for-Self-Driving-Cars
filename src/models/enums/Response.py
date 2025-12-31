from enum import Enum
from .ProcessingEnum import ProcessingEnum

class Response(Enum):
    max_file_size = 524288000  # 500 MB
    allowed_types = [ext.value for ext in ProcessingEnum]

    File_type_not_supported = (
        f"File type is not supported. Types allowed are "
        f"{', '.join(allowed_types)}"
    )

    File_too_large = (
        f"File is too large. Max file size is "
        f"{max_file_size // (1024 * 1024)}MB"
    )

    File_Uploaded_Successfully = "File was Uploaded Successfully"
    File_Upload_Failed = "File Upload Failed"
