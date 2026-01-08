from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, Query

from app.services.storage_service import SUPPORTED_KINDS, reset, save_file

router = APIRouter()


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(..., description="CSV or Excel dataset"),
    kind: Optional[str] = Query(None, description=f"Optional dataset kind override ({', '.join(sorted(SUPPORTED_KINDS))})"),
) -> Dict[str, Any]:
    content = await file.read()
    try:
        detected, count = save_file(content, file.filename, kind=kind)
        return {"kind": detected, "rows": count}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:  # pragma: no cover - catch-all for upload errors
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/upload/batch")
async def upload_datasets(
    files: List[UploadFile] = File(..., description="Multiple CSV or Excel datasets"),
    kind: Optional[str] = Query(None, description=f"Optional dataset kind override ({', '.join(sorted(SUPPORTED_KINDS))})"),
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for file in files:
        content = await file.read()
        try:
            detected, count = save_file(content, file.filename, kind=kind)
            results.append({"file": file.filename, "kind": detected, "rows": count})
        except ValueError as exc:
            errors.append({"file": file.filename, "status": 400, "error": str(exc)})
        except RuntimeError as exc:
            errors.append({"file": file.filename, "status": 502, "error": str(exc)})
        except Exception as exc:  # pragma: no cover - catch-all for upload errors
            errors.append({"file": file.filename, "status": 500, "error": str(exc)})

    if errors and not results:
        first = errors[0]
        raise HTTPException(status_code=first["status"], detail=first["error"])

    return {"uploaded": results, "errors": errors}


@router.delete("/reset")
async def reset_datasets(
    kind: Optional[str] = Query(None, description="Reset a specific dataset (matches, deliveries, players, wellness, candidates) or all when omitted"),
) -> Dict[str, Any]:
    try:
        cleared = reset(kind)
        return {"reset": cleared}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
