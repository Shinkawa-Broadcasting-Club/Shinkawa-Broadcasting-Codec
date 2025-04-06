from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from endecode import sbc_encoder as SbcEncoder, sbc_decoder as SbcDecoder
import os
import tempfile
from typing import Optional
import pathlib

# Define FastAPI
app = FastAPI()

# Endpoint [POST] /encode
@app.post('/encode')
async def encode(
    file: UploadFile = File(...),
    q: int = Form(4),
    transfer: str = Form('709')
):
    if not file.filename:
        raise HTTPException(status_code=400, detail='No selected file')
    original_name = pathlib.Path(file.filename).stem
    output_filename = f"{original_name}.sbc"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tempInput:
        inputPath = tempInput.name
        content = await file.read()
        with open(inputPath, 'wb') as f:
            f.write(content)
    outputPath = os.path.join(tempfile.gettempdir(), output_filename)

    try:
        SbcEncoder(inputPath, q, outputPath, transfer)
        return FileResponse(
            path=outputPath,
            filename=output_filename,
            media_type='application/octet-stream'
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        os.remove(inputPath)

# Endpoint [POST] /decode
@app.post('/decode')
async def decode(
    file: UploadFile = File(...)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail='No selected file')
    
    original_name = pathlib.Path(file.filename).stem
    output_filename = f"{original_name}_decoded.mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".sbc") as tempInput:
        inputPath = tempInput.name
        content = await file.read()
        with open(inputPath, 'wb') as f:
            f.write(content)
    outputPath = os.path.join(tempfile.gettempdir(), output_filename)

    try:
        SbcDecoder(inputPath)
        return FileResponse(
            path=outputPath,
            filename=output_filename,
            media_type='video/mp4'
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        os.remove(inputPath)

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)