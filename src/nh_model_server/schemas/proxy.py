from pydantic import BaseModel

class DiscoverBody(BaseModel):
    node_key: str

class DiscoverResponse(BaseModel):
    success: bool
    message: str
    address: str

class RelayResponse(BaseModel):
    success: bool
    message: str