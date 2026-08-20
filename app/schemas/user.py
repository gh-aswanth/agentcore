from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    # bcrypt truncates past 72 bytes; capping here turns a runtime error into a
    # 422 with a clear message.
    password: str = Field(min_length=8, max_length=72)


class UserOut(BaseModel):
    id: str
    email: str

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
