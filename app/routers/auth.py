from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select

from app.core.dependencies import DbSession
from app.core.logging import bind, get_logger
from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.schemas.errors import UNPROCESSABLE, _response
from app.schemas.user import TokenResponse, UserCreate

router = APIRouter(prefix="/auth", tags=["auth"])
log = get_logger(__name__)


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register and receive a token",
    description="Creates the account and returns a bearer token immediately — there is "
                "no email verification step. The token scopes every other endpoint to "
                "this user.",
    responses={
        409: _response("That email is already registered.", (409, "Email already registered")),
        422: UNPROCESSABLE,
    },
)
async def register(body: UserCreate, db: DbSession):
    existing = (
        await db.execute(select(User).where(User.email == body.email))
    ).scalar_one_or_none()
    if existing is not None:
        # No email in the record: an auth log is exactly where personal data
        # accumulates, and the user id identifies them just as well.
        await log.awarning("register_rejected_duplicate", user_id=existing.id)
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")

    user = User(email=body.email, hashed_password=hash_password(body.password))
    db.add(user)
    await db.commit()

    bind(user_id=user.id)
    await log.ainfo("user_registered")
    return TokenResponse(access_token=create_access_token(user.id))


@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Exchange credentials for a token",
    description="Standard OAuth2 password flow, so the *Authorize* button in the docs "
                "works against it directly.",
    responses={
        401: _response("Unknown email or wrong password — the same response for both, "
                       "so the endpoint cannot be used to discover which emails exist.",
                       (401, "Incorrect email or password")),
        422: UNPROCESSABLE,
    },
)
async def login(form: Annotated[OAuth2PasswordRequestForm, Depends()], db: DbSession):
    user = (
        await db.execute(select(User).where(User.email == form.username))
    ).scalar_one_or_none()
    # Same 401 whether the email is unknown or the password is wrong.
    if user is None or not verify_password(form.password, user.hashed_password):
        await log.awarning("login_failed", reason="unknown_email" if user is None
                           else "bad_password")
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    bind(user_id=user.id)
    await log.ainfo("login_succeeded")
    return TokenResponse(access_token=create_access_token(user.id))
