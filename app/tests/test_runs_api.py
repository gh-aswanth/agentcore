"""T-6 — API surface: auth, tool validation, and user isolation."""
import pytest
from httpx import ASGITransport, AsyncClient

from app.agent.tools import TOOL_REGISTRY
from app.core.dependencies import get_db, get_redis
from app.main import app
from app.models.agent import AgentRun, AgentSession, RunStatus


@pytest.fixture
async def client(db, redis, monkeypatch):
    # The Celery hop is out of scope here; the task itself is covered by the
    # runner tests. Only the enqueue call is stubbed.
    enqueued: list[dict] = []
    monkeypatch.setattr(
        "app.routers.runs.execute_agent_run",
        type(
            "Stub",
            (),
            {"delay": staticmethod(
                lambda run_id, **kwargs: enqueued.append({"run_id": run_id, **kwargs})
            )},
        ),
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_redis] = lambda: redis
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        ac.enqueued = enqueued
        yield ac
    app.dependency_overrides.clear()


async def register(client, email: str) -> dict:
    response = await client.post(
        "/auth/register", json={"email": email, "password": "hunter2hunter2"}
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


async def test_health_is_open(client):
    assert (await client.get("/health")).status_code == 200


async def test_protected_route_requires_a_token(client):
    assert (await client.get("/sessions")).status_code == 401
    bad = await client.get("/sessions", headers={"Authorization": "Bearer nonsense"})
    assert bad.status_code == 401
    assert bad.headers["WWW-Authenticate"] == "Bearer"


async def test_login_returns_a_usable_token(client):
    await register(client, "login@example.com")
    response = await client.post(
        "/auth/token", data={"username": "login@example.com", "password": "hunter2hunter2"}
    )
    assert response.status_code == 200
    token = response.json()["access_token"]
    assert (
        await client.get("/sessions", headers={"Authorization": f"Bearer {token}"})
    ).status_code == 200


async def test_unknown_tool_is_rejected_with_422_naming_the_valid_tools(client):
    headers = await register(client, "tools@example.com")
    response = await client.post(
        "/sessions",
        json={"name": "Bad", "tools_enabled": ["calculator", "nonexistent"]},
        headers=headers,
    )

    assert response.status_code == 422
    error = response.json()["error"]["detail"][0]
    # the failure is located at the offending index, not the whole list
    assert error["loc"] == ["body", "tools_enabled", 1]
    assert error["input"] == "nonexistent"
    assert "calculator" in error["msg"]      # the valid set comes back with the error


async def test_openapi_enumerates_the_registry(client):
    """The allowlist is the request type, so /docs and generated clients get the
    real tool names without anything being written down twice."""
    schema = (await client.get("/openapi.json")).json()
    request_schema = schema["components"]["schemas"]["AgentSessionCreate"]

    assert request_schema["properties"]["tools_enabled"]["items"]["enum"] == sorted(
        TOOL_REGISTRY
    )


async def test_valid_tools_round_trip(client):
    headers = await register(client, "valid-tools@example.com")
    response = await client.post(
        "/sessions",
        json={"name": "Research Assistant", "tools_enabled": ["web_search", "calculator"]},
        headers=headers,
    )

    assert response.status_code == 201
    assert response.json()["tools_enabled"] == ["web_search", "calculator"]


async def test_run_is_accepted_with_202_and_enqueued(client, db):
    headers = await register(client, "runner@example.com")
    session_id = (
        await client.post(
            "/sessions",
            json={"name": "Research Assistant", "tools_enabled": ["calculator"]},
            headers=headers,
        )
    ).json()["id"]

    response = await client.post(
        f"/sessions/{session_id}/run", json={"message": "what is 2+2"}, headers=headers
    )

    assert response.status_code == 202
    assert response.json()["status"] == RunStatus.QUEUED.value
    assert [e["run_id"] for e in client.enqueued] == [response.json()["run_id"]]


async def test_user_b_gets_404_not_403_on_user_a_resources(client, db):
    headers_a = await register(client, "a@example.com")
    headers_b = await register(client, "b@example.com")

    session_id = (
        await client.post("/sessions", json={"name": "A's session"}, headers=headers_a)
    ).json()["id"]
    run_id = (
        await client.post(
            f"/sessions/{session_id}/run", json={"message": "hi"}, headers=headers_a
        )
    ).json()["run_id"]

    # 404 rather than 403 everywhere: a 403 would confirm the ID exists.
    for method, url in [
        ("get", f"/sessions/{session_id}"),
        ("delete", f"/sessions/{session_id}"),
        ("get", f"/runs/{run_id}/status"),
        ("get", f"/runs/{run_id}/steps"),
        ("delete", f"/runs/{run_id}"),
    ]:
        response = await getattr(client, method)(url, headers=headers_b)
        assert response.status_code == 404, f"{method} {url} returned {response.status_code}"

    # user B's own listing is empty, and A's resources are still intact
    assert (await client.get("/sessions", headers=headers_b)).json() == []
    assert (await client.get(f"/sessions/{session_id}", headers=headers_a)).status_code == 200


async def test_status_reports_step_count_without_loading_steps(client, db):
    headers = await register(client, "status@example.com")
    session_id = (
        await client.post("/sessions", json={"name": "S"}, headers=headers)
    ).json()["id"]
    run_id = (
        await client.post(f"/sessions/{session_id}/run", json={"message": "hi"}, headers=headers)
    ).json()["run_id"]

    response = await client.get(f"/runs/{run_id}/status", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["step_count"] == 0
    assert body["tokens_used"] == 0


async def test_cancelling_a_terminal_run_returns_409(client, db):
    headers = await register(client, "cancel@example.com")
    session_id = (
        await client.post("/sessions", json={"name": "S"}, headers=headers)
    ).json()["id"]
    run_id = (
        await client.post(f"/sessions/{session_id}/run", json={"message": "hi"}, headers=headers)
    ).json()["run_id"]

    assert (await client.delete(f"/runs/{run_id}", headers=headers)).status_code == 204
    assert (await client.delete(f"/runs/{run_id}", headers=headers)).status_code == 409

    run = await db.get(AgentRun, run_id)
    await db.refresh(run)
    assert run.status is RunStatus.CANCELLED


# ------------------------------ documentation ------------------------------- #
async def test_scalar_reference_is_served(client):
    response = await client.get("/scalar")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "/openapi.json" in body          # it points at this app's schema
    assert "AgentCore" in body


async def test_scalar_config_is_what_we_asked_for(client):
    """scalar-fastapi enables telemetry by default; a deliverable should not
    report usage from someone else's machine."""
    body = (await client.get("/scalar")).text

    assert '"telemetry": false' in body
    assert '"persistAuth": true' in body          # token survives a page reload
    assert '"url": "/openapi.json"' in body


async def test_scalar_is_not_itself_an_endpoint_in_the_schema(client):
    schema = (await client.get("/openapi.json")).json()
    assert "/scalar" not in schema["paths"]


async def test_every_tag_used_by_a_route_is_documented(client):
    """A tag with no description is a bare heading in the sidebar."""
    schema = (await client.get("/openapi.json")).json()

    documented = {tag["name"]: tag.get("description", "") for tag in schema["tags"]}
    used = {
        tag
        for path in schema["paths"].values()
        for operation in path.values()
        for tag in operation.get("tags", [])
    }
    assert used <= set(documented), used - set(documented)
    assert all(documented[tag].strip() for tag in used), documented


async def test_the_docs_landing_explains_how_to_authenticate(client):
    schema = (await client.get("/openapi.json")).json()
    description = schema["info"]["description"]
    assert "/auth/register" in description
    assert "404, not 403" in description


async def test_scalar_is_the_only_reference_ui(client):
    """Swagger and ReDoc are switched off in the app factory (`docs_url=None`,
    `redoc_url=None`); Scalar renders the same schema. The schema itself stays
    served, because that is what generated clients and this suite read."""
    assert (await client.get("/scalar")).status_code == 200
    assert (await client.get("/openapi.json")).status_code == 200

    assert (await client.get("/docs")).status_code == 404
    assert (await client.get("/redoc")).status_code == 404


# ------------------------- rate limiting (bonus) ---------------------------- #
async def test_the_eleventh_concurrent_run_is_rejected_with_429(client, db, redis):
    from app.core.config import settings

    headers = await register(client, "burst@example.com")
    session_id = (
        await client.post("/sessions", json={"name": "S"}, headers=headers)
    ).json()["id"]

    for i in range(settings.MAX_CONCURRENT_RUNS_PER_USER):
        response = await client.post(
            f"/sessions/{session_id}/run", json={"message": f"run {i}"}, headers=headers
        )
        assert response.status_code == 202, f"run {i + 1} should have been accepted"

    response = await client.post(
        f"/sessions/{session_id}/run", json={"message": "one too many"}, headers=headers
    )

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "5"
    detail = response.json()["error"]["detail"]
    assert "10 active runs" in detail
    # the rejected submission left no row behind
    assert len(client.enqueued) == settings.MAX_CONCURRENT_RUNS_PER_USER


async def test_the_limit_is_per_user_not_global(client, db, redis):
    from app.core.config import settings

    headers_a = await register(client, "quota-a@example.com")
    headers_b = await register(client, "quota-b@example.com")
    session_a = (await client.post("/sessions", json={"name": "A"}, headers=headers_a)).json()["id"]
    session_b = (await client.post("/sessions", json={"name": "B"}, headers=headers_b)).json()["id"]

    for _ in range(settings.MAX_CONCURRENT_RUNS_PER_USER):
        await client.post(f"/sessions/{session_a}/run", json={"message": "m"}, headers=headers_a)

    assert (
        await client.post(f"/sessions/{session_a}/run", json={"message": "m"}, headers=headers_a)
    ).status_code == 429
    # user B is untouched by user A exhausting their quota
    assert (
        await client.post(f"/sessions/{session_b}/run", json={"message": "m"}, headers=headers_b)
    ).status_code == 202


async def test_cancelling_a_queued_run_frees_a_slot(client, db, redis):
    from app.core.config import settings

    headers = await register(client, "free-slot@example.com")
    session_id = (await client.post("/sessions", json={"name": "S"}, headers=headers)).json()["id"]

    run_ids = []
    for _ in range(settings.MAX_CONCURRENT_RUNS_PER_USER):
        run_ids.append(
            (await client.post(f"/sessions/{session_id}/run", json={"message": "m"},
                               headers=headers)).json()["run_id"]
        )
    assert (
        await client.post(f"/sessions/{session_id}/run", json={"message": "m"}, headers=headers)
    ).status_code == 429

    assert (await client.delete(f"/runs/{run_ids[0]}", headers=headers)).status_code == 204

    assert (
        await client.post(f"/sessions/{session_id}/run", json={"message": "m"}, headers=headers)
    ).status_code == 202


# ---------------------- OpenAPI enrichment (bonus) -------------------------- #
async def test_every_operation_has_an_explicit_summary(client):
    spec = (await client.get("/openapi.json")).json()
    missing = [
        f"{m.upper()} {p}"
        for p, ops in spec["paths"].items()
        for m, op in ops.items()
        if not op.get("summary")
    ]
    assert missing == []


async def test_protected_routes_document_401_and_404(client):
    spec = (await client.get("/openapi.json")).json()

    for path, method in [
        ("/sessions/{session_id}", "get"),
        ("/runs/{run_id}/status", "get"),
        ("/runs/{run_id}/steps", "get"),
        ("/runs/{run_id}/stream", "get"),
        ("/memory/{memory_id}", "delete"),
    ]:
        responses = spec["paths"][path][method]["responses"]
        assert "401" in responses, f"{method} {path}"
        assert "404" in responses, f"{method} {path}"


async def test_the_documented_failures_match_what_the_routes_actually_return(client):
    spec = (await client.get("/openapi.json")).json()

    run_post = spec["paths"]["/sessions/{session_id}/run"]["post"]["responses"]
    assert set(run_post) >= {"202", "401", "404", "422", "429"}

    run_delete = spec["paths"]["/runs/{run_id}"]["delete"]["responses"]
    assert set(run_delete) >= {"204", "401", "404", "409"}

    token = spec["paths"]["/auth/token"]["post"]["responses"]
    assert "401" in token


async def test_error_responses_carry_a_schema_not_an_untyped_blob(client):
    spec = (await client.get("/openapi.json")).json()
    not_found = spec["paths"]["/runs/{run_id}/status"]["get"]["responses"]["404"]

    assert "ErrorResponse" in str(not_found["content"]["application/json"]["schema"])
    assert not_found["description"]


async def test_the_request_id_is_handed_to_the_worker(client):
    """The API knows the request id; the worker cannot discover it. Without
    passing it explicitly, a run's worker records are unlinkable to the request
    that submitted it."""
    headers = await register(client, "trace@example.com")
    session_id = (
        await client.post("/sessions", json={"name": "S"}, headers=headers)
    ).json()["id"]

    response = await client.post(
        f"/sessions/{session_id}/run",
        json={"message": "hi"},
        headers={**headers, "X-Request-ID": "trace-me-123"},
    )

    assert response.headers["X-Request-ID"] == "trace-me-123"
    assert client.enqueued[-1]["request_id"] == "trace-me-123"


async def test_a_run_that_cannot_be_queued_does_not_hold_a_slot(client, db, redis, monkeypatch):
    """If the row commits but the enqueue fails, the run exists as QUEUED with no
    task behind it. Left alone it counts as active forever, permanently costing
    the user one of their ten slots."""
    from app.core.limits import count_active_runs

    headers = await register(client, "orphan@example.com")
    session_id = (await client.post("/sessions", json={"name": "S"}, headers=headers)).json()["id"]

    def explode(run_id, **kwargs):
        raise RuntimeError("broker unreachable")

    monkeypatch.setattr(
        "app.routers.runs.execute_agent_run", type("Stub", (), {"delay": staticmethod(explode)})
    )

    with pytest.raises(RuntimeError):
        await client.post(f"/sessions/{session_id}/run", json={"message": "hi"}, headers=headers)

    from sqlalchemy import select

    from app.models.agent import AgentRun, RunStatus
    from app.models.user import User

    user = (
        await db.execute(select(User).where(User.email == "orphan@example.com"))
    ).scalar_one()
    run = (await db.execute(select(AgentRun))).scalars().all()[-1]

    assert run.status is RunStatus.FAILED          # not left QUEUED forever
    assert await count_active_runs(db, user.id) == 0
    assert int(await redis.get(f"user:{user.id}:active_runs")) == 0


async def test_an_abandoned_run_cannot_release_the_slot_a_second_time(
    client, db, redis, monkeypatch
):
    """A DELETE on the orphaned run must not decrement again: it is terminal, so
    the route answers 409 rather than releasing."""
    headers = await register(client, "orphan2@example.com")
    session_id = (await client.post("/sessions", json={"name": "S"}, headers=headers)).json()["id"]

    monkeypatch.setattr(
        "app.routers.runs.execute_agent_run",
        type("Stub", (), {"delay": staticmethod(
            lambda run_id, **kw: (_ for _ in ()).throw(RuntimeError("broker down")))}),
    )
    with pytest.raises(RuntimeError):
        await client.post(f"/sessions/{session_id}/run", json={"message": "hi"}, headers=headers)

    from sqlalchemy import select

    from app.models.agent import AgentRun
    from app.models.user import User

    user = (
        await db.execute(select(User).where(User.email == "orphan2@example.com"))
    ).scalar_one()
    run = (await db.execute(select(AgentRun))).scalars().all()[-1]

    response = await client.delete(f"/runs/{run.id}", headers=headers)

    assert response.status_code == 409
    assert int(await redis.get(f"user:{user.id}:active_runs")) == 0     # still zero, not -1
