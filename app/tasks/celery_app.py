from celery import Celery
from celery.signals import setup_logging, worker_ready

from app.core.config import settings
from app.core.logging import configure_logging


@worker_ready.connect
def _report_event_loop(**_kwargs):
    """State the loop out loud. A silent fallback to the selector loop is the
    kind of regression that shows up as 'the agent got slower' months later."""
    from app.core.logging import get_logger
    from app.tasks.agent_tasks import uvloop

    get_logger(__name__).info(
        "worker_event_loop", loop="uvloop" if uvloop is not None else "asyncio"
    )


@setup_logging.connect
def _use_structlog(**_kwargs):
    """Celery replaces the root logger's handlers on worker start, which would
    strip the JSON renderer and print our event dicts as Python reprs. Connecting
    to this signal tells Celery logging is already handled and to leave it alone.
    """
    configure_logging()


celery_app = Celery(
    "agentcore",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks.agent_tasks"],
)

# The hard kill trails the graceful stop, and the broker must not redeliver a
# message while its task can still legitimately be running — so both are derived
# from one number rather than three that drift apart.
SOFT_TIME_LIMIT = settings.AGENT_RUN_TIME_LIMIT_SECONDS
HARD_TIME_LIMIT = SOFT_TIME_LIMIT + 60
VISIBILITY_TIMEOUT = HARD_TIME_LIMIT + 300

celery_app.conf.update(
    # ---- serialisation ------------------------------------------------- #
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # ---- logging ------------------------------------------------------- #
    worker_hijack_root_logger=False,
    # ---- delivery ------------------------------------------------------ #
    # Ack when the task finishes, not when it is received. A worker killed
    # mid-run would otherwise take the message with it, and the run would sit at
    # RUNNING forever holding one of the user's concurrency slots — nothing else
    # in the system ever revisits it. The cost is at-least-once delivery, which
    # `_claim_and_run` already handles with a SELECT ... FOR UPDATE claim.
    task_acks_late=True,
    # With acks_late, a child killed outright (OOM, SIGKILL) leaves the message
    # unacked; this requeues it instead of silently marking the task failed.
    task_reject_on_worker_lost=True,
    # Celery's default is 4, so a --concurrency=2 worker reserves 8 messages.
    # These tasks run for minutes, so prefetched messages queue behind the one in
    # progress while another worker sits idle. 1 means a worker holds exactly what
    # it is working on and the queue stays the shared thing it is meant to be.
    worker_prefetch_multiplier=settings.WORKER_PREFETCH_MULTIPLIER,
    # ---- limits -------------------------------------------------------- #
    # Soft first: raises SoftTimeLimitExceeded *inside* the task so the run is
    # finished honestly and the SSE client gets its `done` frame. The hard limit
    # is the backstop for a task that ignores the soft one.
    task_soft_time_limit=SOFT_TIME_LIMIT,
    task_time_limit=HARD_TIME_LIMIT,
    # Recycle children periodically: workers here are long-lived and hold async
    # engines, so a slow leak would otherwise accumulate for the process's life.
    worker_max_tasks_per_child=settings.WORKER_MAX_TASKS_PER_CHILD,
    # ---- broker -------------------------------------------------------- #
    broker_transport_options={
        # Redis has no real ack: it redelivers anything unacked after this long.
        # Below the hard time limit it would hand a still-running task to a second
        # worker, and with acks_late that means genuine double execution.
        "visibility_timeout": VISIBILITY_TIMEOUT,
    },
    # Celery 6 defaults this off; without it a worker that loses Redis at startup
    # exits instead of waiting for it.
    broker_connection_retry_on_startup=True,
    # ---- results ------------------------------------------------------- #
    task_track_started=True,
    result_expires=3600,
)
