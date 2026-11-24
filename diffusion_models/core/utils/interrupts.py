import contextlib
import signal


@contextlib.contextmanager
def sigterm_as_interrupt():
  """Treat SIGTERM like Ctrl-C so an interrupted checkpoint gets saved."""

  def handler(signum, frame):
    raise KeyboardInterrupt

  try:
    previous = signal.signal(signal.SIGTERM, handler)
  except ValueError:  # Not in the main thread.
    yield
    return
  try:
    yield
  finally:
    signal.signal(signal.SIGTERM, previous)
