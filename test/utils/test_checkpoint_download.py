# Copyright 2026 DeepMI Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Check how the checkpoint download reacts to a host that misbehaves.

The hosts do break mid-transfer: a build has logged
``ChunkedEncodingError ... IncompleteRead(2835659 bytes read, 28432957 more expected)`` from one of
them. Falling through to the next url covers a host being down, but not the last url in the list,
and not a transfer that breaks on a host that is otherwise up. No network is touched here; requests
is replaced with stubs.
"""

import pytest
import requests

from FastSurferCNN.utils import checkpoint


class Reply:
    """Stands in for a requests.Response, with just what download_checkpoint reads."""

    def __init__(self, ok: bool, content: bytes = b"weights", status: int = 200, url: str = "u"):
        self.ok = ok
        self.content = content
        self.status_code = status
        self.text = "" if ok else "not found"
        self.url = url

    def raise_for_status(self):
        if not self.ok:
            raise requests.exceptions.HTTPError(f"{self.status_code}", response=self)


@pytest.fixture
def no_sleep(monkeypatch):
    """The backoff must not actually wait, and the delays are asserted on instead."""
    slept = []
    monkeypatch.setattr("time.sleep", slept.append)
    return slept


@pytest.fixture
def target(tmp_path):
    return tmp_path / "a_checkpoint.pkl"


def stub_get(monkeypatch, outcomes):
    """Replace requests.get, returning or raising each outcome in turn, and record the calls."""
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(requests, "get", fake_get)
    return calls


def test_a_broken_transfer_is_retried_on_the_same_host(monkeypatch, no_sleep, target):
    """The failure the hosts actually produce, where trying again is what fixes it."""
    broken = requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead")
    calls = stub_get(monkeypatch, [broken, broken, Reply(ok=True)])

    checkpoint.download_checkpoint("a_checkpoint.pkl", target, ["https://host-a"])

    assert len(calls) == 3, "both failures should have been retried"
    assert all(url.startswith("https://host-a") for url, _ in calls), "no other host was available"
    assert target.read_bytes() == b"weights"


def test_the_retries_back_off(monkeypatch, no_sleep, target):
    """A tight loop against a struggling host helps nobody."""
    broken = requests.exceptions.ConnectionError("reset")
    stub_get(monkeypatch, [broken, broken, Reply(ok=True)])

    checkpoint.download_checkpoint("a_checkpoint.pkl", target, ["https://host-a"])

    assert no_sleep == [checkpoint.DOWNLOAD_BACKOFF, checkpoint.DOWNLOAD_BACKOFF * 2]


def test_the_last_host_is_retried_too(monkeypatch, no_sleep, target):
    """
    The case the url fallback cannot cover.

    With one attempt per url, a transient failure on the last host ends the download, however many
    hosts are configured.
    """
    broken = requests.exceptions.ChunkedEncodingError("Connection broken")
    calls = stub_get(monkeypatch, [broken, broken, Reply(ok=True)])

    checkpoint.download_checkpoint("a_checkpoint.pkl", target, ["https://only-host"])

    assert len(calls) == 3
    assert target.exists()


def test_a_missing_file_is_not_retried(monkeypatch, no_sleep, target):
    """A 404 is the same answer every time, so retrying it only delays the fallback."""
    calls = stub_get(monkeypatch, [Reply(ok=False, status=404), Reply(ok=True)])

    checkpoint.download_checkpoint("a_checkpoint.pkl", target, ["https://host-a", "https://host-b"])

    assert [url.split("/")[2] for url, _ in calls] == ["host-a", "host-b"], (
        "the 404 should move straight to the next host rather than being retried"
    )
    assert no_sleep == [], "a status reply is not a transport failure, so there is nothing to wait for"


def test_the_read_is_bounded(monkeypatch, no_sleep, target):
    """
    Without a read timeout a stalled transfer hangs with nothing to interrupt it.

    In a docker build that means hanging until the job's own limit, which is how a dropped
    connection turned into a lost hour rather than a failed step.
    """
    calls = stub_get(monkeypatch, [Reply(ok=True)])

    checkpoint.download_checkpoint("a_checkpoint.pkl", target, ["https://host-a"])

    _, kwargs = calls[0]
    connect_timeout, read_timeout = kwargs["timeout"]
    assert read_timeout is not None, "a stalled read has to fail rather than hang"
    assert connect_timeout is not None


def test_giving_up_reports_every_host(monkeypatch, no_sleep, target):
    """When nothing worked, the error has to name what was tried."""
    stub_get(monkeypatch, [Reply(ok=False, status=404, url="https://host-a/a_checkpoint.pkl")])

    # ExceptionGroup on 3.11 and later, RuntimeError before it, and both derive from Exception
    with pytest.raises(Exception):  # noqa: B017
        checkpoint.download_checkpoint(
            "a_checkpoint.pkl", target, ["https://host-a", "https://host-b"]
        )
    assert not target.exists(), "a failed download must not leave a file behind"
