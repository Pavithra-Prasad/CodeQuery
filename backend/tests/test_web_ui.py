"""Selenium smoke tests for the web page.

The real FastAPI app and frontend run on a local port; only the indexing, the vector store
and the LLM behind the API are replaced by a fake backend, so the browser sees real
HTTP responses with controlled data.

Run only these:   pytest -m e2e
Skip them:        pytest -m "not e2e"
"""

import socket
import threading
import time

import pytest
import uvicorn
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import main

pytestmark = pytest.mark.e2e


class FakeBackend:
    def __init__(self):
        self.reset()

    def reset(self):
        self.repos = {}                 # repo name -> number of chunks
        self.answer = "It hashes the password."
        self.sources = []
        self.query_error = None
        self.questions = []

    def list_collections(self):
        return list(self.repos)

    def get_collection_count(self, name):
        return self.repos.get(name, 0)

    def delete_collection(self, name):
        self.repos.pop(name, None)

    def index_codebase(self, source, repo_name, progress_callback):
        self.repos[repo_name] = 12
        return {"status": "success", "repo_name": repo_name, "num_files": 4, "num_chunks": 12,
                "message": "Successfully indexed 4 files into 12 chunks."}

    async def ask_question(self, question, repo_name, top_k):
        self.questions.append((question, repo_name))
        if self.query_error:
            raise RuntimeError(self.query_error)
        return {"answer": self.answer, "sources": self.sources}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def backend():
    fake = FakeBackend()
    with pytest.MonkeyPatch.context() as mp:
        for name in ("list_collections", "get_collection_count", "delete_collection", "index_codebase", "ask_question"):
            mp.setattr(main, name, getattr(fake, name))
        yield fake


@pytest.fixture(scope="module")
def base_url(backend):
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            pytest.fail("test server did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture(scope="module")
def browser():
    options = webdriver.ChromeOptions()
    for arg in ("--headless=new", "--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage", "--window-size=1280,900"):
        options.add_argument(arg)
    try:
        driver = webdriver.Chrome(options=options)
    except WebDriverException as e:
        pytest.skip(f"Chrome is not available: {e.msg}")
    yield driver
    driver.quit()


@pytest.fixture
def page(browser, base_url, backend):
    """Opens the page fresh for each test, after the test has set up the fake backend."""
    backend.reset()

    def open_page():
        browser.get(base_url)
        WebDriverWait(browser, 10).until(lambda d: d.find_element(By.ID, "reposList").text.strip())
        return browser

    return open_page


def wait_for(driver, css, text=""):
    WebDriverWait(driver, 10).until(EC.text_to_be_present_in_element((By.CSS_SELECTOR, css), text))
    return driver.find_element(By.CSS_SELECTOR, css)


def select_repo(driver, name):
    driver.find_element(By.CSS_SELECTOR, f'.repo-item[data-repo="{name}"]').click()
    wait_for(driver, "#activeRepo", name)


def ask(driver, question):
    driver.find_element(By.ID, "questionInput").send_keys(question)
    driver.find_element(By.ID, "sendBtn").click()


def test_empty_state_when_nothing_is_indexed(page):
    driver = page()
    assert "No repositories indexed yet" in driver.find_element(By.ID, "reposList").text
    assert driver.title.startswith("CodeQuery AI")


def test_indexed_repos_are_listed_and_selectable(page, backend):
    backend.repos = {"demo-repo": 42}
    driver = page()
    item = wait_for(driver, ".repo-item", "demo-repo")
    assert "42 chunks" in item.text
    select_repo(driver, "demo-repo")
    assert "active" in item.get_attribute("class")


def test_asking_a_question_shows_the_answer_and_sources(page, backend):
    backend.repos = {"demo-repo": 42}
    backend.answer = "Login **hashes** the password in `auth.py`."
    backend.sources = [{"file_path": "auth.py", "start_line": 1, "end_line": 9, "relevance": 0.91}]
    driver = page()
    select_repo(driver, "demo-repo")

    ask(driver, "How does login work?")

    bubble = wait_for(driver, ".message.assistant .message-bubble", "hashes")
    assert bubble.find_element(By.TAG_NAME, "strong").text == "hashes"
    assert bubble.find_element(By.TAG_NAME, "code").text == "auth.py"
    assert "How does login work?" in driver.find_element(By.CSS_SELECTOR, ".message.user").text
    assert "1 source referenced" in driver.find_element(By.CSS_SELECTOR, ".sources-toggle").text
    assert driver.find_element(By.CSS_SELECTOR, ".source-file").get_attribute("textContent") == "auth.py"
    assert backend.questions == [("How does login work?", "demo-repo")]


def test_server_errors_show_an_error_message(page, backend):
    backend.repos = {"demo-repo": 42}
    backend.query_error = "LLM timeout"
    driver = page()
    select_repo(driver, "demo-repo")

    ask(driver, "Why is it slow?")

    assert "Error generating answer: LLM timeout" in wait_for(driver, ".error-bubble", "LLM timeout").text


def test_asking_without_selecting_a_repo_is_blocked(page, backend):
    backend.repos = {"demo-repo": 42}
    driver = page()

    ask(driver, "How does login work?")

    assert "Please select a repository first" in wait_for(driver, ".toast.error", "select a repository").text
    assert backend.questions == []


def test_indexing_a_repo_adds_and_selects_it(page, backend):
    driver = page()
    driver.find_element(By.ID, "sourceInput").send_keys("https://github.com/someone/shop")
    driver.find_element(By.ID, "indexBtn").click()

    wait_for(driver, "#progressText", "Successfully indexed 4 files into 12 chunks.")
    wait_for(driver, '.repo-item[data-repo="shop"]', "12 chunks")
    assert "shop" in driver.find_element(By.ID, "activeRepo").text
    assert backend.repos == {"shop": 12}


def test_deleting_a_repo_after_confirming(page, backend):
    backend.repos = {"old-repo": 5}
    driver = page()
    wait_for(driver, ".repo-item", "old-repo")

    driver.find_element(By.CSS_SELECTOR, ".repo-delete").click()
    WebDriverWait(driver, 5).until(EC.alert_is_present()).accept()

    wait_for(driver, "#reposList", "No repositories indexed yet")
    assert backend.repos == {}
