"""Send the refined classifier to the GitHub repository as one commit on main.

The commit holds the refined model card, the judged particles behind it, and a
version bump. Pushing to main starts the Windows build, which publishes the new
installer as Release v<version> a few minutes later.

Needs a GitHub personal access token with write access to the repository's
contents (fine-grained token: Repository access = Particle_Classifier,
Permissions > Contents = Read and write).
"""
import base64
import datetime as dt
import json
import re
import urllib.error
import urllib.request

OWNER, REPO, BRANCH = "akhalil1996", "Particle_Classifier", "main"
AUTHOR = {"name": "akhalil1996", "email": "200668417+akhalil1996@users.noreply.github.com"}
API = f"https://api.github.com/repos/{OWNER}/{REPO}"


class PublishError(RuntimeError):
    pass


def _call(token, method, url, body=None):
    req = urllib.request.Request(url if url.startswith("http") else API + url, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                          "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "CauliQuant"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:300]
        if e.code == 401:
            raise PublishError("GitHub did not accept the token (401). Check that it is pasted in full and not expired.")
        if e.code in (403, 404):
            raise PublishError(f"GitHub refused the request ({e.code}). The token needs write access to the contents of "
                               f"{OWNER}/{REPO}.\n{msg}")
        raise PublishError(f"GitHub error {e.code}: {msg}")
    except urllib.error.URLError as e:
        raise PublishError(f"Could not reach GitHub: {e.reason}")


def check_token(token):
    """Returns the repository's permissions for this token, or raises PublishError."""
    info = _call(token, "GET", "")
    if not info.get("permissions", {}).get("push"):
        raise PublishError("This token can read the repository but cannot write to it.")
    return info


def next_version(v):
    a, b, c = (int(x) for x in v.split("."))
    return f"{a}.{b}.{c + 1}"


def publish(token, files, message, bump=True):
    """Commit files ({repository path: bytes}) to main in one commit. Returns dict(version, commit_url, actions_url)."""
    ref = _call(token, "GET", f"/git/ref/heads/{BRANCH}")
    head = ref["object"]["sha"]
    base_tree = _call(token, "GET", f"/git/commits/{head}")["tree"]["sha"]
    files = dict(files)
    version = None
    if bump:
        init = _call(token, "GET", f"/contents/cauliquant/__init__.py?ref={head}")
        text = base64.b64decode(init["content"]).decode("utf-8")
        m = re.search(r'__version__\s*=\s*"(\d+\.\d+\.\d+)"', text)
        if not m:
            raise PublishError("Could not find the version number in cauliquant/__init__.py.")
        version = next_version(m.group(1))
        files["cauliquant/__init__.py"] = text.replace(m.group(0), f'__version__ = "{version}"').encode("utf-8")
    tree = []
    for path, data in files.items():
        blob = _call(token, "POST", "/git/blobs", {"content": base64.b64encode(data).decode(), "encoding": "base64"})
        tree.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
    new_tree = _call(token, "POST", "/git/trees", {"base_tree": base_tree, "tree": tree})["sha"]
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    who = dict(AUTHOR, date=now)
    commit = _call(token, "POST", "/git/commits", {"message": message, "tree": new_tree, "parents": [head],
                                                  "author": who, "committer": who})
    _call(token, "PATCH", f"/git/refs/heads/{BRANCH}", {"sha": commit["sha"], "force": False})
    return dict(version=version, commit=commit["sha"], commit_url=commit.get("html_url", ""),
                actions_url=f"https://github.com/{OWNER}/{REPO}/actions",
                release_url=f"https://github.com/{OWNER}/{REPO}/releases/tag/v{version}" if version else "")
