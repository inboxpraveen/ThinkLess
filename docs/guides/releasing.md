# Releasing to PyPI

Releases go to PyPI through
[trusted publishing](https://docs.pypi.org/trusted-publishers/): GitHub
Actions proves to PyPI which repository and workflow it is running, and PyPI
issues a short-lived upload token. No API token is stored anywhere.

The workflow is `.github/workflows/release.yml`. Pushing a tag such as
`v0.2.0` starts it. It then:

1. checks that the tag matches `src/thinkless/_version.py`, and stops if not,
2. builds the wheel and the source distribution,
3. runs `twine check --strict` on both,
4. installs the wheel into a clean virtual environment and runs
   `thinkless --version`, `thinkless doctor` and `examples/01_rules_only.py`,
5. uploads to PyPI from a job bound to the `pypi` environment.

The CI workflow runs the same wheel install and smoke test on Linux, macOS and
Windows for every push, so packaging problems show up before a tag is pushed.

## One-time setup

Do this once, before the first release.

### 1. Register a pending publisher on PyPI

A pending publisher lets the workflow create the `thinkless` project on its
first upload.

1. Sign in at [pypi.org](https://pypi.org).
2. Open [Your account, Publishing](https://pypi.org/manage/account/publishing/).
3. Under **Add a new pending publisher**, pick the **GitHub** tab.
4. Fill in the form:

    | Field | Value |
    |---|---|
    | PyPI Project Name | `thinkless` |
    | Owner | `inboxpraveen` |
    | Repository name | `ThinkLess` |
    | Workflow name | `release.yml` |
    | Environment name | `pypi` |

5. Click **Add**.

The workflow name is the file name only, not the path. Every value has to
match exactly. A mismatch makes the upload fail with `invalid-publisher`.

A pending publisher does not reserve the name. Until the first upload
succeeds, anyone can still register `thinkless`, so publish soon after this
step.

### 2. Create the `pypi` environment on GitHub

1. Open the repository on GitHub, then **Settings**, then **Environments**.
2. Click **New environment**, enter `pypi` as the name, and click
   **Configure environment**.
3. Recommended: under **Deployment protection rules**, tick
   **Required reviewers**, add `inboxpraveen`, and click
   **Save protection rules**. Every upload then waits for your approval.
4. Recommended: under **Deployment branches and tags**, choose
   **Selected branches and tags**, click **Add deployment branch or tag rule**,
   set **Ref type** to **Tag**, enter `v*` as the name pattern, and click
   **Add rule**. Only version tags can then publish.

Nothing else is needed. The workflow requests `id-token: write` itself, and no
secrets are used.

## Cutting a release

### 1. Prepare the version

1. Set the version in `src/thinkless/_version.py`, for example
   `__version__ = "0.2.0"`.
2. In `CHANGELOG.md`, move the entries under **Unreleased** into a new
   `## 0.2.0 - YYYY-MM-DD` section with the release date.
3. Set `version` and `date-released` in `CITATION.cff`.
4. Run the checks locally:

    ```bash
    ruff check src tests scripts benchmarks examples
    ruff format --check src tests scripts benchmarks examples
    mypy
    pytest
    python scripts/check_prose.py
    mkdocs build --strict
    ```

5. Commit and push to `main`, and wait for CI to pass.

### 2. Tag and push

```bash
git tag -a v0.2.0 -m "ThinkLess 0.2.0"
git push origin v0.2.0
```

### 3. Approve the upload

Open the **Actions** tab and select the **Release** run. When **Build and
check** finishes, **Publish to PyPI** waits for approval if you set required
reviewers. Click **Review deployments**, tick **pypi**, and click
**Approve and deploy**.

### 4. Check the result

The project appears at
[pypi.org/project/thinkless](https://pypi.org/project/thinkless/). Install it
into a fresh environment to confirm:

```bash
python -m venv /tmp/tl-check
/tmp/tl-check/bin/pip install thinkless==0.2.0
/tmp/tl-check/bin/thinkless --version
/tmp/tl-check/bin/thinkless doctor
```

On Windows, the executables live in `Scripts` instead of `bin`:

```powershell
python -m venv $env:TEMP\tl-check
& $env:TEMP\tl-check\Scripts\pip install thinkless==0.2.0
& $env:TEMP\tl-check\Scripts\thinkless --version
```

A new version can take a minute or two to reach every PyPI mirror.

### 5. Publish the GitHub release

1. Open **Releases**, then **Draft a new release**.
2. Choose the tag `v0.2.0`.
3. Title: `ThinkLess 0.2.0`.
4. Paste that version's section from `CHANGELOG.md` as the description.
5. Click **Publish release**.

After the first upload, PyPI turns the pending publisher into a normal
trusted publisher for the project. It shows under the project's
**Manage**, then **Publishing**. Later releases only need the steps in this
section.

## When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `Tag v0.2.1 does not match the package version 0.2.0` | the tag and `_version.py` differ | delete the tag (below), fix the version, tag again |
| `invalid-publisher: valid token, but no corresponding publisher` | a field on PyPI does not match the repository, workflow file or environment | correct the publisher on PyPI, then **Re-run failed jobs** |
| `File already exists` | that version was uploaded before | PyPI never accepts the same version twice, even after deleting it; bump to the next patch version |
| The publish job never starts | it is waiting for approval, or the tag rule does not match | approve it under **Review deployments**, or check the environment's tag rule |
| `twine check` fails | the README does not render on PyPI | fix the Markdown; relative links and images must be absolute URLs |

To delete a tag that has not published anything:

```bash
git tag -d v0.2.1
git push origin :refs/tags/v0.2.1
```

A bad release that is already on PyPI can be yanked from the project's
**Manage** page. Yanked versions stay downloadable when pinned exactly, but
`pip install thinkless` skips them. Fix the problem in a new version.

## Versioning

ThinkLess follows [semantic versioning](https://semver.org/). Before 1.0, a
minor version (0.3.0) can change the public API and a patch version (0.2.1)
only fixes bugs. The public API is what `thinkless/__init__.py` exports, the
CLI, and the trace format.
