# Quick Start

- Install [Git LFS](https://github.com/git-lfs/git-lfs)
- Install Python 3.10 (Consider using [pyenv](https://github.com/pyenv/pyenv) for that).
- Install [Poetry](https://python-poetry.org).
- Create virtual environment and install dependencies:

  ```shell
  poetry install
  ```

- Activate virtual environment:

  ```shell
  poetry shell
  ```

- Install [pre-commit](https://pre-commit.com/) hooks:

  ```shell
  pre-commit install
  ```

- (Optional) Logging to the HuggingFace Hub in order to download some of the required datasets and models:

  ```shell
  huggingface-cli login
  ```

# Release Process

We use [bump-my-version](https://github.com/callowayproject/bump-my-version) to bump the package's version number and to create a commit and tag for that change.

Use `bump-my-version show-bump` to visualize potential versioning paths from the current version:

```shell
$ bump-my-version show-bump
0.1.0-dev0 ── bump ─┬─ major ─ 1.0.0-dev0
                    ├─ minor ─ 0.2.0-dev0
                    ├─ patch ─ 0.1.1-dev0
                    ├─ pre_l ─ 0.1.0
                    ╰─ pre_n ─ 0.1.0-dev1
```

Use the `--dry-run` flag at first with each command to confirm that the changes will be correct. 

## Steps

1. If this is not a patch release, then bump the minor or the major version number first:

   ```shell
   bump-my-version bump --verbose <VERSION PART>
   ```

   Where `<VERSION PART>` is either `minor` or `major` depending on the release type.

   Use the `--dry-run` flag at first to confirm that the changes will be correct. 
 
2. Switch from development to release version, commit the change and create a tag:

   ```shell
   bump-my-version bump --verbose --commit --tag --sign-tags pre_l
   ```

   This will:
   
   - Remove the `-devXX` suffix from the current version number, e.g. `0.1.0-dev0` -> `0.1.0`
   - Create a commit for the change with a default message
   - Create a tag for this commit
   
   Use the `--dry-run` flag at first to confirm that the changes will be correct. 

3. Create Github release with description extracted from the [CHANGELOG.md](CHANGELOG.md) from and with the License file attached.

4. Make sure CI was triggered and package was pushed to PyPI.

5. Switch from release to development version:

   ```shell
   bump-my-version bump --commit --verbose patch
   ```

   This will bump the patch part of the version and add the `-dev0` suffix, e.g. `0.1.0` -> `0.1.1-dev0`
   
   Use the `--dry-run` flag at first to confirm that the changes will be correct. 

6. Add new `## Unreleased` section to [CHANGELOG.md](CHANGELOG.md) to prepare for the next release.
