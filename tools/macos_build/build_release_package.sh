#!/bin/bash

# abort on the first failure: nearly every step here feeds the next one, so an unchecked error
# (a partial cp of fs-pruned, a py2app build that did not produce the .app, a sed that wrote an
# empty postinstall) otherwise runs on into pkgbuild and yields an installer that looks fine but
# is incomplete. pipefail additionally covers the "git ls-files | rsync" pipeline below, where a
# failing git would just hand rsync an empty file list and "succeed".
set -e
set -o pipefail

if [[ "$#" -lt 1 ]] || { [[ "$1" != "arm" ]] && [[ "$1" != "intel" ]] ; } ; then
  echo
  echo "Usage:  build_release_package.sh <arm|intel> [--fs-download-cache path] [--fs-pruned-cache-dir dir]"
  echo "                                             [--py2app-venv dir] [--uv-cache-dir dir]"
  echo "                                             [--checkpoints-dir dir]"
  echo
  echo "--fs-download-cache points at a file path for the raw FreeSurfer tarball: if it already"
  echo "  exists there (e.g. from a prior, interrupted local run), it is reused instead of"
  echo "  downloading again; if not, the download is saved there for a later run to reuse."
  echo "  (default: \$FS_DOWNLOAD_CACHE, if set)"
  echo "--fs-pruned-cache-dir points at a directory for the pruned FreeSurfer install: if a valid"
  echo "  one is already there, the whole download+prune step is skipped."
  echo "  (default: \$FS_PRUNED_CACHE_DIR, if set)"
  echo "--py2app-venv points at a venv to create (if missing) and reuse for the py2app packaging"
  echo "  step, isolated from your normal dev venv, whose extra installed packages (matplotlib,"
  echo "  etc.) py2app's dependency scanner can otherwise trip over."
  echo "  (default: \$PY2APP_VENV, or tools/macos_build/.venv-py2app)"
  echo "--uv-cache-dir points at a directory for uv's download cache (the standalone python"
  echo "  distribution and the dependency wheels), so repeated builds and CI do not re-download"
  echo "  several hundred MB."
  echo "  (default: \$UV_CACHE_DIR, if set, else uv's own default)"
  echo "--checkpoints-dir points at a directory holding the network checkpoints. They are copied"
  echo "  into the package, so the installed FastSurfer needs no download on first run. If the"
  echo "  directory is missing or incomplete, the missing checkpoints are downloaded into it."
  echo "  (default: \$FASTSURFER_CHECKPOINTS_DIR, else <fastsurfer>/checkpoints)"
  echo
  exit
fi
ARCH_TYPE=$1 # chip architecture - arm or intel
shift

fs_download_cache="$FS_DOWNLOAD_CACHE"
fs_pruned_cache_dir="$FS_PRUNED_CACHE_DIR"
py2app_venv="$PY2APP_VENV"
uv_cache_dir="$UV_CACHE_DIR"
checkpoints_dir="$FASTSURFER_CHECKPOINTS_DIR"
while [[ "$#" -ge 1 ]] ; do
  case "$1" in
  --fs-download-cache) fs_download_cache=$2 ; shift ; shift ;;
  --fs-pruned-cache-dir) fs_pruned_cache_dir=$2 ; shift ; shift ;;
  --py2app-venv) py2app_venv=$2 ; shift ; shift ;;
  --uv-cache-dir) uv_cache_dir=$2 ; shift ; shift ;;
  --checkpoints-dir) checkpoints_dir=$2 ; shift ; shift ;;
  *) echo "Invalid argument $1" ; exit 1 ;;
  esac
done

if [[ -z "${BASH_SOURCE[0]}" ]]; then THIS_SCRIPT="$0"
else THIS_SCRIPT="${BASH_SOURCE[0]}"
fi
# resolve to an absolute path: the py2app step below runs inside a pushd, where a relative
# script/venv path (e.g. from `tools/macos_build/build_release_package.sh`) no longer resolves
build_dir=$(cd "$(dirname "$THIS_SCRIPT")" && pwd)
tools_dir=$(dirname "$build_dir")
py2app_venv="${py2app_venv:-$build_dir/.venv-py2app}"
case "$py2app_venv" in /*) ;; *) py2app_venv="$PWD/$py2app_venv" ;; esac

FASTSURFER_HOME=$(dirname "$tools_dir") # directory to fastsurfer
# version of the project
VERSION=$(python3 "$tools_dir/read_toml.py" --file "$FASTSURFER_HOME/pyproject.toml" --key project.version)
VERSION_NO_DOTS=${VERSION//./}
#version of the freesurfer
FREESURFER_VERSION=$(python3 "$tools_dir/read_toml.py" --file "$FASTSURFER_HOME/pyproject.toml" --key tool.freesurfer.version)
# freesurfer install url
URL_TO_FREESURFER_TEMP=$(python3 "$tools_dir/read_toml.py" --file "$FASTSURFER_HOME/pyproject.toml" --key tool.freesurfer.urls.macOS)
sub="{version}"
URL_TO_FREESURFER="${URL_TO_FREESURFER_TEMP//$sub/$FREESURFER_VERSION}"

ARCH_TYPE_NAME="arm64"
if [[ "$ARCH_TYPE" = "intel" ]] ; then ARCH_TYPE_NAME="x86_64" ; fi

RESOURCES_DIR="$build_dir/resources"
# name of the package displayed in the installer
PACKAGE_NAME=FastSurfer$VERSION_NO_DOTS-macos-darwin_${ARCH_TYPE_NAME}
# package identifier (f.e. com.mycompany.productid)
ID="org.deep-mi.FastSurfer.${VERSION_NO_DOTS}_${ARCH_TYPE_NAME}"
# install location for the content of the package
INSTALLATION_DIR="/Applications"
# raw package file to be created
OUTPUT_PKG="$build_dir/raw_package/$PACKAGE_NAME.pkg"
# installer to be created
INSTALLER_PKG="$build_dir/installer/$PACKAGE_NAME.pkg"

# create temporary folder to package and copy FastSurfer over
STAGED_DIR="$build_dir/FastSurferPackageContent"
FASTSURFER_TO_PACKAGE="$STAGED_DIR/FastSurfer$VERSION"
# start from an empty staging tree: an interrupted earlier build leaves its content behind, and
# neither the copy below nor pkgbuild removes anything, so leftovers would be packaged. Notably a
# run from before fs-pruned was nested leaves a "freesurfer" directory here, which would put the
# package back to installing into /Applications/freesurfer.
rm -rf "$STAGED_DIR"
mkdir -p "$FASTSURFER_TO_PACKAGE"
# top-level paths that are not needed to run FastSurfer and so stay out of the installed package.
# Note README.md, LICENSE and pyproject.toml must NOT be listed here: postinstall pip-installs
# $FASTSURFER_HOME and pyproject.toml references both files.
not_packaged=(
  # build-side only, never part of an install
  tools
  requirements.txt
  requirements.cpu.txt
  # development/CI material
  .github
  .codespellignore
  .dockerignore
  .gitignore
  CODE_OF_CONDUCT.md
  CONTRIBUTING.md
  # published at fastsurfer.org / only relevant in a source checkout (the build reads doc/ from
  # $FASTSURFER_HOME itself, not from this copy)
  doc
  test
  Tutorial
  Documentation
  env
)
# package git-tracked files only: everything else in the working tree (build artifacts, downloaded
# tarballs, scratch dirs, checkpoints fetched by a previous run) would otherwise end up in the
# installer and silently add gigabytes to it. Checkpoints are deliberately not shipped, postinstall
# downloads them on the user's machine.
if git -C "$FASTSURFER_HOME" rev-parse --git-dir > /dev/null 2>&1
then
  pathspecs=()
  for path in "${not_packaged[@]}" ; do pathspecs+=(":(exclude)$path") ; done
  git -C "$FASTSURFER_HOME" ls-files -z -- . "${pathspecs[@]}" \
    | rsync -av --from0 --files-from=- "$FASTSURFER_HOME/" "$FASTSURFER_TO_PACKAGE" || exit 1
else
  # not a git checkout (e.g. building from a source tarball): fall back to excluding by name
  excludes=(--exclude /.git)
  for path in "${not_packaged[@]}" ; do excludes+=(--exclude "/$path") ; done
  rsync -av --progress "$FASTSURFER_HOME/" "$FASTSURFER_TO_PACKAGE" "${excludes[@]}"
fi

# install pruned freesurfer (not a full install, so nested inside FastSurfer's own directory
# rather than the canonical /Applications/freesurfer, to avoid colliding with a real FreeSurfer install).
# --fs-pruned-cache-dir, if set (e.g. by CI, restored via actions/cache), lets install_fs_pruned.sh
# skip the download+prune entirely when a matching install is already there; either way, the result
# is copied into the staged package content, so the packaging steps below don't need to know about caching.
fs_pruned_where="${fs_pruned_cache_dir:-$FASTSURFER_TO_PACKAGE}"
download_cache_args=()
if [[ -n "$fs_download_cache" ]] ; then download_cache_args=(--fs-download-cache "$fs_download_cache") ; fi
"$tools_dir/build/install_fs_pruned.sh" "$fs_pruned_where" --url "$URL_TO_FREESURFER" --name fs-pruned "${download_cache_args[@]}"

if [[ ! -f "$fs_pruned_where/fs-pruned/build-stamp.txt" ]]
then
  echo "FreeSurfer install was unsuccessful!"
  exit 1
fi

if [[ -n "$fs_pruned_cache_dir" ]]
then
  mkdir -p "$FASTSURFER_TO_PACKAGE"
  # remove any stale fs-pruned first: cp -R copies *into* an existing destination dir instead of
  # replacing it, which would silently nest a leftover from an interrupted prior build
  rm -rf "$FASTSURFER_TO_PACKAGE/fs-pruned"
  cp -R "$fs_pruned_where/fs-pruned" "$FASTSURFER_TO_PACKAGE/"
fi

SCRIPTS_DIR="$tools_dir/macos_build/scripts" # directory with scripts executed during installation process (f.e. preinstall postinstall)
# the exact python that is bundled into the package. This must be an exact version, not a range:
# the bundled interpreter lives at python/bin/python$PYTHON_VERSION and its packages at
# python/lib/python$PYTHON_VERSION/site-packages, both referred to by name.
# tool.python.version is that exact value, shared with the docker build (see pyproject.toml).
PYTHON_VERSION=$(python3 "$tools_dir/read_toml.py" --file "$FASTSURFER_HOME/pyproject.toml" --key tool.python.version)

# where the package will live once installed, substituted into the installer scripts and the
# console script. The bundled python distribution does not need to know this path: it derives its
# prefix from the interpreter's own location, so it works wherever the installer puts it.
PATH_TO_FASTSURFER="$INSTALLATION_DIR/FastSurfer$VERSION"

# ============================ BUNDLED PYTHON ENVIRONMENT ==================================
# Ship a complete, self-contained python environment so installing needs no network and no
# pre-installed python: FreeSurfer's own installer works the same way. Homebrew's python cannot be
# bundled -- even a --copies venv built from it links against /opt/homebrew/Cellar/... and resolves
# its stdlib there -- so this uses a relocatable standalone CPython (python-build-standalone, the
# distribution uv manages), which only links against /usr/lib and system frameworks.
if ! command -v uv > /dev/null 2>&1
then
  echo "ERROR: uv not found, but it is required to fetch the standalone python and the" >&2
  echo "  dependencies for the bundled environment. Install it with 'brew install uv' or" >&2
  echo "  'curl -LsSf https://astral.sh/uv/install.sh | sh'." >&2
  exit 1
fi
if [[ -n "$uv_cache_dir" ]] ; then export UV_CACHE_DIR="$uv_cache_dir" ; fi

BUNDLED_PYTHON="$FASTSURFER_TO_PACKAGE/python"

echo "Fetching standalone python $PYTHON_VERSION ..."
# Keep the managed distribution in a build-local directory rather than uv's default: that default is
# shared with the developer's own uv installs, and this step copies a whole distribution, so the
# build has to know exactly which one it got.
UV_PYTHON_DIR="$build_dir/.uv-pythons"
UV_PYTHON_INSTALL_DIR="$UV_PYTHON_DIR" uv python install "$PYTHON_VERSION"
# Locate it by globbing the install directory instead of asking uv to search: the directory is
# build-local and holds exactly the distribution just installed, which is more predictable than
# relying on interpreter-discovery semantics (those also consider virtual environments and the
# system python, and change between uv releases).
standalone_root=""
for candidate in "$UV_PYTHON_DIR"/cpython-"$PYTHON_VERSION"*/ ; do
  if [[ -x "$candidate/bin/python$PYTHON_VERSION" ]] ; then standalone_root="${candidate%/}" ; fi
done
if [[ -z "$standalone_root" ]]
then
  echo "ERROR: uv installed no usable standalone python $PYTHON_VERSION under $UV_PYTHON_DIR." >&2
  echo "  Found: $(ls "$UV_PYTHON_DIR" 2>/dev/null | tr '\n' ' ')" >&2
  exit 1
fi
echo "  using $standalone_root"

echo "Bundling the python distribution ..."
rm -rf "$BUNDLED_PYTHON"
cp -R "$standalone_root" "$BUNDLED_PYTHON"
BUNDLED_INTERPRETER="$BUNDLED_PYTHON/bin/python$PYTHON_VERSION"

# Dependencies go straight into the bundled distribution's own site-packages, with no virtual
# environment in between. A venv would have to be un-picked afterwards: it records its location in
# pyvenv.cfg, its activate scripts and every console-script shebang, none of which survive the move
# from the staging directory to $PATH_TO_FASTSURFER. The distribution itself needs no such fixup --
# it derives its prefix from the interpreter's own location, so the whole tree can be moved
# anywhere. It also avoids a second, 18 MB copy of the interpreter inside the venv.
#
# uv marks distributions it manages with an EXTERNALLY-MANAGED file saying they "should not be
# modified", which is correct for the shared installation it keeps for the developer, but not for
# this private copy that is about to be shipped as one unit. Remove it in the copy only.
rm -f "$BUNDLED_PYTHON/lib/python$PYTHON_VERSION/EXTERNALLY-MANAGED"

echo "Installing dependencies into the bundled distribution ..."
# requirements.txt pins exact versions, so every build of a given commit ships the same packages.
# It is exported from the linux container, so fall back to resolving from pyproject.toml if a pin
# has no macOS wheel for this python.
if uv pip install --python "$BUNDLED_INTERPRETER" -r "$FASTSURFER_HOME/requirements.txt"
then
  # requirements.txt already pins whippersnappy, i.e. the whole of the [qc] extra
  echo "  installed from requirements.txt (pinned)"
else
  echo "  WARNING: requirements.txt did not resolve for macOS/python$PYTHON_VERSION," >&2
  echo "    falling back to resolving from pyproject.toml (versions are then build-date dependent)" >&2
  # [qc] pulls in whippersnappy, which run_fastsurfer.sh --qc_snap requires
  uv pip install --python "$BUNDLED_INTERPRETER" "$FASTSURFER_HOME[qc]"
fi

# FastSurfer itself is deliberately NOT installed into the environment: the package ships its own
# source tree at $PATH_TO_FASTSURFER and both run_fastsurfer.sh and the console put that on
# PYTHONPATH. Installing it as well would put a second, shadowed copy of every module in
# site-packages, which is how the console and the pipeline previously ended up importing different
# copies of the same module.

# ============================ BUNDLED CHECKPOINTS =========================================
# Ship the network weights, so a fresh install does not have to download ~300 MB on first run.
echo "Bundling checkpoints ..."
checkpoints_dir="${checkpoints_dir:-$FASTSURFER_HOME/checkpoints}"
mkdir -p "$checkpoints_dir"
# download_checkpoints.py is idempotent: it skips files that are already present, so an existing
# (or CI-cached) directory only fetches what is missing.
FASTSURFER_HOME="$FASTSURFER_HOME" PYTHONPATH="$FASTSURFER_HOME" "$BUNDLED_INTERPRETER" \
    "$FASTSURFER_HOME/FastSurferCNN/download_checkpoints.py" --all
rm -rf "$FASTSURFER_TO_PACKAGE/checkpoints"
cp -R "$checkpoints_dir" "$FASTSURFER_TO_PACKAGE/checkpoints"

# Retarget the bundled distribution from the staging directory to the install directory. The
# interpreter needs no help -- it derives its prefix from its own location -- but pip and uv write
# console scripts as /bin/sh wrappers that exec the interpreter by absolute path, so all ~57 of them
# (including pip itself and neuroreg's coreg/robreg) would otherwise exec a path that does not exist
# on the user's machine. This also strips compiled bytecode, which records source paths and cannot be
# rewritten; postinstall regenerates it in place.
# Unlike a virtual environment, this does not stop the distribution working in the staging tree, so
# it does not have to be the last step -- but keeping it here means the verification inside covers
# everything that came before.
echo "Retargeting the bundled python to the install prefix ..."
python3 "$build_dir/finalize_bundled_python.py" \
    --dist "$BUNDLED_PYTHON" \
    --from "$FASTSURFER_TO_PACKAGE" \
    --to "$PATH_TO_FASTSURFER"

# Assemble the installer scripts in a directory of their own. pkgbuild --scripts packages the whole
# directory it is given, so pointing it at the tracked source directory shipped the template and any
# stray AppleDouble (._*) files inside the installer, and forced the generated files to be written
# into (and then deleted from) a git-tracked directory.
PKG_SCRIPTS_DIR="$build_dir/pkg-scripts"
rm -rf "$PKG_SCRIPTS_DIR"
mkdir -p "$PKG_SCRIPTS_DIR"

# substitute values in postinstall script
sed -e "s|<fastsurfer_home_dir>|${PATH_TO_FASTSURFER}|g" \
    -e "s|<python_version>|${PYTHON_VERSION}|g" \
    < "$SCRIPTS_DIR/postinstall.sh.template" \
    > "$PKG_SCRIPTS_DIR/postinstall"
# postinstall calls link_fs.sh, so it has to travel with it
cp "$tools_dir/build/link_fs.sh" "$PKG_SCRIPTS_DIR/link_fs.sh"

chmod +x "$PKG_SCRIPTS_DIR/postinstall" "$PKG_SCRIPTS_DIR/link_fs.sh"
# Note: the installer's script archive will also contain AppleDouble (._postinstall, ._link_fs.sh)
# entries. That is unavoidable rather than an oversight: pkgbuild stores extended attributes that
# way, and macOS tags every file with com.apple.provenance, which `xattr -c` cannot remove. They are
# 163 bytes each and are ignored by the installer.

# assemble resources
mkdir -p "$RESOURCES_DIR"
cp "$FASTSURFER_HOME/doc/images/fastsurfer.png" "$RESOURCES_DIR"
cp "$FASTSURFER_HOME/LICENSE" "$RESOURCES_DIR/LICENSE.txt"
# final screen of the installer, registered as text/html by edit_distribution.py
sed -e "s|<fastsurfer>|FastSurfer${VERSION}|g" \
    < "$build_dir/conclusion.html.template" \
    > "$RESOURCES_DIR/conclusion.html"

# create fastsurfer applet
sed -e "s|<fastsurfer>|FastSurfer${VERSION}|g" \
    < "$build_dir/FastSurfer.py.template" \
    > "$build_dir/FastSurfer.py"

sed -e "s|<fastsurfer>|FastSurfer${VERSION}|g" \
    -e "s|<python_version>|${PYTHON_VERSION}|g" \
    < "$build_dir/macos_setup_fastsurfer.sh.template" \
    > "$build_dir/macos_setup_fastsurfer.sh"

mv "$build_dir/macos_setup_fastsurfer.sh" "$FASTSURFER_TO_PACKAGE/"

# isolated venv for py2app, kept separate from any dev venv: py2app's dependency scanner walks the
# whole environment it runs in, and unrelated packages there (e.g. matplotlib) can make it fail
if [[ ! -x "$py2app_venv/bin/python3" ]]
then
  echo "Creating isolated venv for py2app at $py2app_venv ..."
  python3 -m venv "$py2app_venv" || exit 1
fi
# checked separately from the venv itself, which may pre-exist (--py2app-venv) or be a leftover
# from an interrupted pip install, so its presence alone does not mean py2app is installed
if ! "$py2app_venv/bin/python3" -c "import py2app" > /dev/null 2>&1
then
  echo "Installing py2app into $py2app_venv ..."
  "$py2app_venv/bin/python3" -m pip install --upgrade pip || exit 1
  "$py2app_venv/bin/python3" -m pip install py2app || exit 1
fi

pushd "$build_dir" || exit 1
"$py2app_venv/bin/python3" "setup.py" py2app --iconfile "${RESOURCES_DIR:$((${#build_dir} + 1))}/fastsurfer.png"
popd || exit 1
mv "$build_dir/dist/FastSurfer.app" "$STAGED_DIR/FastSurfer$VERSION.app"

rm -f "$build_dir/FastSurfer.py"
chmod -R 755 "$STAGED_DIR"/*

# create raw package
mkdir -p "$build_dir/raw_package"
pkgbuild \
    --root "$STAGED_DIR" \
    --version "$VERSION" \
    --identifier "$ID" \
    --install-location "$INSTALLATION_DIR" \
    --scripts "$PKG_SCRIPTS_DIR" \
    "$OUTPUT_PKG"

# create distribution file template based on provided package
DISTRIBUTION_FILE="$RESOURCES_DIR/distribution.xml"

productbuild --synthesize --package "$OUTPUT_PKG" "$DISTRIBUTION_FILE"

# edit the distribution file
# set title to package name (f.e. package_name.pkg -> <title>package_name</title>)
python3 "$build_dir/edit_distribution.py" --file "$DISTRIBUTION_FILE" --title "$PACKAGE_NAME"

# create installer package
mkdir -p "$build_dir/installer"
productbuild \
    --distribution "$DISTRIBUTION_FILE" \
    --resources "$RESOURCES_DIR" \
    --package-path "$build_dir/raw_package" \
    "$INSTALLER_PKG"

# get rid of temporary folders. PKG_SCRIPTS_DIR and .uv-pythons are build-local, so nothing has to
# be cleaned out of the tracked source tree any more.
rm -rf "$STAGED_DIR" "$RESOURCES_DIR" "$build_dir/dist" "$build_dir/build" "$PKG_SCRIPTS_DIR"
